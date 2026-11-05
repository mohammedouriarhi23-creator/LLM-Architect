from typing import List
from app.models.schemas import (
    CostEstimate,
    LLMModelSpec,
    ProjectRequirements,
    QuantizationFit,
    RecommendationResponse,
    RecommendedModel,
)
from app.scrapers.manager import ScraperManager
from app.scrapers.vram_utils import calculate_vram_footprints
from app.services.lmstudio_client import LMStudioClient


SYSTEM_ADVISOR_PROMPT = """You are LLMarchitect, an expert AI systems architect fine-tuned with Unsloth to recommend the optimal Large Language Model for software projects.
Given the user's project constraints (VRAM, budget, latency, context window, tasks) and the top candidate models scraped from Hugging Face Leaderboards, OpenRouter, and Artificial Analysis, provide:
1. Why the #1 model is the best architectural choice.
2. Quantization / deployment advice (e.g., Q4_K_M / Q8_0 GGUF on LM Studio vs Cloud API).
3. Monthly Total Cost of Ownership (TCO) and key trade-offs with the runner-up alternatives."""


class RecommenderService:
    def __init__(self, scraper_manager: ScraperManager | None = None, lmstudio: LMStudioClient | None = None):
        self.scraper_manager = scraper_manager or ScraperManager()
        self.lmstudio = lmstudio or LMStudioClient()

    def evaluate_candidate(
        self, model: LLMModelSpec, req: ProjectRequirements
    ) -> RecommendedModel | None:
        score = 50.0
        task_score = 65.0
        hardware_score = 75.0
        cost_score = 75.0
        strengths: List[str] = []
        tradeoffs: List[str] = []

        q4 = model.vram_q4_gb or model.min_vram_gb
        q8 = model.vram_q8_gb
        fp16 = model.vram_fp16_gb
        if model.parameters_b and not q4:
            q4, q8, fp16 = calculate_vram_footprints(model.parameters_b, model.context_length)

        # 1. Deployment & VRAM Constraints
        fits_vram = True
        rec_quant = "Cloud Managed API"
        if req.deployment_mode == "local":
            if not model.open_weights:
                return None
            if req.max_vram_gb is not None and q4 is not None:
                if fp16 and fp16 <= req.max_vram_gb:
                    rec_quant = "FP16 / BF16 Native"
                    hardware_score = 98.0
                    score += 20.0
                    strengths.append(f"Runs in full FP16 (~{fp16} GB) within your {req.max_vram_gb} GB VRAM")
                elif q8 and q8 <= req.max_vram_gb:
                    rec_quant = "Q8_0 GGUF (Near-lossless)"
                    hardware_score = 94.0
                    score += 19.0
                    strengths.append(f"Fits near-lossless Q8_0 GGUF (~{q8} GB) in {req.max_vram_gb} GB VRAM")
                elif q4 <= req.max_vram_gb:
                    rec_quant = "Q4_K_M GGUF (Optimal 4-bit)"
                    hardware_score = 88.0
                    score += 18.0
                    strengths.append(
                        f"Fits in local GPU memory (~{q4} GB VRAM in Q4_K_M GGUF vs {req.max_vram_gb} GB limit)"
                    )
                else:
                    fits_vram = False
                    rec_quant = "Requires CPU Offload / IQ3_XS"
                    hardware_score = 30.0
                    score -= 35.0
                    tradeoffs.append(
                        f"Requires ~{q4} GB VRAM (Q4_K_M), exceeding your {req.max_vram_gb} GB target"
                    )
        elif req.deployment_mode == "cloud":
            if model.input_price_per_1m and model.input_price_per_1m > 0:
                score += 8.0
                hardware_score = 95.0
            else:
                strengths.append("Can be self-hosted or accessed via serverless open-weight providers")
        else:
            if model.open_weights and q4:
                rec_quant = "Q4_K_M GGUF or Cloud API"

        if req.require_open_source and not model.open_weights:
            score -= 40.0
            tradeoffs.append("Closed-weights proprietary license")
        elif model.open_weights:
            score += 6.0
            strengths.append(f"Open-weights ({model.license}) — fine-tunable with Unsloth")

        # 2. Context Length Check
        if model.context_length >= req.min_context_length:
            score += 8.0
            if model.context_length >= 128000:
                strengths.append(f"Large {model.context_length // 1000}k token context window")
        else:
            score -= 20.0
            tradeoffs.append(f"Context window ({model.context_length}) is below required {req.min_context_length}")

        # 3. Task Alignment & Benchmarks
        task_weights = [t.lower() for t in req.primary_tasks]
        task_accum = []
        if "coding" in task_weights and model.coding_score:
            score += (model.coding_score - 60.0) * 0.45
            task_accum.append(model.coding_score)
            if model.coding_score >= 80:
                strengths.append(f"Strong coding benchmark ({model.coding_score}%)")
        if ("reasoning" in task_weights or "agentic" in task_weights) and model.reasoning_score:
            score += (model.reasoning_score - 60.0) * 0.45
            task_accum.append(model.reasoning_score)
            if model.reasoning_score >= 80:
                strengths.append(f"High reasoning capability ({model.reasoning_score}%)")
        if "math" in task_weights and model.math_score:
            score += (model.math_score - 55.0) * 0.40
            task_accum.append(model.math_score)
            if model.math_score >= 75:
                strengths.append(f"Strong mathematical accuracy ({model.math_score}%)")
        if model.average_benchmark:
            score += (model.average_benchmark - 60.0) * 0.25
            task_accum.append(model.average_benchmark)

        if task_accum:
            task_score = round(min(99.0, sum(task_accum) / len(task_accum)), 1)

        desc_lower = req.project_description.lower()
        for tag in model.tags:
            if tag.lower() in desc_lower or tag.lower() in task_weights:
                score += 3.5

        # 4. Budget, TCO & Speed
        in_price = model.input_price_per_1m or 0.0
        out_price = model.output_price_per_1m or 0.0
        blended_cost = (in_price + out_price) / 2.0
        monthly_cost = round(
            req.monthly_input_tokens_m * in_price + req.monthly_output_tokens_m * out_price, 2
        )
        cost_per_1k_req = round((in_price * 0.8 + out_price * 0.4) / 1000.0 * 1000.0, 4)

        if req.max_budget_per_1m_tokens is not None:
            if blended_cost <= req.max_budget_per_1m_tokens:
                score += 6.0
                cost_score = 95.0 if blended_cost == 0 else max(65.0, round(95.0 - blended_cost * 5, 1))
                if blended_cost == 0.0:
                    strengths.append("Zero token API cost when run locally on LM Studio")
                elif blended_cost < 1.0:
                    strengths.append(f"Cost-efficient API (~${blended_cost:.2f}/1M tokens, ~${monthly_cost}/mo)")
            else:
                cost_score = 35.0
                score -= 18.0
                tradeoffs.append(
                    f"Blended API price (${blended_cost:.2f}/1M) exceeds budget (${req.max_budget_per_1m_tokens:.2f}/1M)"
                )

        if req.min_tokens_per_sec and model.median_tokens_per_sec:
            if model.median_tokens_per_sec >= req.min_tokens_per_sec:
                score += 5.0
                strengths.append(f"Fast throughput (~{model.median_tokens_per_sec:.0f} tokens/sec)")
            else:
                tradeoffs.append(f"Throughput ({model.median_tokens_per_sec:.0f} tok/s) below target")

        bounded_score = max(0.0, min(99.5, round(score, 1)))
        if bounded_score <= 10.0:
            return None

        if model.open_weights and q4 and q4 <= 8.0:
            hw_tier = "Consumer GPU (6–8GB VRAM: RTX 3060/4060 / Apple M1-M4)"
            deployment_fit = f"Local LM Studio ({rec_quant}, ~{q4}GB VRAM) + Unsloth QLoRA"
        elif model.open_weights and q4 and q4 <= 24.0:
            hw_tier = "Prosumer GPU (12–24GB VRAM: RTX 3090/4090 / A10G)"
            deployment_fit = f"Workstation LM Studio (~{q4}GB VRAM) or Cloud Serverless"
        elif model.open_weights:
            hw_tier = "Multi-GPU / Enterprise Node (A100/H100)"
            deployment_fit = f"Dedicated GPU Cluster (~{q4}GB VRAM) or OpenRouter API"
        else:
            hw_tier = "Cloud API Only (Zero Local VRAM)"
            deployment_fit = f"Managed Cloud API ({model.provider})"

        if not strengths:
            strengths.append("Balanced general-purpose performance")
        if not tradeoffs:
            tradeoffs.append("Standard compute/quality trade-off for its parameter class")

        return RecommendedModel(
            model=model,
            match_score=bounded_score,
            task_score=task_score,
            hardware_score=hardware_score,
            cost_score=cost_score,
            strengths=strengths[:4],
            tradeoffs=tradeoffs[:3],
            deployment_fit=deployment_fit,
            quantization=QuantizationFit(
                q4_k_m_vram_gb=q4,
                q8_0_vram_gb=q8,
                fp16_vram_gb=fp16,
                fits_in_user_vram=fits_vram,
                recommended_quant=rec_quant,
            ),
            cost_estimate=CostEstimate(
                monthly_api_cost_usd=monthly_cost,
                cost_per_1k_requests_usd=cost_per_1k_req,
                local_hardware_tier=hw_tier,
            ),
        )

    async def recommend(self, req: ProjectRequirements) -> RecommendationResponse:
        catalog = self.scraper_manager.load_catalog()
        if not catalog:
            status = await self.scraper_manager.refresh_catalog()
            catalog = status.models

        scored: List[RecommendedModel] = []
        for model in catalog:
            rec = self.evaluate_candidate(model, req)
            if rec is not None:
                scored.append(rec)

        scored.sort(key=lambda x: x.match_score, reverse=True)
        top_picks = scored[:6]

        advisor_text = ""
        used_lmstudio = False

        if req.use_lmstudio_advisor and top_picks:
            candidates_summary = "\n".join(
                [
                    f"{idx + 1}. {c.model.name} ({c.model.id}) - Match: {c.match_score}%, "
                    f"VRAM(Q4): {c.model.min_vram_gb or 'Cloud'}GB, Context: {c.model.context_length}, "
                    f"Coding: {c.model.coding_score}, Reasoning: {c.model.reasoning_score}, "
                    f"Monthly Cost: ${c.cost_estimate.monthly_api_cost_usd if c.cost_estimate else 0}, "
                    f"Strengths: {', '.join(c.strengths)}"
                    for idx, c in enumerate(top_picks[:3])
                ]
            )
            user_prompt = (
                f"Project: {req.project_name}\n"
                f"Description: {req.project_description}\n"
                f"Primary Tasks: {', '.join(req.primary_tasks)}\n"
                f"Deployment Mode: {req.deployment_mode} (Max VRAM: {req.max_vram_gb} GB)\n"
                f"Max Budget/1M tokens: ${req.max_budget_per_1m_tokens}\n\n"
                f"Top Scraped Candidates:\n{candidates_summary}\n\n"
                "Provide a concise architectural recommendation explaining which LLM to pick and how to deploy or fine-tune it."
            )
            llm_reply = await self.lmstudio.chat_completion(
                system_prompt=SYSTEM_ADVISOR_PROMPT,
                user_prompt=user_prompt,
            )
            if llm_reply:
                advisor_text = llm_reply
                used_lmstudio = True

        if not advisor_text:
            if top_picks:
                best = top_picks[0]
                runner_up = top_picks[1] if len(top_picks) > 1 else None
                q_info = best.quantization.recommended_quant if best.quantization else "Q4_K_M"
                tco = best.cost_estimate.monthly_api_cost_usd if best.cost_estimate else 0.0
                advisor_text = (
                    f"### Primary Architectural Recommendation: **{best.model.name}** (`{best.model.id}`)\n\n"
                    f"- **Overall Compatibility**: **{best.match_score}%** (Task Fit: {best.task_score}%, Hardware Fit: {best.hardware_score}%, Cost Fit: {best.cost_score}%)\n"
                    f"- **Deployment & Quantization**: {best.deployment_fit} — Recommended Quantization: **{q_info}**\n"
                    f"- **Estimated Monthly TCO**: **${tco:.2f}/month** (at {req.monthly_input_tokens_m}M input / {req.monthly_output_tokens_m}M output tokens)\n"
                    f"- **Key Strengths**: {'; '.join(best.strengths)}.\n"
                )
                if runner_up:
                    advisor_text += (
                        f"\n### Runner-Up Alternative: **{runner_up.model.name}** ({runner_up.match_score}%)\n"
                        f"- **When to switch**: Choose **{runner_up.model.name}** (`{runner_up.model.id}`) if you want to trade `{best.tradeoffs[0]}` "
                        f"for `{runner_up.strengths[0]}`."
                    )
            else:
                advisor_text = "No models matched the strict constraints. Try increasing Max VRAM or relaxing open-source filters."

        resp = RecommendationResponse(
            recommendations=top_picks,
            advisor_analysis=advisor_text,
            used_lmstudio=used_lmstudio,
        )
        eval_id = self.scraper_manager.db.save_evaluation(
            project_name=req.project_name,
            requirements=req.model_dump(),
            top_model_id=top_picks[0].model.id if top_picks else None,
            top_match_score=top_picks[0].match_score if top_picks else None,
            used_lmstudio=used_lmstudio,
            response_payload=resp.model_dump(),
        )
        resp.evaluation_id = eval_id
        return resp
