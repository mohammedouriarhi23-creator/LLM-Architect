import logging
from typing import List
import httpx
from app.models.schemas import LLMModelSpec
from app.scrapers.vram_utils import calculate_vram_footprints, infer_params_from_name

logger = logging.getLogger(__name__)


class OpenRouterLiveScraper:
    """
    Scrapes the live OpenRouter public catalog API (https://openrouter.ai/api/v1/models)
    which requires no API key and returns real-time context windows, per-token pricing,
    modalities, architectures, and open vs. proprietary model metadata.
    """

    OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

    OPEN_WEIGHT_PROVIDERS = {
        "meta-llama",
        "qwen",
        "deepseek",
        "mistralai",
        "google/gemma",
        "microsoft",
        "nousresearch",
        "01-ai",
        "cognitivecomputations",
        "nvidia",
        "allenai",
    }

    async def scrape(self, limit: int = 60) -> List[LLMModelSpec]:
        models: List[LLMModelSpec] = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(self.OPENROUTER_MODELS_URL)
                if resp.status_code != 200:
                    logger.warning("OpenRouter returned status %s", resp.status_code)
                    return []

                payload = resp.json()
                raw_list = payload.get("data", [])

                for item in raw_list[:limit]:
                    model_id = item.get("id", "")
                    if not model_id or ":" in model_id:
                        continue

                    name = item.get("name") or model_id
                    provider = model_id.split("/")[0] if "/" in model_id else "Cloud"
                    context_len = int(item.get("context_length") or 8192)

                    pricing = item.get("pricing") or {}
                    try:
                        prompt_price_per_tok = float(pricing.get("prompt") or 0.0)
                        completion_price_per_tok = float(pricing.get("completion") or 0.0)
                        input_1m = round(prompt_price_per_tok * 1_000_000, 4)
                        output_1m = round(completion_price_per_tok * 1_000_000, 4)
                    except (ValueError, TypeError):
                        input_1m, output_1m = 0.0, 0.0

                    params_b = infer_params_from_name(model_id) or infer_params_from_name(name)
                    is_open = any(
                        model_id.lower().startswith(prefix) for prefix in self.OPEN_WEIGHT_PROVIDERS
                    )

                    q4, q8, fp16 = calculate_vram_footprints(params_b, context_len) if is_open else (None, None, None)

                    arch_obj = item.get("architecture") or {}
                    modality = arch_obj.get("modality") or "text->text"
                    arch_type = arch_obj.get("tokenizer") or "Transformer"

                    # Estimate baseline capability tiers from parameter class / family
                    mmlu, coding, reasoning, math = self._estimate_benchmarks(
                        model_id.lower(), params_b, input_1m
                    )
                    avg = round((mmlu + coding + reasoning + math) / 4.0, 1)

                    tags = []
                    if is_open:
                        tags.append("open-source")
                    if params_b and params_b <= 4.0:
                        tags.extend(["low-vram", "unsloth-ready", "edge"])
                    elif params_b and params_b <= 9.0:
                        tags.extend(["unsloth-ready", "single-gpu"])
                    if "coder" in model_id.lower() or "codestral" in model_id.lower():
                        tags.append("coding")
                    if "r1" in model_id.lower() or "qwq" in model_id.lower() or "o1" in model_id.lower() or "o3" in model_id.lower():
                        tags.extend(["reasoning", "math"])
                    if context_len >= 100000:
                        tags.extend(["long-context", "rag"])
                    if "vision" in modality or "image" in modality:
                        tags.append("multimodal")

                    speed = 110.0 if (params_b and params_b <= 8) else 65.0

                    models.append(
                        LLMModelSpec(
                            id=model_id,
                            name=name,
                            provider=provider,
                            source="openrouter_live",
                            parameters_b=params_b,
                            context_length=context_len,
                            license="open-weights" if is_open else "proprietary",
                            open_weights=is_open,
                            mmlu_score=mmlu,
                            coding_score=coding,
                            reasoning_score=reasoning,
                            math_score=math,
                            average_benchmark=avg,
                            input_price_per_1m=input_1m,
                            output_price_per_1m=output_1m,
                            median_tokens_per_sec=speed,
                            ttft_seconds=0.38,
                            min_vram_gb=q4,
                            vram_q4_gb=q4,
                            vram_q8_gb=q8,
                            vram_fp16_gb=fp16,
                            architecture=arch_type,
                            modality=modality,
                            tags=list(dict.fromkeys(tags)),
                            url=f"https://openrouter.ai/{model_id}",
                        )
                    )
        except Exception as exc:
            logger.warning("OpenRouter live scrape error: %s", exc)

        return models

    def _estimate_benchmarks(
        self, model_id_lower: str, params_b: float | None, input_price: float
    ) -> tuple[float, float, float, float]:
        if "claude-3.5-sonnet" in model_id_lower or "claude-3-7-sonnet" in model_id_lower:
            return 88.7, 92.5, 90.0, 80.0
        if "gpt-4o" in model_id_lower and "mini" not in model_id_lower:
            return 88.7, 90.2, 87.6, 76.6
        if "deepseek-r1" in model_id_lower:
            if params_b and params_b <= 8:
                return 74.8, 79.2, 89.5, 88.1
            return 90.8, 91.5, 94.0, 93.2
        if "deepseek-chat" in model_id_lower or "deepseek-v3" in model_id_lower:
            return 88.5, 90.8, 89.1, 90.2
        if "coder" in model_id_lower or "codestral" in model_id_lower:
            base = 76.0 if (params_b and params_b >= 7) else 68.0
            return base, min(92.0, base + 12.0), base + 3.0, base + 1.0
        if params_b:
            if params_b <= 3.5:
                return 64.5, 63.0, 65.0, 58.0
            if params_b <= 9.0:
                return 72.5, 73.0, 74.0, 64.0
            if params_b <= 16.0:
                return 77.5, 79.0, 78.5, 72.0
            if params_b <= 35.0:
                return 82.0, 84.0, 83.0, 78.0
            return 85.5, 86.5, 86.0, 80.0
        if input_price >= 2.0:
            return 86.5, 88.0, 87.0, 79.0
        return 76.0, 77.0, 76.5, 70.0
