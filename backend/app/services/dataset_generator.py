import json
from pathlib import Path
from typing import List
from app.config import settings
from app.models.schemas import (
    DatasetGenerateRequest,
    DatasetGenerateResponse,
    ProjectRequirements,
)
from app.scrapers.manager import ScraperManager
from app.services.lmstudio_client import LMStudioClient
from app.services.recommender import RecommenderService, SYSTEM_ADVISOR_PROMPT


SYNTHETIC_PROJECT_TEMPLATES: List[dict] = [
    {
        "project_name": "Local Code Assistant IDE Plugin",
        "project_description": "Offline coding autocomplete and refactoring assistant running on a developer laptop with an RTX 4060 8GB GPU.",
        "primary_tasks": ["coding", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 6.5,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 32768,
        "require_open_source": True,
    },
    {
        "project_name": "Edge Customer Support Triage Bot",
        "project_description": "Fast ticket classification and JSON extraction running on consumer 6GB VRAM hardware with sub-second latency.",
        "primary_tasks": ["extraction", "chat"],
        "deployment_mode": "local",
        "max_vram_gb": 4.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 8192,
        "require_open_source": True,
    },
    {
        "project_name": "Enterprise Multi-Repo RAG & Legal Document Analyzer",
        "project_description": "Analyze 150-page contracts and cross-reference clauses with high accuracy via cloud API.",
        "primary_tasks": ["rag", "reasoning", "extraction"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 10.0,
        "min_context_length": 128000,
        "require_open_source": False,
    },
    {
        "project_name": "High-Volume E-Commerce Product Tagging Pipeline",
        "project_description": "Process 5 million product descriptions daily at minimal API cost and maximum token/sec throughput.",
        "primary_tasks": ["extraction", "chat"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 0.50,
        "min_context_length": 16384,
        "require_open_source": False,
    },
    {
        "project_name": "Automated Theorem & Competitive Math Tutor",
        "project_description": "Step-by-step chain-of-thought mathematical proof assistant running on a single 8GB GPU.",
        "primary_tasks": ["math", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 8.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 32768,
        "require_open_source": True,
    },
    {
        "project_name": "Autonomous Multi-Step Software Engineering Agent",
        "project_description": "Complex multi-file bug fixing and tool-calling agent with top-tier SWE-bench / coding performance.",
        "primary_tasks": ["coding", "agentic", "reasoning"],
        "deployment_mode": "any",
        "max_vram_gb": 16.0,
        "max_budget_per_1m_tokens": 15.0,
        "min_context_length": 128000,
        "require_open_source": False,
    },
]


class DatasetGeneratorService:
    """
    Builds ShareGPT / ChatML JSONL fine-tuning datasets from scraped Hugging Face &
    Artificial Analysis benchmarks, optionally enriched by LM Studio local inference,
    ready to be consumed by `finetuning/train_unsloth.py`.
    """

    def __init__(
        self,
        scraper_manager: ScraperManager | None = None,
        recommender: RecommenderService | None = None,
        lmstudio: LMStudioClient | None = None,
    ):
        self.scraper_manager = scraper_manager or ScraperManager()
        self.recommender = recommender or RecommenderService(self.scraper_manager)
        self.lmstudio = lmstudio or LMStudioClient()
        self.dataset_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "finetuning"
            / "datasets"
            / "llm_advisor_train.jsonl"
        )

    async def generate_dataset(self, req: DatasetGenerateRequest) -> DatasetGenerateResponse:
        catalog = self.scraper_manager.load_catalog()
        if not catalog:
            status = await self.scraper_manager.refresh_catalog()
            catalog = status.models

        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        samples: List[dict] = []

        for idx in range(req.num_samples):
            template = SYNTHETIC_PROJECT_TEMPLATES[idx % len(SYNTHETIC_PROJECT_TEMPLATES)].copy()
            # Vary VRAM / budget slightly across synthetic samples for diversity
            if template.get("max_vram_gb"):
                template["max_vram_gb"] = round(template["max_vram_gb"] + ((idx % 3) - 1) * 0.5, 1)

            proj_req = ProjectRequirements(
                **template,
                use_lmstudio_advisor=req.use_lmstudio,
            )
            rec_resp = await self.recommender.recommend(proj_req)
            top = rec_resp.recommendations[:3]

            user_instruction = (
                f"Recommend the optimal LLM for this project:\n"
                f"- Project Name: {proj_req.project_name}\n"
                f"- Use Case: {proj_req.project_description}\n"
                f"- Primary Tasks: {', '.join(proj_req.primary_tasks)}\n"
                f"- Deployment: {proj_req.deployment_mode} (Max GPU VRAM: {proj_req.max_vram_gb} GB)\n"
                f"- Max Cost per 1M tokens: ${proj_req.max_budget_per_1m_tokens}\n"
                f"- Min Context Window: {proj_req.min_context_length} tokens\n"
                f"- Open-Weights Required: {proj_req.require_open_source}"
            )

            ranking_block = "\n".join(
                [
                    f"{i + 1}. **{r.model.name}** (`{r.model.id}`) — Score: {r.match_score}% | "
                    f"Deployment: {r.deployment_fit} | Strengths: {', '.join(r.strengths)}"
                    for i, r in enumerate(top)
                ]
            )
            assistant_response = (
                f"Based on current Hugging Face Leaderboard and Artificial Analysis benchmarks, "
                f"here are the top recommended models:\n\n{ranking_block}\n\n"
                f"{rec_resp.advisor_analysis}"
            )

            sample = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_ADVISOR_PROMPT},
                    {"from": "human", "value": user_instruction},
                    {"from": "gpt", "value": assistant_response},
                ],
                "metadata": {
                    "top_model_id": top[0].model.id if top else None,
                    "synthesized_with_lmstudio": rec_resp.used_lmstudio,
                },
            }
            samples.append(sample)

        with self.dataset_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        return DatasetGenerateResponse(
            samples_generated=len(samples),
            dataset_path=str(self.dataset_path),
            preview=samples[:3],
        )
