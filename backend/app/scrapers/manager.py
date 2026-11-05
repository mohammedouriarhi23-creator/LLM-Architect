import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from app.config import settings
from app.database import Database
from app.models.schemas import LLMModelSpec, ScrapeStatusResponse
from app.scrapers.artificial_analysis_scraper import ArtificialAnalysisScraper
from app.scrapers.huggingface_scraper import HuggingFaceScraper
from app.scrapers.openrouter_scraper import OpenRouterLiveScraper


class ScraperManager:
    def __init__(self, catalog_path: Path | None = None, db: Optional[Database] = None):
        self.catalog_path = catalog_path or (settings.data_dir / "models_catalog.json")
        self.db = db or Database(
            db_path=(self.catalog_path.parent / "llmarchitect.db") if catalog_path else None
        )
        self.hf_scraper = HuggingFaceScraper()
        self.aa_scraper = ArtificialAnalysisScraper()
        self.or_scraper = OpenRouterLiveScraper()

    async def refresh_catalog(self) -> ScrapeStatusResponse:
        started_at = datetime.now(timezone.utc).isoformat()
        hf_models = await self.hf_scraper.scrape(limit=15)
        aa_models = await self.aa_scraper.scrape()
        or_models = await self.or_scraper.scrape(limit=40)

        # Merge with priority: Curated/HF & Artificial Analysis benchmarks enrich OpenRouter live data
        merged: dict[str, LLMModelSpec] = {}
        for m in or_models + hf_models + aa_models:
            if m.id in merged:
                prev = merged[m.id]
                merged[m.id] = prev.model_copy(
                    update={
                        "mmlu_score": m.mmlu_score or prev.mmlu_score,
                        "coding_score": m.coding_score or prev.coding_score,
                        "reasoning_score": m.reasoning_score or prev.reasoning_score,
                        "math_score": m.math_score or prev.math_score,
                        "median_tokens_per_sec": m.median_tokens_per_sec or prev.median_tokens_per_sec,
                        "ttft_seconds": m.ttft_seconds or prev.ttft_seconds,
                        "input_price_per_1m": (
                            m.input_price_per_1m
                            if m.input_price_per_1m is not None
                            else prev.input_price_per_1m
                        ),
                        "output_price_per_1m": (
                            m.output_price_per_1m
                            if m.output_price_per_1m is not None
                            else prev.output_price_per_1m
                        ),
                        "min_vram_gb": m.min_vram_gb or prev.min_vram_gb,
                        "vram_q4_gb": m.vram_q4_gb or prev.vram_q4_gb,
                        "vram_q8_gb": m.vram_q8_gb or prev.vram_q8_gb,
                        "vram_fp16_gb": m.vram_fp16_gb or prev.vram_fp16_gb,
                    }
                )
            else:
                merged[m.id] = m

        model_list = list(merged.values())
        self.save_catalog(model_list)
        self.db.upsert_models(model_list)

        completed_at = datetime.now(timezone.utc).isoformat()
        source_counts = {
            "huggingface_leaderboards": len(hf_models),
            "artificial_analysis": len(aa_models),
            "openrouter_live": len(or_models),
        }
        run_id = self.db.record_scrape_run(
            started_at=started_at,
            completed_at=completed_at,
            sources=list(source_counts.keys()),
            models_discovered=len(model_list),
            status="success",
            details=source_counts,
        )

        return ScrapeStatusResponse(
            run_id=run_id,
            total_models=len(model_list),
            sources_scraped=list(source_counts.keys()),
            source_counts=source_counts,
            catalog_path=str(self.catalog_path),
            models=model_list,
        )

    def save_catalog(self, models: List[LLMModelSpec]) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [m.model_dump() for m in models]
        self.catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_catalog(self) -> List[LLMModelSpec]:
        db_models = self.db.list_models()
        if db_models:
            return db_models
        if not self.catalog_path.exists():
            return []
        raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        models = [LLMModelSpec(**item) for item in raw]
        self.db.upsert_models(models)
        return models
