import json
from typing import List, Optional
from fastapi import APIRouter, Query
from app.models.schemas import (
    DatasetGenerateRequest,
    DatasetGenerateResponse,
    FinetuneJobRequest,
    LLMModelSpec,
    ProjectRequirements,
    RecommendationResponse,
    ScrapeStatusResponse,
)
from app.scrapers.manager import ScraperManager
from app.services.dataset_generator import DatasetGeneratorService
from app.services.finetune_manager import FinetuneManager
from app.services.lmstudio_client import LMStudioClient
from app.services.recommender import RecommenderService

router = APIRouter(prefix="/api")

scraper_manager = ScraperManager()
lmstudio_client = LMStudioClient()
recommender_service = RecommenderService(scraper_manager, lmstudio_client)
dataset_service = DatasetGeneratorService(scraper_manager, recommender_service, lmstudio_client)
finetune_manager = FinetuneManager(scraper_manager.db)


@router.get("/health")
async def health_check():
    lmstudio_up = await lmstudio_client.is_available()
    lmstudio_models = await lmstudio_client.list_loaded_models() if lmstudio_up else []
    catalog = scraper_manager.load_catalog()
    scrape_runs = scraper_manager.db.list_scrape_runs(limit=5)
    return {
        "status": "ok",
        "lmstudio_connected": lmstudio_up,
        "lmstudio_url": lmstudio_client.base_url,
        "lmstudio_loaded_models": lmstudio_models,
        "catalog_models_count": len(catalog),
        "recent_scrape_runs": scrape_runs,
    }


@router.get("/models", response_model=List[LLMModelSpec])
async def list_models(
    search: Optional[str] = Query(default=None),
    open_weights_only: bool = Query(default=False),
    max_vram_gb: Optional[float] = Query(default=None),
    source: Optional[str] = Query(default=None),
):
    catalog = scraper_manager.load_catalog()
    if not catalog:
        await scraper_manager.refresh_catalog()
    return scraper_manager.db.list_models(
        search=search,
        open_weights_only=open_weights_only,
        max_vram_gb=max_vram_gb,
        source=source,
    )


@router.post("/scrape", response_model=ScrapeStatusResponse)
async def trigger_scrape():
    return await scraper_manager.refresh_catalog()


@router.get("/scrape/history")
async def get_scrape_history():
    return scraper_manager.db.list_scrape_runs(limit=15)


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_llm(req: ProjectRequirements):
    return await recommender_service.recommend(req)


@router.get("/evaluations")
async def list_evaluations():
    return scraper_manager.db.list_evaluations(limit=20)


@router.post("/dataset/generate", response_model=DatasetGenerateResponse)
async def generate_finetune_dataset(req: DatasetGenerateRequest):
    return await dataset_service.generate_dataset(req)


@router.get("/dataset/samples")
async def get_dataset_samples(limit: int = Query(default=30, ge=1, le=200)):
    path = dataset_service.dataset_path
    if not path.exists():
        return {"total": 0, "dataset_path": str(path), "samples": []}
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return {
        "total": len(samples),
        "dataset_path": str(path),
        "samples": samples[:limit],
    }


@router.get("/finetune/status")
async def get_finetune_status():
    return finetune_manager.get_system_status()


@router.post("/finetune/run")
async def run_finetune_job(req: FinetuneJobRequest):
    return await finetune_manager.start_job(req)
