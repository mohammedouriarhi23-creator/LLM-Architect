import pytest
from app.models.schemas import ProjectRequirements
from app.scrapers.manager import ScraperManager
from app.services.dataset_generator import DatasetGeneratorService
from app.models.schemas import DatasetGenerateRequest
from app.services.recommender import RecommenderService


@pytest.mark.asyncio
async def test_scrape_and_recommend_local_low_vram(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    manager = ScraperManager(catalog_path=catalog_file)
    status = await manager.refresh_catalog()

    assert status.total_models >= 15
    assert catalog_file.exists()
    assert status.run_id is not None

    # Verify SQLite filtering works
    low_vram_models = manager.db.list_models(open_weights_only=True, max_vram_gb=6.5)
    assert len(low_vram_models) >= 3

    recommender = RecommenderService(scraper_manager=manager)
    req = ProjectRequirements(
        project_name="Low VRAM Coding Helper",
        project_description="Local coding assistant for an 8GB laptop using Unsloth and LM Studio.",
        primary_tasks=["coding", "reasoning"],
        deployment_mode="local",
        max_vram_gb=6.5,
        max_budget_per_1m_tokens=0.0,
        min_context_length=16384,
        require_open_source=True,
        use_lmstudio_advisor=False,
    )
    res = await recommender.recommend(req)
    assert len(res.recommendations) > 0
    assert res.evaluation_id is not None
    top = res.recommendations[0]
    assert top.model.open_weights is True
    assert top.quantization is not None
    assert top.cost_estimate is not None
    assert top.model.min_vram_gb is not None and top.model.min_vram_gb <= 6.5


@pytest.mark.asyncio
async def test_dataset_generation(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    manager = ScraperManager(catalog_path=catalog_file)
    await manager.refresh_catalog()

    gen = DatasetGeneratorService(scraper_manager=manager)
    gen.dataset_path = tmp_path / "train.jsonl"
    resp = await gen.generate_dataset(DatasetGenerateRequest(num_samples=4, use_lmstudio=False))

    assert resp.samples_generated == 4
    assert gen.dataset_path.exists()
    assert len(resp.preview) == 3
