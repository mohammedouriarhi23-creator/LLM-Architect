from typing import List, Optional
from pydantic import BaseModel, Field


class LLMModelSpec(BaseModel):
    id: str
    name: str
    provider: str
    source: str = Field(description="e.g., huggingface_leaderboards, artificial_analysis, openrouter_live")
    parameters_b: Optional[float] = Field(default=None, description="Model size in billions of parameters")
    context_length: int = Field(default=8192, description="Max context window in tokens")
    license: str = Field(default="apache-2.0")
    open_weights: bool = Field(default=True)
    # Benchmarks (0-100 scale)
    mmlu_score: Optional[float] = None
    coding_score: Optional[float] = None
    reasoning_score: Optional[float] = None
    math_score: Optional[float] = None
    elo_arena: Optional[float] = None
    average_benchmark: Optional[float] = None
    # Performance & Pricing (from Artificial Analysis / OpenRouter / providers)
    input_price_per_1m: Optional[float] = Field(default=None, description="USD per 1M input tokens (0 for local)")
    output_price_per_1m: Optional[float] = Field(default=None, description="USD per 1M output tokens (0 for local)")
    median_tokens_per_sec: Optional[float] = Field(default=None, description="Output speed in tokens/sec")
    ttft_seconds: Optional[float] = Field(default=None, description="Time to first token in seconds")
    min_vram_gb: Optional[float] = Field(default=None, description="Estimated VRAM for 4-bit quantization")
    vram_q4_gb: Optional[float] = Field(default=None, description="VRAM required for Q4_K_M GGUF")
    vram_q8_gb: Optional[float] = Field(default=None, description="VRAM required for Q8_0 GGUF")
    vram_fp16_gb: Optional[float] = Field(default=None, description="VRAM required for FP16 weights")
    architecture: Optional[str] = Field(default="Transformer")
    modality: Optional[str] = Field(default="text->text")
    downloads: Optional[int] = Field(default=0)
    likes: Optional[int] = Field(default=0)
    tags: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    updated_at: Optional[str] = None


class ProjectRequirements(BaseModel):
    project_name: str = Field(default="My AI Project")
    project_description: str = Field(
        ...,
        description="Detailed description of the project, use case, and domain.",
    )
    primary_tasks: List[str] = Field(
        default_factory=lambda: ["coding", "reasoning"],
        description="e.g., coding, reasoning, rag, chat, math, extraction, agentic",
    )
    deployment_mode: str = Field(
        default="any",
        description="'local' (self-hosted/open-weights), 'cloud' (API), or 'any'",
    )
    max_vram_gb: Optional[float] = Field(
        default=8.0,
        description="Available GPU VRAM in GB if running locally",
    )
    max_budget_per_1m_tokens: Optional[float] = Field(
        default=5.0,
        description="Max acceptable blended USD cost per 1M tokens",
    )
    min_context_length: int = Field(default=8192)
    min_tokens_per_sec: Optional[float] = Field(default=25.0)
    monthly_input_tokens_m: float = Field(default=10.0, description="Estimated monthly input tokens in millions")
    monthly_output_tokens_m: float = Field(default=5.0, description="Estimated monthly output tokens in millions")
    require_open_source: bool = Field(default=False)
    use_lmstudio_advisor: bool = Field(
        default=True,
        description="Whether to query the fine-tuned model in LM Studio for natural language architectural advice",
    )


class QuantizationFit(BaseModel):
    q4_k_m_vram_gb: Optional[float] = None
    q8_0_vram_gb: Optional[float] = None
    fp16_vram_gb: Optional[float] = None
    fits_in_user_vram: bool = True
    recommended_quant: str = "Q4_K_M"


class CostEstimate(BaseModel):
    monthly_api_cost_usd: float = 0.0
    cost_per_1k_requests_usd: float = 0.0
    local_hardware_tier: str = "Consumer GPU (6-8GB VRAM)"


class RecommendedModel(BaseModel):
    model: LLMModelSpec
    match_score: float = Field(description="0-100 compatibility score")
    task_score: float = Field(default=75.0)
    hardware_score: float = Field(default=80.0)
    cost_score: float = Field(default=80.0)
    strengths: List[str]
    tradeoffs: List[str]
    deployment_fit: str
    quantization: Optional[QuantizationFit] = None
    cost_estimate: Optional[CostEstimate] = None


class RecommendationResponse(BaseModel):
    evaluation_id: Optional[int] = None
    recommendations: List[RecommendedModel]
    advisor_analysis: str = Field(
        description="Architectural justification from fine-tuned LM Studio model or heuristic engine"
    )
    used_lmstudio: bool = False


class ScrapeStatusResponse(BaseModel):
    run_id: Optional[int] = None
    total_models: int
    sources_scraped: List[str]
    source_counts: dict = Field(default_factory=dict)
    catalog_path: str
    models: List[LLMModelSpec]


class FinetuneJobRequest(BaseModel):
    base_model: str = Field(default="qwen-2.5-3b", description="qwen-2.5-3b or llama-3.2-3b")
    max_steps: int = Field(default=60, ge=5, le=1000)
    lora_r: int = Field(default=16)
    quantization: str = Field(default="q4_k_m")
    dry_run: bool = Field(default=False)


class DatasetGenerateRequest(BaseModel):
    num_samples: int = Field(default=20, ge=1, le=500)
    use_lmstudio: bool = Field(
        default=True,
        description="Use local LM Studio server to synthesize rich training pairs from scraped models",
    )


class DatasetGenerateResponse(BaseModel):
    samples_generated: int
    dataset_path: str
    preview: List[dict]
