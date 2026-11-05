export interface LLMModelSpec {
  id: string;
  name: string;
  provider: string;
  source: string;
  parameters_b?: number | null;
  context_length: number;
  license: string;
  open_weights: boolean;
  mmlu_score?: number | null;
  coding_score?: number | null;
  reasoning_score?: number | null;
  math_score?: number | null;
  elo_arena?: number | null;
  average_benchmark?: number | null;
  input_price_per_1m?: number | null;
  output_price_per_1m?: number | null;
  median_tokens_per_sec?: number | null;
  ttft_seconds?: number | null;
  min_vram_gb?: number | null;
  vram_q4_gb?: number | null;
  vram_q8_gb?: number | null;
  vram_fp16_gb?: number | null;
  architecture?: string | null;
  modality?: string | null;
  downloads?: number | null;
  likes?: number | null;
  tags: string[];
  url?: string | null;
  updated_at?: string | null;
}

export interface ProjectRequirements {
  project_name: string;
  project_description: string;
  primary_tasks: string[];
  deployment_mode: "local" | "cloud" | "any";
  max_vram_gb?: number | null;
  max_budget_per_1m_tokens?: number | null;
  min_context_length: number;
  min_tokens_per_sec?: number | null;
  monthly_input_tokens_m: number;
  monthly_output_tokens_m: number;
  require_open_source: boolean;
  use_lmstudio_advisor: boolean;
}

export interface QuantizationFit {
  q4_k_m_vram_gb?: number | null;
  q8_0_vram_gb?: number | null;
  fp16_vram_gb?: number | null;
  fits_in_user_vram: boolean;
  recommended_quant: string;
}

export interface CostEstimate {
  monthly_api_cost_usd: number;
  cost_per_1k_requests_usd: number;
  local_hardware_tier: string;
}

export interface RecommendedModel {
  model: LLMModelSpec;
  match_score: number;
  task_score: number;
  hardware_score: number;
  cost_score: number;
  strengths: string[];
  tradeoffs: string[];
  deployment_fit: string;
  quantization?: QuantizationFit | null;
  cost_estimate?: CostEstimate | null;
}

export interface RecommendationResponse {
  evaluation_id?: number | null;
  recommendations: RecommendedModel[];
  advisor_analysis: string;
  used_lmstudio: boolean;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Failed to connect to backend");
  return res.json();
}

export async function fetchModels(params?: {
  search?: string;
  open_weights_only?: boolean;
  max_vram_gb?: number;
  source?: string;
}): Promise<LLMModelSpec[]> {
  const qs = new URLSearchParams();
  if (params?.search) qs.set("search", params.search);
  if (params?.open_weights_only) qs.set("open_weights_only", "true");
  if (params?.max_vram_gb) qs.set("max_vram_gb", String(params.max_vram_gb));
  if (params?.source) qs.set("source", params.source);
  const queryStr = qs.toString() ? `?${qs.toString()}` : "";
  const res = await fetch(`${API_BASE}/models${queryStr}`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function triggerScrape() {
  const res = await fetch(`${API_BASE}/scrape`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to run scraper");
  return res.json();
}

export async function requestRecommendation(
  payload: ProjectRequirements
): Promise<RecommendationResponse> {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to get recommendation");
  return res.json();
}

export async function fetchEvaluations() {
  const res = await fetch(`${API_BASE}/evaluations`);
  if (!res.ok) throw new Error("Failed to load evaluations");
  return res.json();
}

export async function generateDataset(numSamples: number, useLmstudio: boolean) {
  const res = await fetch(`${API_BASE}/dataset/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_samples: numSamples, use_lmstudio: useLmstudio }),
  });
  if (!res.ok) throw new Error("Failed to generate dataset");
  return res.json();
}

export async function fetchDatasetSamples(limit = 30) {
  const res = await fetch(`${API_BASE}/dataset/samples?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to load dataset samples");
  return res.json();
}

export async function fetchFinetuneStatus() {
  const res = await fetch(`${API_BASE}/finetune/status`);
  if (!res.ok) throw new Error("Failed to load finetune status");
  return res.json();
}

export async function triggerFinetuneJob(payload: {
  base_model: string;
  max_steps: number;
  lora_r: number;
  quantization: string;
  dry_run: boolean;
}) {
  const res = await fetch(`${API_BASE}/finetune/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to start finetune job");
  return res.json();
}
