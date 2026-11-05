"use client";

import React, { useEffect, useState } from "react";
import {
  fetchDatasetSamples,
  fetchEvaluations,
  fetchFinetuneStatus,
  fetchHealth,
  fetchModels,
  generateDataset,
  LLMModelSpec,
  ProjectRequirements,
  RecommendationResponse,
  requestRecommendation,
  triggerFinetuneJob,
  triggerScrape,
} from "@/lib/api";

const AVAILABLE_TASKS = [
  "coding",
  "reasoning",
  "rag",
  "chat",
  "math",
  "extraction",
  "agentic",
];

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<
    "advisor" | "catalog" | "finetune" | "history"
  >("advisor");
  const [health, setHealth] = useState<{
    status: string;
    lmstudio_connected: boolean;
    lmstudio_url: string;
    lmstudio_loaded_models?: { id: string }[];
    catalog_models_count: number;
    recent_scrape_runs?: {
      id: number;
      completed_at: string;
      models_discovered: number;
      details: Record<string, number>;
    }[];
  } | null>(null);

  const [models, setModels] = useState<LLMModelSpec[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterOpenOnly, setFilterOpenOnly] = useState(false);
  const [filterSource, setFilterSource] = useState("");
  const [filterMaxVram, setFilterMaxVram] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [scraping, setScraping] = useState(false);
  const [result, setResult] = useState<RecommendationResponse | null>(null);

  // Dataset & Fine-tuning states
  const [numSamplesToGen, setNumSamplesToGen] = useState(30);
  const [datasetSamples, setDatasetSamples] = useState<{
    total: number;
    dataset_path: string;
    samples: {
      conversations: { from: string; value: string }[];
      metadata?: { top_model_id?: string; synthesized_with_lmstudio?: boolean };
    }[];
  } | null>(null);
  const [ftStatus, setFtStatus] = useState<{
    dataset_samples: number;
    gguf_artifacts: { name: string; path: string; size_mb: number }[];
    recent_jobs: {
      id: string;
      created_at: string;
      base_model: string;
      max_steps: number;
      lora_r: number;
      quantization: string;
      status: string;
      loss?: number;
      gguf_path?: string;
      logs?: string;
    }[];
  } | null>(null);
  const [ftBaseModel, setFtBaseModel] = useState("qwen-2.5-3b");
  const [ftSteps, setFtSteps] = useState(60);
  const [ftQuant, setFtQuant] = useState("q4_k_m");
  const [evaluations, setEvaluations] = useState<
    {
      id: number;
      created_at: string;
      project_name: string;
      top_model_id: string;
      top_match_score: number;
      used_lmstudio: boolean;
    }[]
  >([]);

  const [form, setForm] = useState<ProjectRequirements>({
    project_name: "Offline Code & RAG Assistant",
    project_description:
      "We need an LLM to index Python/TypeScript repositories, answer architectural questions, and generate patches locally on a consumer GPU (6-8GB VRAM).",
    primary_tasks: ["coding", "reasoning", "rag"],
    deployment_mode: "local",
    max_vram_gb: 8.0,
    max_budget_per_1m_tokens: 1.0,
    min_context_length: 32768,
    min_tokens_per_sec: 30,
    monthly_input_tokens_m: 15.0,
    monthly_output_tokens_m: 5.0,
    require_open_source: true,
    use_lmstudio_advisor: true,
  });

  const refreshAll = async () => {
    try {
      const [h, m, ds, ft, ev] = await Promise.all([
        fetchHealth(),
        fetchModels(),
        fetchDatasetSamples(20),
        fetchFinetuneStatus(),
        fetchEvaluations(),
      ]);
      setHealth(h);
      setModels(m);
      setDatasetSamples(ds);
      setFtStatus(ft);
      setEvaluations(ev);
    } catch {
      // Backend may still be starting
    }
  };

  useEffect(() => {
    refreshAll();
  }, []);

  const handleFilterCatalog = async () => {
    try {
      const m = await fetchModels({
        search: searchQuery || undefined,
        open_weights_only: filterOpenOnly,
        source: filterSource || undefined,
        max_vram_gb: filterMaxVram ? parseFloat(filterMaxVram) : undefined,
      });
      setModels(m);
    } catch {
      // ignore
    }
  };

  const toggleTask = (task: string) => {
    setForm((prev) => ({
      ...prev,
      primary_tasks: prev.primary_tasks.includes(task)
        ? prev.primary_tasks.filter((t) => t !== task)
        : [...prev.primary_tasks, task],
    }));
  };

  const handleRecommend = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await requestRecommendation(form);
      setResult(res);
      const ev = await fetchEvaluations();
      setEvaluations(ev);
    } catch {
      alert("Error connecting to FastAPI backend on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleScrape = async () => {
    setScraping(true);
    try {
      const res = await triggerScrape();
      setModels(res.models);
      const h = await fetchHealth();
      setHealth(h);
    } catch {
      alert("Failed to scrape sources.");
    } finally {
      setScraping(false);
    }
  };

  const handleGenerateDataset = async () => {
    setLoading(true);
    try {
      await generateDataset(numSamplesToGen, true);
      const [ds, ft] = await Promise.all([
        fetchDatasetSamples(20),
        fetchFinetuneStatus(),
      ]);
      setDatasetSamples(ds);
      setFtStatus(ft);
    } catch {
      alert("Failed to generate dataset.");
    } finally {
      setLoading(false);
    }
  };

  const handleStartFinetune = async (dryRun: boolean) => {
    setLoading(true);
    try {
      await triggerFinetuneJob({
        base_model: ftBaseModel,
        max_steps: ftSteps,
        lora_r: 16,
        quantization: ftQuant,
        dry_run: dryRun,
      });
      const ft = await fetchFinetuneStatus();
      setFtStatus(ft);
    } catch {
      alert("Failed to execute Unsloth job.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-6 mb-8">
        <div>
          <div className="flex items-center gap-3">
            <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              LLMarchitect v0.1
            </span>
            <span className="text-xs text-slate-400">
              Unsloth 3B QLoRA + LM Studio + Live Scrapers
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white mt-2">
            AI Model Selection & Fine-Tuning Studio
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Scrapes Hugging Face Leaderboards & Artificial Analysis to recommend the ideal LLM for your project constraints.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  health?.lmstudio_connected ? "bg-emerald-400" : "bg-amber-400"
                }`}
              />
              <span className="text-slate-300 font-medium">
                LM Studio:{" "}
                {health?.lmstudio_connected ? "Connected (Port 1234)" : "Standby / Heuristic Mode"}
              </span>
            </div>
            <div className="text-slate-500 mt-0.5">
              Catalog: {models.length} scraped models
            </div>
          </div>

          <button
            onClick={handleScrape}
            disabled={scraping}
            className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition disabled:opacity-50"
          >
            {scraping ? "Scraping Sources..." : "Refresh Scrapers"}
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2 mb-8 border-b border-slate-800 pb-3">
        {(
          [
            { id: "advisor", label: "1. Project LLM Advisor & TCO/VRAM" },
            { id: "catalog", label: `2. Live Scraped Catalog (${models.length})` },
            { id: "finetune", label: "3. Unsloth 3B & LM Studio Studio" },
            { id: "history", label: `4. Evaluations & Scrape Logs (${evaluations.length})` },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === tab.id
                ? "bg-indigo-600 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* TAB 1: ADVISOR */}
      {activeTab === "advisor" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <form
            onSubmit={handleRecommend}
            className="lg:col-span-5 bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-5"
          >
            <h2 className="text-lg font-semibold text-white">
              Project Specifications
            </h2>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Project Name
              </label>
              <input
                type="text"
                value={form.project_name}
                onChange={(e) => setForm({ ...form, project_name: e.target.value })}
                className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Project Use Case & Architecture Description
              </label>
              <textarea
                rows={4}
                value={form.project_description}
                onChange={(e) =>
                  setForm({ ...form, project_description: e.target.value })
                }
                className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-2">
                Primary Tasks
              </label>
              <div className="flex flex-wrap gap-2">
                {AVAILABLE_TASKS.map((task) => {
                  const selected = form.primary_tasks.includes(task);
                  return (
                    <button
                      type="button"
                      key={task}
                      onClick={() => toggleTask(task)}
                      className={`px-3 py-1 rounded-full text-xs font-medium border transition ${
                        selected
                          ? "bg-indigo-600/30 border-indigo-500 text-indigo-200"
                          : "bg-slate-950 border-slate-800 text-slate-400"
                      }`}
                    >
                      {task}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Deployment Mode
                </label>
                <select
                  value={form.deployment_mode}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      deployment_mode: e.target.value as "local" | "cloud" | "any",
                    })
                  }
                  className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white"
                >
                  <option value="local">Local (LM Studio / Unsloth)</option>
                  <option value="cloud">Cloud API</option>
                  <option value="any">Any (Hybrid)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Max GPU VRAM (GB)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={form.max_vram_gb ?? 8}
                  onChange={(e) =>
                    setForm({ ...form, max_vram_gb: parseFloat(e.target.value) })
                  }
                  className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Max Cost ($ / 1M Tokens)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={form.max_budget_per_1m_tokens ?? 5}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      max_budget_per_1m_tokens: parseFloat(e.target.value),
                    })
                  }
                  className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Min Context Length
                </label>
                <input
                  type="number"
                  step="1024"
                  value={form.min_context_length}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      min_context_length: parseInt(e.target.value, 10),
                    })
                  }
                  className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white"
                />
              </div>
            </div>

            <div className="space-y-2 pt-1">
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={form.require_open_source}
                  onChange={(e) =>
                    setForm({ ...form, require_open_source: e.target.checked })
                  }
                />
                Require Open-Weights (Fine-tunable with Unsloth)
              </label>

              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={form.use_lmstudio_advisor}
                  onChange={(e) =>
                    setForm({ ...form, use_lmstudio_advisor: e.target.checked })
                  }
                />
                Query Fine-Tuned Model in LM Studio (`localhost:1234/v1`)
              </label>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition disabled:opacity-50"
            >
              {loading ? "Evaluating Models..." : "Find Best LLM for Project"}
            </button>
          </form>

          {/* Recommendations Output */}
          <div className="lg:col-span-7 space-y-6">
            {!result ? (
              <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-10 text-center text-slate-400">
                Submit your project requirements to rank scraped LLMs and get architectural deployment guidance.
              </div>
            ) : (
              <>
                <div className="bg-indigo-950/30 border border-indigo-500/30 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold uppercase tracking-wider text-indigo-300">
                      LLMarchitect Advisor Synthesis
                    </span>
                    <span className="text-xs px-2.5 py-0.5 rounded bg-indigo-500/20 text-indigo-200">
                      {result.used_lmstudio
                        ? "Generated by Local LM Studio Model"
                        : "Generated by Multi-Metric Scoring Engine"}
                    </span>
                  </div>
                  <div className="text-sm text-slate-200 whitespace-pre-line leading-relaxed">
                    {result.advisor_analysis}
                  </div>
                </div>

                <div className="space-y-4">
                  {result.recommendations.map((rec, idx) => (
                    <div
                      key={rec.model.id}
                      className="bg-slate-900/80 border border-slate-800 rounded-xl p-5"
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 text-xs font-bold rounded bg-indigo-600 text-white">
                              #{idx + 1}
                            </span>
                            <h3 className="text-base font-bold text-white">
                              {rec.model.name}
                            </h3>
                            <span className="text-xs text-slate-400">
                              ({rec.model.provider})
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 mt-1">
                            {rec.deployment_fit} • Context:{" "}
                            {(rec.model.context_length / 1024).toFixed(0)}k • Source:{" "}
                            {rec.model.source}
                          </p>
                        </div>
                        <div className="text-right">
                          <span className="text-xl font-bold text-emerald-400">
                            {rec.match_score}%
                          </span>
                          <div className="text-[11px] text-slate-400">Match Score</div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 my-4 text-center bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80 text-xs">
                        <div>
                          <div className="text-slate-400">MMLU</div>
                          <div className="font-semibold text-white">
                            {rec.model.mmlu_score ?? "—"}%
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-400">Coding</div>
                          <div className="font-semibold text-white">
                            {rec.model.coding_score ?? "—"}%
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-400">Reasoning</div>
                          <div className="font-semibold text-white">
                            {rec.model.reasoning_score ?? "—"}%
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-400">VRAM (Q4 / Q8)</div>
                          <div className="font-semibold text-indigo-300">
                            {rec.quantization?.q4_k_m_vram_gb
                              ? `${rec.quantization.q4_k_m_vram_gb}G / ${rec.quantization.q8_0_vram_gb ?? "—"}G`
                              : "Cloud API"}
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-400">Est. Monthly TCO</div>
                          <div className="font-semibold text-emerald-300">
                            {rec.cost_estimate?.monthly_api_cost_usd === 0
                              ? "$0 (Local GPU)"
                              : `$${rec.cost_estimate?.monthly_api_cost_usd}/mo`}
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        <div>
                          <span className="text-emerald-400 font-semibold">
                            Strengths:
                          </span>
                          <ul className="list-disc list-inside text-slate-300 mt-1 space-y-0.5">
                            {rec.strengths.map((s, i) => (
                              <li key={i}>{s}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <span className="text-amber-400 font-semibold">
                            Trade-offs:
                          </span>
                          <ul className="list-disc list-inside text-slate-300 mt-1 space-y-0.5">
                            {rec.tradeoffs.map((t, i) => (
                              <li key={i}>{t}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: CATALOG */}
      {activeTab === "catalog" && (
        <div className="space-y-4">
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center gap-3">
            <input
              type="text"
              placeholder="Search model name, provider, or tag (e.g. qwen, coding, r1)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 min-w-[220px] rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-white"
            />
            <select
              value={filterSource}
              onChange={(e) => setFilterSource(e.target.value)}
              className="rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-white"
            >
              <option value="">All Sources</option>
              <option value="huggingface">Hugging Face Leaderboards</option>
              <option value="artificial_analysis">Artificial Analysis</option>
              <option value="openrouter_live">OpenRouter Live API</option>
            </select>
            <input
              type="number"
              placeholder="Max VRAM (GB)"
              value={filterMaxVram}
              onChange={(e) => setFilterMaxVram(e.target.value)}
              className="w-32 rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-white"
            />
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={filterOpenOnly}
                onChange={(e) => setFilterOpenOnly(e.target.checked)}
              />
              Open-Weights Only
            </label>
            <button
              onClick={handleFilterCatalog}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold"
            >
              Apply Filters
            </button>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-xl overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-950 text-slate-400">
                  <th className="p-3.5">Model</th>
                  <th className="p-3.5">Source</th>
                  <th className="p-3.5">Params</th>
                  <th className="p-3.5">VRAM (Q4 / Q8 / FP16)</th>
                  <th className="p-3.5">Context</th>
                  <th className="p-3.5">Coding</th>
                  <th className="p-3.5">Reasoning</th>
                  <th className="p-3.5">Speed (tok/s)</th>
                  <th className="p-3.5">API Price (/1M)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {models.map((m) => (
                  <tr key={m.id} className="hover:bg-slate-800/40">
                    <td className="p-3.5 font-medium text-white">
                      <a
                        href={m.url || "#"}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:text-indigo-400 underline-offset-2 hover:underline"
                      >
                        {m.name}
                      </a>
                      <div className="text-[11px] text-slate-500">{m.id}</div>
                    </td>
                    <td className="p-3.5 text-slate-300">{m.source}</td>
                    <td className="p-3.5 text-slate-300">
                      {m.parameters_b ? `${m.parameters_b}B` : "Proprietary"}
                    </td>
                    <td className="p-3.5 text-indigo-300">
                      {m.vram_q4_gb
                        ? `${m.vram_q4_gb}G / ${m.vram_q8_gb}G / ${m.vram_fp16_gb}G`
                        : "Cloud Only"}
                    </td>
                    <td className="p-3.5 text-slate-300">
                      {(m.context_length / 1024).toFixed(0)}k
                    </td>
                    <td className="p-3.5 text-slate-200">{m.coding_score ?? "—"}%</td>
                    <td className="p-3.5 text-slate-200">{m.reasoning_score ?? "—"}%</td>
                    <td className="p-3.5 text-slate-200">
                      {m.median_tokens_per_sec ?? "—"}
                    </td>
                    <td className="p-3.5 text-slate-200">
                      {m.input_price_per_1m === 0
                        ? "Free (Local)"
                        : `$${m.input_price_per_1m} / $${m.output_price_per_1m}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: UNSLOTH + LM STUDIO PIPELINE */}
      {activeTab === "finetune" && (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-white">
                Step 1: ShareGPT Dataset Synthesizer (LM Studio + Scraped Benchmarks)
              </h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Generates ShareGPT multi-turn conversations from live scraped Hugging Face, OpenRouter, and Artificial Analysis benchmarks. Currently{" "}
                <strong className="text-white">
                  {datasetSamples?.total ?? ftStatus?.dataset_samples ?? 0}
                </strong>{" "}
                conversations stored in{" "}
                <code className="text-indigo-300">
                  finetuning/datasets/llm_advisor_train.jsonl
                </code>
                .
              </p>
              <div className="flex items-center gap-3">
                <input
                  type="number"
                  min={4}
                  max={200}
                  value={numSamplesToGen}
                  onChange={(e) => setNumSamplesToGen(parseInt(e.target.value, 10))}
                  className="w-24 rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-white"
                />
                <button
                  onClick={handleGenerateDataset}
                  disabled={loading}
                  className="px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition disabled:opacity-50"
                >
                  {loading
                    ? "Synthesizing..."
                    : `Generate ${numSamplesToGen} ShareGPT Training Pairs`}
                </button>
              </div>
            </div>

            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
              <h2 className="text-lg font-semibold text-white">
                Step 2: Unsloth 3B QLoRA Job Runner & GGUF Exporter
              </h2>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Base Model (6-8GB VRAM)</label>
                  <select
                    value={ftBaseModel}
                    onChange={(e) => setFtBaseModel(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-2.5 py-2 text-white"
                  >
                    <option value="qwen-2.5-3b">Qwen 2.5 3B Instruct (4-bit)</option>
                    <option value="llama-3.2-3b">Llama 3.2 3B Instruct (4-bit)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Max Steps</label>
                  <input
                    type="number"
                    value={ftSteps}
                    onChange={(e) => setFtSteps(parseInt(e.target.value, 10))}
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-2.5 py-2 text-white"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">GGUF Quantization</label>
                  <select
                    value={ftQuant}
                    onChange={(e) => setFtQuant(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-2.5 py-2 text-white"
                  >
                    <option value="q4_k_m">Q4_K_M (Recommended)</option>
                    <option value="q8_0">Q8_0 (High Quality)</option>
                    <option value="f16">FP16</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => handleStartFinetune(true)}
                  disabled={loading}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700"
                >
                  Run Pipeline Verification (Dry-Run)
                </button>
                <button
                  onClick={() => handleStartFinetune(false)}
                  disabled={loading}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold"
                >
                  Launch Full Unsloth CUDA QLoRA Training
                </button>
              </div>
            </div>
          </div>

          {/* Recent Unsloth Jobs & Logs */}
          {ftStatus?.recent_jobs && ftStatus.recent_jobs.length > 0 && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
              <h3 className="text-sm font-semibold text-white">
                Unsloth Fine-Tuning Jobs & Console Output
              </h3>
              <div className="space-y-3">
                {ftStatus.recent_jobs.map((job) => (
                  <div
                    key={job.id}
                    className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-indigo-300">
                        Job {job.id} • {job.base_model} ({job.quantization.toUpperCase()})
                      </span>
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-200">
                        {job.status}
                      </span>
                    </div>
                    {job.logs && (
                      <pre className="p-3 rounded bg-black/60 text-slate-300 overflow-x-auto text-[11px]">
                        {job.logs}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dataset Sample Inspector */}
          {datasetSamples?.samples && datasetSamples.samples.length > 0 && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
              <h3 className="text-sm font-semibold text-white">
                ShareGPT Training Dataset Inspector (First 5 Conversations)
              </h3>
              <div className="space-y-3">
                {datasetSamples.samples.slice(0, 5).map((item, i) => (
                  <div
                    key={i}
                    className="p-4 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-2"
                  >
                    <div className="text-indigo-400 font-semibold">
                      Sample #{i + 1} — Top Model: {item.metadata?.top_model_id}
                    </div>
                    <div className="text-slate-300 whitespace-pre-line">
                      <strong>Prompt:</strong> {item.conversations[1]?.value}
                    </div>
                    <div className="text-slate-400 whitespace-pre-line border-t border-slate-800 pt-2">
                      <strong>Target Completion:</strong>{" "}
                      {item.conversations[2]?.value}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: EVALUATION HISTORY & SCRAPE RUNS */}
      {activeTab === "history" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-white">
              Saved Project Evaluations (SQLite)
            </h2>
            <div className="space-y-3">
              {evaluations.map((ev) => (
                <div
                  key={ev.id}
                  className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs flex items-center justify-between"
                >
                  <div>
                    <div className="font-semibold text-white">{ev.project_name}</div>
                    <div className="text-slate-400">
                      Top Pick: <span className="text-indigo-300">{ev.top_model_id}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-emerald-400">
                      {ev.top_match_score}%
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {new Date(ev.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-white">
              Live Scraper Audit Log
            </h2>
            <div className="space-y-3">
              {health?.recent_scrape_runs?.map((run) => (
                <div
                  key={run.id}
                  className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-1"
                >
                  <div className="flex justify-between">
                    <span className="font-semibold text-emerald-400">
                      Run #{run.id} — {run.models_discovered} Total Unique Models
                    </span>
                    <span className="text-slate-500">
                      {new Date(run.completed_at).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-slate-400">
                    Sources: HF ({run.details?.huggingface_leaderboards ?? 0}), Artificial Analysis ({run.details?.artificial_analysis ?? 0}), OpenRouter Live ({run.details?.openrouter_live ?? 0})
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
