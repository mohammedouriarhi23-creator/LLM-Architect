import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import settings
from app.models.schemas import LLMModelSpec


class Database:
    """
    Production SQLite persistence layer for LLMarchitect:
    - models: Scraped LLM specifications, benchmarks, pricing, and VRAM metrics
    - scrape_runs: Audit log of live scraping jobs per source
    - evaluations: Saved project requirement evaluations & recommendations
    - dataset_samples: ShareGPT fine-tuning conversations & quality scores
    - finetune_jobs: Unsloth training runs, logs, and exported GGUF artifacts
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.data_dir / "llmarchitect.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source TEXT NOT NULL,
                    parameters_b REAL,
                    context_length INTEGER NOT NULL,
                    license TEXT NOT NULL,
                    open_weights INTEGER NOT NULL,
                    mmlu_score REAL,
                    coding_score REAL,
                    reasoning_score REAL,
                    math_score REAL,
                    elo_arena REAL,
                    average_benchmark REAL,
                    input_price_per_1m REAL,
                    output_price_per_1m REAL,
                    median_tokens_per_sec REAL,
                    ttft_seconds REAL,
                    min_vram_gb REAL,
                    vram_q4_gb REAL,
                    vram_q8_gb REAL,
                    vram_fp16_gb REAL,
                    architecture TEXT,
                    modality TEXT,
                    downloads INTEGER,
                    likes INTEGER,
                    tags_json TEXT NOT NULL,
                    url TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scrape_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    models_discovered INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    requirements_json TEXT NOT NULL,
                    top_model_id TEXT,
                    top_match_score REAL,
                    used_lmstudio INTEGER NOT NULL,
                    response_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS finetune_jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    max_steps INTEGER NOT NULL,
                    lora_r INTEGER NOT NULL,
                    quantization TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_step INTEGER DEFAULT 0,
                    loss REAL,
                    gguf_path TEXT,
                    logs TEXT DEFAULT ''
                );
                """
            )

    def upsert_models(self, models: List[LLMModelSpec]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        with self.connect() as conn:
            for m in models:
                conn.execute(
                    """
                    INSERT INTO models (
                        id, name, provider, source, parameters_b, context_length,
                        license, open_weights, mmlu_score, coding_score, reasoning_score,
                        math_score, elo_arena, average_benchmark, input_price_per_1m,
                        output_price_per_1m, median_tokens_per_sec, ttft_seconds,
                        min_vram_gb, vram_q4_gb, vram_q8_gb, vram_fp16_gb,
                        architecture, modality, downloads, likes, tags_json, url, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        provider=excluded.provider,
                        source=excluded.source,
                        parameters_b=COALESCE(excluded.parameters_b, models.parameters_b),
                        context_length=excluded.context_length,
                        license=excluded.license,
                        open_weights=excluded.open_weights,
                        mmlu_score=COALESCE(excluded.mmlu_score, models.mmlu_score),
                        coding_score=COALESCE(excluded.coding_score, models.coding_score),
                        reasoning_score=COALESCE(excluded.reasoning_score, models.reasoning_score),
                        math_score=COALESCE(excluded.math_score, models.math_score),
                        elo_arena=COALESCE(excluded.elo_arena, models.elo_arena),
                        average_benchmark=COALESCE(excluded.average_benchmark, models.average_benchmark),
                        input_price_per_1m=COALESCE(excluded.input_price_per_1m, models.input_price_per_1m),
                        output_price_per_1m=COALESCE(excluded.output_price_per_1m, models.output_price_per_1m),
                        median_tokens_per_sec=COALESCE(excluded.median_tokens_per_sec, models.median_tokens_per_sec),
                        ttft_seconds=COALESCE(excluded.ttft_seconds, models.ttft_seconds),
                        min_vram_gb=COALESCE(excluded.min_vram_gb, models.min_vram_gb),
                        vram_q4_gb=COALESCE(excluded.vram_q4_gb, models.vram_q4_gb),
                        vram_q8_gb=COALESCE(excluded.vram_q8_gb, models.vram_q8_gb),
                        vram_fp16_gb=COALESCE(excluded.vram_fp16_gb, models.vram_fp16_gb),
                        architecture=COALESCE(excluded.architecture, models.architecture),
                        modality=COALESCE(excluded.modality, models.modality),
                        downloads=COALESCE(excluded.downloads, models.downloads),
                        likes=COALESCE(excluded.likes, models.likes),
                        tags_json=excluded.tags_json,
                        url=COALESCE(excluded.url, models.url),
                        updated_at=excluded.updated_at
                    """,
                    (
                        m.id,
                        m.name,
                        m.provider,
                        m.source,
                        m.parameters_b,
                        m.context_length,
                        m.license,
                        1 if m.open_weights else 0,
                        m.mmlu_score,
                        m.coding_score,
                        m.reasoning_score,
                        m.math_score,
                        m.elo_arena,
                        m.average_benchmark,
                        m.input_price_per_1m,
                        m.output_price_per_1m,
                        m.median_tokens_per_sec,
                        m.ttft_seconds,
                        m.min_vram_gb,
                        m.vram_q4_gb,
                        m.vram_q8_gb,
                        m.vram_fp16_gb,
                        m.architecture,
                        m.modality,
                        m.downloads,
                        m.likes,
                        json.dumps(m.tags),
                        m.url,
                        now,
                    ),
                )
                count += 1
        return count

    def list_models(
        self,
        search: Optional[str] = None,
        open_weights_only: bool = False,
        max_vram_gb: Optional[float] = None,
        source: Optional[str] = None,
    ) -> List[LLMModelSpec]:
        query = "SELECT * FROM models WHERE 1=1"
        params: List[Any] = []

        if open_weights_only:
            query += " AND open_weights = 1"
        if max_vram_gb is not None:
            query += " AND min_vram_gb IS NOT NULL AND min_vram_gb <= ?"
            params.append(max_vram_gb)
        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")
        if search:
            query += " AND (LOWER(name) LIKE ? OR LOWER(id) LIKE ? OR LOWER(provider) LIKE ? OR LOWER(tags_json) LIKE ?)"
            q = f"%{search.lower()}%"
            params.extend([q, q, q, q])

        query += " ORDER BY COALESCE(average_benchmark, 0) DESC, COALESCE(downloads, 0) DESC"

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_model(r) for r in rows]

    def _row_to_model(self, row: sqlite3.Row) -> LLMModelSpec:
        return LLMModelSpec(
            id=row["id"],
            name=row["name"],
            provider=row["provider"],
            source=row["source"],
            parameters_b=row["parameters_b"],
            context_length=row["context_length"],
            license=row["license"],
            open_weights=bool(row["open_weights"]),
            mmlu_score=row["mmlu_score"],
            coding_score=row["coding_score"],
            reasoning_score=row["reasoning_score"],
            math_score=row["math_score"],
            elo_arena=row["elo_arena"],
            average_benchmark=row["average_benchmark"],
            input_price_per_1m=row["input_price_per_1m"],
            output_price_per_1m=row["output_price_per_1m"],
            median_tokens_per_sec=row["median_tokens_per_sec"],
            ttft_seconds=row["ttft_seconds"],
            min_vram_gb=row["min_vram_gb"],
            vram_q4_gb=row["vram_q4_gb"],
            vram_q8_gb=row["vram_q8_gb"],
            vram_fp16_gb=row["vram_fp16_gb"],
            architecture=row["architecture"] or "Transformer",
            modality=row["modality"] or "text->text",
            downloads=row["downloads"] or 0,
            likes=row["likes"] or 0,
            tags=json.loads(row["tags_json"] or "[]"),
            url=row["url"],
            updated_at=row["updated_at"],
        )

    def record_scrape_run(
        self,
        started_at: str,
        completed_at: str,
        sources: List[str],
        models_discovered: int,
        status: str,
        details: Dict[str, Any],
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scrape_runs (started_at, completed_at, sources_json, models_discovered, status, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    completed_at,
                    json.dumps(sources),
                    models_discovered,
                    status,
                    json.dumps(details),
                ),
            )
            return int(cur.lastrowid)

    def list_scrape_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scrape_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "started_at": r["started_at"],
                    "completed_at": r["completed_at"],
                    "sources": json.loads(r["sources_json"]),
                    "models_discovered": r["models_discovered"],
                    "status": r["status"],
                    "details": json.loads(r["details_json"]),
                }
                for r in rows
            ]

    def save_evaluation(
        self,
        project_name: str,
        requirements: Dict[str, Any],
        top_model_id: Optional[str],
        top_match_score: Optional[float],
        used_lmstudio: bool,
        response_payload: Dict[str, Any],
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluations (created_at, project_name, requirements_json, top_model_id, top_match_score, used_lmstudio, response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    project_name,
                    json.dumps(requirements),
                    top_model_id,
                    top_match_score,
                    1 if used_lmstudio else 0,
                    json.dumps(response_payload),
                ),
            )
            return int(cur.lastrowid)

    def list_evaluations(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, project_name, requirements_json, top_model_id, top_match_score, used_lmstudio FROM evaluations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "project_name": r["project_name"],
                    "requirements": json.loads(r["requirements_json"]),
                    "top_model_id": r["top_model_id"],
                    "top_match_score": r["top_match_score"],
                    "used_lmstudio": bool(r["used_lmstudio"]),
                }
                for r in rows
            ]

    def upsert_finetune_job(self, job: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO finetune_jobs (
                    id, created_at, updated_at, base_model, dataset_path, max_steps,
                    lora_r, quantization, status, current_step, loss, gguf_path, logs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    current_step=excluded.current_step,
                    loss=excluded.loss,
                    gguf_path=excluded.gguf_path,
                    logs=excluded.logs
                """,
                (
                    job["id"],
                    job.get("created_at", now),
                    now,
                    job["base_model"],
                    job["dataset_path"],
                    job["max_steps"],
                    job["lora_r"],
                    job["quantization"],
                    job["status"],
                    job.get("current_step", 0),
                    job.get("loss"),
                    job.get("gguf_path"),
                    job.get("logs", ""),
                ),
            )

    def list_finetune_jobs(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM finetune_jobs ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            return [dict(r) for r in rows]
