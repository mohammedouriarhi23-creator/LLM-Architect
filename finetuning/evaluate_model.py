import argparse
import html
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "backend"
DATABASE_PATH = ROOT / "backend" / "data" / "llmarchitect.db"
DATASET_PATH = ROOT / "finetuning" / "datasets" / "llm_advisor_train.jsonl"
OUTPUT_DIR = ROOT / "finetuning" / "evaluation"
ADVISOR_MODEL_ID = "qwen2.5-3b-advisor"
BASE_MODEL_ID = "qwen2.5-3b-instruct"

SCENARIOS = [
    {
        "name": "Assistant de migration hors ligne",
        "description": "Un outil de migration de code doit fonctionner sans réseau sur une station avec 8 Go de VRAM.",
        "primary_tasks": ["coding", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 8.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 32768,
        "require_open_source": True,
    },
    {
        "name": "Recherche dans des dossiers réglementaires",
        "description": "Comparer de longs dossiers réglementaires et retrouver les clauses avec leurs références.",
        "primary_tasks": ["rag", "extraction", "reasoning"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 12.0,
        "min_context_length": 256000,
        "require_open_source": False,
    },
    {
        "name": "Extraction de reçus à faible coût",
        "description": "Extraire des champs JSON de reçus anonymisés avec un budget API strict.",
        "primary_tasks": ["extraction", "reasoning"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 0.15,
        "min_context_length": 8192,
        "require_open_source": True,
    },
    {
        "name": "Tuteur de démonstration mathématique local",
        "description": "Un tuteur interactif explique et vérifie des démonstrations mathématiques sur une machine locale.",
        "primary_tasks": ["math", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 10.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 32768,
        "require_open_source": True,
    },
    {
        "name": "Synthèse rapide des demandes entrantes",
        "description": "Résumer des messages entrants et proposer une réponse en temps réel via une API.",
        "primary_tasks": ["chat", "extraction"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 2.0,
        "min_context_length": 16384,
        "min_tokens_per_sec": 95.0,
        "require_open_source": False,
    },
    {
        "name": "Classification sur appareil industriel",
        "description": "Classer et structurer des observations courtes sur un appareil isolé, sans service distant.",
        "primary_tasks": ["extraction", "chat"],
        "deployment_mode": "local",
        "max_vram_gb": 3.5,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 8192,
        "require_open_source": True,
    },
    {
        "name": "Assistant documentaire souverain",
        "description": "Indexer la documentation interne dans un environnement privé et justifier chaque réponse.",
        "primary_tasks": ["rag", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 20.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 65536,
        "require_open_source": True,
    },
    {
        "name": "Revue automatisée de dépôts logiciels",
        "description": "Analyser des changements répartis entre plusieurs dépôts et expliquer les risques de régression.",
        "primary_tasks": ["coding", "agentic"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 20.0,
        "min_context_length": 96000,
        "require_open_source": False,
    },
    {
        "name": "Croisement de subventions longues",
        "description": "Croiser plusieurs dossiers de subvention et en extraire les critères d'éligibilité.",
        "primary_tasks": ["extraction", "rag"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 6.0,
        "min_context_length": 256000,
        "require_open_source": False,
    },
    {
        "name": "Résumé local des échanges de support",
        "description": "Résumer des échanges de support sensibles sur un poste interne et produire des réponses cohérentes.",
        "primary_tasks": ["chat", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 12.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 48000,
        "require_open_source": True,
    },
    {
        "name": "Assistant SQL pour un poste de travail",
        "description": "Aider les analystes à écrire, corriger et expliquer des requêtes SQL sur un poste avec 6 Go de VRAM.",
        "primary_tasks": ["coding", "reasoning"],
        "deployment_mode": "local",
        "max_vram_gb": 6.0,
        "max_budget_per_1m_tokens": 0.0,
        "min_context_length": 16384,
        "require_open_source": True,
    },
    {
        "name": "Révision multilingue de contenus",
        "description": "Réviser et reformuler des contenus multilingues avec un débit régulier et un plafond de coût serré.",
        "primary_tasks": ["chat", "reasoning"],
        "deployment_mode": "cloud",
        "max_vram_gb": None,
        "max_budget_per_1m_tokens": 0.8,
        "min_context_length": 32768,
        "min_tokens_per_sec": 45.0,
        "require_open_source": False,
    },
]


def load_models() -> tuple[list[Any], dict[str, Any]]:
    import sys

    sys.path.insert(0, str(BACKEND_PATH))
    from app.models.schemas import LLMModelSpec
    from app.services.recommender import RecommenderService

    uri = f"file:{DATABASE_PATH.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM models").fetchall()
        job = connection.execute(
            """
            SELECT base_model, max_steps, current_step, loss, logs, gguf_path
            FROM finetune_jobs
            WHERE status = 'completed'
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()

    models = []
    for row in rows:
        data = dict(row)
        data["tags"] = json.loads(data.pop("tags_json"))
        models.append(LLMModelSpec(**data))
    if not models:
        raise RuntimeError(f"Aucun modèle n'est enregistré dans {DATABASE_PATH}.")
    if not job:
        raise RuntimeError("Aucune exécution d'entraînement terminée n'est enregistrée.")

    recommender = object.__new__(RecommenderService)
    return models, {"job": dict(job), "recommender": recommender}


def read_dataset_count() -> int:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return sum(1 for line in dataset_file if line.strip())


def build_recommendations(models: list[Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    from app.models.schemas import ProjectRequirements

    recommender = context["recommender"]
    cases = []
    for scenario in SCENARIOS:
        requirements = ProjectRequirements(
            project_name=scenario["name"],
            project_description=scenario["description"],
            primary_tasks=scenario["primary_tasks"],
            deployment_mode=scenario["deployment_mode"],
            max_vram_gb=scenario["max_vram_gb"],
            max_budget_per_1m_tokens=scenario["max_budget_per_1m_tokens"],
            min_context_length=scenario["min_context_length"],
            min_tokens_per_sec=scenario.get("min_tokens_per_sec"),
            require_open_source=scenario["require_open_source"],
            use_lmstudio_advisor=False,
        )
        scored = [
            result
            for model in models
            if (result := recommender.evaluate_candidate(model, requirements)) is not None
        ]
        scored.sort(key=lambda result: result.match_score, reverse=True)
        if not scored:
            raise RuntimeError(f"Aucune recommandation du moteur pour « {scenario['name']} ».")

        top = scored[:3]
        highest_score = scored[0].match_score
        winners = [
            result.model.id
            for result in top
            if abs(result.match_score - highest_score) < 0.05
        ]
        prompt = (
            f"Project: {requirements.project_name}\n"
            f"Description: {requirements.project_description}\n"
            f"Primary Tasks: {', '.join(requirements.primary_tasks)}\n"
            f"Deployment Mode: {requirements.deployment_mode} "
            f"(Max VRAM: {requirements.max_vram_gb} GB)\n"
            f"Max Budget/1M tokens: ${requirements.max_budget_per_1m_tokens}\n\n"
            "Top Scraped Candidates:\n"
            + "\n".join(
                f"{index + 1}. {candidate.model.name} ({candidate.model.id}) - "
                f"Match: {candidate.match_score}%, "
                f"VRAM(Q4): {candidate.model.min_vram_gb or 'Cloud'}GB, "
                f"Context: {candidate.model.context_length}, "
                f"Coding: {candidate.model.coding_score}, "
                f"Reasoning: {candidate.model.reasoning_score}, "
                f"Monthly Cost: ${candidate.cost_estimate.monthly_api_cost_usd if candidate.cost_estimate else 0}, "
                f"Strengths: {', '.join(candidate.strengths)}"
                for index, candidate in enumerate(top)
            )
            + "\n\nProvide a concise architectural recommendation explaining which LLM "
            "to pick and how to deploy or fine-tune it."
        )
        cases.append(
            {
                "scenario": scenario,
                "prompt": prompt,
                "expected_ids": winners,
                "candidates": [
                    {
                        "id": candidate.model.id,
                        "name": candidate.model.name,
                        "match_score": candidate.match_score,
                        "context_length": candidate.model.context_length,
                        "open_weights": candidate.model.open_weights,
                        "q4_vram_gb": candidate.model.vram_q4_gb
                        or candidate.model.min_vram_gb,
                        "input_price_per_1m": candidate.model.input_price_per_1m,
                        "output_price_per_1m": candidate.model.output_price_per_1m,
                    }
                    for candidate in top
                ],
            }
        )
    return cases


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio a répondu HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Impossible de joindre LM Studio ({url}): {error.reason}") from error


def request_completion(
    base_url: str,
    model_id: str,
    prompt: str,
    system_prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
            "stream": False,
        },
    )
    elapsed = time.perf_counter() - started
    try:
        output = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"Réponse LM Studio inattendue pour le modèle « {model_id} »: "
            f"{json.dumps(result, ensure_ascii=False)[:1000]}"
        ) from error
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError(f"Le modèle « {model_id} » a renvoyé une réponse vide.")

    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    timings = result.get("timings") or {}
    generation_rate = timings.get("predicted_per_second")
    if generation_rate is None and completion_tokens:
        generation_rate = completion_tokens / elapsed
    return {
        "output": output,
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "generation_tokens_per_second": generation_rate,
    }


def find_selected_candidate(output: str, candidates: list[dict[str, Any]]) -> str | None:
    id_matches = []
    for candidate in candidates:
        match = re.search(re.escape(candidate["id"]), output, flags=re.IGNORECASE)
        if match:
            id_matches.append((match.start(), candidate["id"]))
    if id_matches:
        first_position = min(match[0] for match in id_matches)
        matched_ids = {
            model_id for position, model_id in id_matches if position == first_position
        }
        return next(iter(matched_ids)) if len(matched_ids) == 1 else None

    normalized_output = re.sub(r"[^a-z0-9]+", " ", output.lower()).strip()
    matches = []
    for index, candidate in enumerate(candidates):
        phrases = {candidate["id"], candidate["name"]}
        if ":" in candidate["name"]:
            phrases.add(candidate["name"].split(":", 1)[1])
        phrases.add(candidate["id"].rsplit("/", 1)[-1])
        aliases = set()
        for phrase in phrases:
            normalized = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
            if not normalized:
                continue
            aliases.add(normalized)
            tokens = normalized.split()
            for end in range(len(tokens) - 1, 2, -1):
                aliases.add(" ".join(tokens[:end]))
        for alias in aliases:
            match = re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized_output,
            )
            if match:
                matches.append((match.start(), -len(alias), index, candidate["id"]))
    if not matches:
        return None
    first_position = min(match[0] for match in matches)
    longest_match = min(match[1] for match in matches if match[0] == first_position)
    matched_ids = {
        match[3]
        for match in matches
        if match[0] == first_position and match[1] == longest_match
    }
    return next(iter(matched_ids)) if len(matched_ids) == 1 else None


def candidate_compliance(candidate: dict[str, Any], scenario: dict[str, Any]) -> list[str]:
    failures = []
    if candidate["context_length"] < scenario["min_context_length"]:
        failures.append("contexte")
    if scenario["require_open_source"] and not candidate["open_weights"]:
        failures.append("poids ouverts")
    if scenario["deployment_mode"] == "local":
        vram = candidate["q4_vram_gb"]
        if vram is None or vram > scenario["max_vram_gb"]:
            failures.append("VRAM")
    elif scenario["max_budget_per_1m_tokens"] is not None:
        in_price = candidate["input_price_per_1m"]
        out_price = candidate["output_price_per_1m"]
        if in_price is None or out_price is None:
            failures.append("prix API inconnu")
        elif (in_price + out_price) / 2 > scenario["max_budget_per_1m_tokens"]:
            failures.append("budget API")
    return failures


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * (
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) ** 0.5 / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    matched = sum(result["decision_match"] for result in results)
    grounded = sum(result["selected_model_id"] is not None for result in results)
    compliant = sum(result["constraint_compliant"] for result in results)
    latencies = [result["elapsed_seconds"] for result in results]
    rates = [
        result["generation_tokens_per_second"]
        for result in results
        if result["generation_tokens_per_second"] is not None
    ]
    tokens = [
        result["completion_tokens"]
        for result in results
        if result["completion_tokens"] is not None
    ]
    low, high = wilson_interval(matched, total)
    return {
        "total": total,
        "top1_correct": matched,
        "top1_accuracy": matched / total,
        "top1_wilson_95": [low, high],
        "grounded": grounded,
        "grounded_rate": grounded / total,
        "constraint_compliant": compliant,
        "constraint_compliance_rate": compliant / total,
        "mean_latency_seconds": mean(latencies) if latencies else 0.0,
        "median_latency_seconds": median(latencies) if latencies else 0.0,
        "mean_generation_tokens_per_second": mean(rates) if rates else None,
        "mean_completion_tokens": mean(tokens) if tokens else None,
    }


def refresh_report_results(
    report: dict[str, Any],
    advisor_model_id: str,
    base_model_id: str,
) -> None:
    for model_id in (advisor_model_id, base_model_id):
        model_results = report["results"][model_id]
        if len(model_results) != len(report["cases"]):
            raise RuntimeError(f"Nombre de réponses incohérent pour « {model_id} ».")
        for case, result in zip(report["cases"], model_results):
            selected_id = find_selected_candidate(result["output"], case["candidates"])
            selected = next(
                (
                    candidate
                    for candidate in case["candidates"]
                    if candidate["id"] == selected_id
                ),
                None,
            )
            failures = (
                candidate_compliance(selected, case["scenario"])
                if selected
                else ["recommandation non identifiable"]
            )
            result.update(
                {
                    "selected_model_id": selected_id,
                    "selected_model_name": selected["name"] if selected else None,
                    "decision_match": selected_id in case["expected_ids"],
                    "candidate_grounded": selected is not None,
                    "constraint_compliant": not failures,
                    "constraint_failures": failures,
                }
            )
    report["metrics"] = {
        model_id: summarize(report["results"][model_id])
        for model_id in (advisor_model_id, base_model_id)
    }
    fine = report["metrics"][advisor_model_id]
    baseline = report["metrics"][base_model_id]
    report["conclusion"] = (
        f"Sur ce jeu de {len(report['cases'])} cas, le modèle affiné atteint "
        f"{fine['top1_correct']}/{len(report['cases'])} décisions conformes au moteur "
        f"({percentage(fine['top1_accuracy'])}), contre "
        f"{baseline['top1_correct']}/{len(report['cases'])} "
        f"({percentage(baseline['top1_accuracy'])}) pour le modèle de base. "
        f"L'écart observé est de "
        f"{(fine['top1_accuracy'] - baseline['top1_accuracy']) * 100:+.1f} points. "
        "Avec un seul échantillon de génération par scénario, cet écart est indicatif, pas une preuve de supériorité."
    )


def evaluate_model(
    base_url: str,
    model_id: str,
    cases: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        response = request_completion(base_url, model_id, case["prompt"], system_prompt)
        selected_id = find_selected_candidate(response["output"], case["candidates"])
        selected = next(
            (candidate for candidate in case["candidates"] if candidate["id"] == selected_id),
            None,
        )
        failures = (
            candidate_compliance(selected, case["scenario"])
            if selected
            else ["recommandation non identifiable"]
        )
        results.append(
            {
                "scenario_name": case["scenario"]["name"],
                "expected_ids": case["expected_ids"],
                "selected_model_id": selected_id,
                "selected_model_name": selected["name"] if selected else None,
                "decision_match": selected_id in case["expected_ids"],
                "candidate_grounded": selected is not None,
                "constraint_compliant": not failures,
                "constraint_failures": failures,
                "elapsed_seconds": response["elapsed_seconds"],
                "completion_tokens": response["completion_tokens"],
                "generation_tokens_per_second": response["generation_tokens_per_second"],
                "output": response["output"],
            }
        )
    return results


def escape(value: Any) -> str:
    return html.escape(str(value) if value is not None else "—")


def percentage(value: float | None) -> str:
    return "N/D" if value is None else f"{value * 100:.1f} %"


def build_report_html(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    fine = metrics[ADVISOR_MODEL_ID]
    baseline = metrics[BASE_MODEL_ID]
    low, high = fine["top1_wilson_95"]

    def card(title: str, value: str, detail: str) -> str:
        return (
            '<div class="card"><div class="label">'
            + escape(title)
            + '</div><div class="value">'
            + escape(value)
            + '</div><div class="detail">'
            + escape(detail)
            + "</div></div>"
        )

    def table_rows() -> str:
        rows = []
        results_by_model = report["results"]
        for index, case in enumerate(report["cases"]):
            predictions = {
                model_id: results_by_model[model_id][index]
                for model_id in (ADVISOR_MODEL_ID, BASE_MODEL_ID)
            }
            expected = ", ".join(case["expected_ids"])
            fine_result = predictions[ADVISOR_MODEL_ID]
            base_result = predictions[BASE_MODEL_ID]
            fine_status = "OK" if fine_result["decision_match"] else "NON"
            base_status = "OK" if base_result["decision_match"] else "NON"
            fine_constraint = "OK" if fine_result["constraint_compliant"] else "NON"
            rows.append(
                "<tr><td>"
                + escape(case["scenario"]["name"])
                + "</td><td>"
                + escape(expected)
                + "</td><td>"
                + escape(base_result["selected_model_id"] or "Non identifiée")
                + f"<br><b>{base_status}</b></td><td>"
                + escape(fine_result["selected_model_id"] or "Non identifiée")
                + f"<br><b>{fine_status}</b></td><td><b>{fine_constraint}</b>"
                + (
                    "<br>" + escape(", ".join(fine_result["constraint_failures"]))
                    if fine_result["constraint_failures"]
                    else ""
                )
                + "</td></tr>"
            )
        return "".join(rows)

    training = report["training"]
    train_loss = (
        f"{training['initial_loss']:.3f} → {training['final_loss']:.4f}"
        if training["initial_loss"] is not None and training["final_loss"] is not None
        else "non enregistré"
    )
    training_steps = (
        f"{training['current_step']}/{training['max_steps']} étapes"
        if training["current_step"] is not None and training["max_steps"] is not None
        else "non enregistré"
    )
    gguf_mib = report["gguf_bytes"] / (1024 * 1024)
    fine_ci = f"{low * 100:.1f}–{high * 100:.1f} %"
    difference = fine["top1_accuracy"] - baseline["top1_accuracy"]
    delta = f"{difference * 100:+.1f} points"
    scenario_cards = "".join(
        [
            card(
                "Décision top-1",
                percentage(fine["top1_accuracy"]),
                f"{fine['top1_correct']}/{fine['total']} cas; IC 95 % de Wilson : {fine_ci}",
            ),
            card(
                "Candidat cité",
                percentage(fine["grounded_rate"]),
                f"{fine['grounded']}/{fine['total']} réponses avec un modèle du top 3",
            ),
            card(
                "Fiches compatibles",
                percentage(fine["constraint_compliance_rate"]),
                f"{fine['constraint_compliant']}/{fine['total']} candidats conformes au catalogue",
            ),
            card(
                "Latence moyenne",
                f"{fine['mean_latency_seconds']:.2f} s",
                f"Médiane : {fine['median_latency_seconds']:.2f} s",
            ),
        ]
    )

    bars = []
    for model_id, label, colour in (
        (BASE_MODEL_ID, "Qwen de base", "#8290a5"),
        (ADVISOR_MODEL_ID, "Qwen affiné", "#5078e3"),
    ):
        value = metrics[model_id]["top1_accuracy"] * 100
        bars.append(
            '<div class="bar-row"><span>'
            + escape(label)
            + '</span><div class="track"><div class="bar" style="width:'
            + f"{value:.2f}%"
            + ";background:"
            + colour
            + '"></div></div><b>'
            + f"{value:.1f} %"
            + "</b></div>"
        )

    candidate_table = table_rows()
    train_metadata = report["training"]
    loss_reduction = (
        (train_metadata["initial_loss"] - train_metadata["final_loss"])
        / train_metadata["initial_loss"]
        * 100
        if train_metadata["initial_loss"] and train_metadata["final_loss"] is not None
        else None
    )
    loss_text = (
        f"baisse de {loss_reduction:.1f} %"
        if loss_reduction is not None
        else "évolution indisponible"
    )
    gguf_text = f"{gguf_mib:.0f} Mio ({gguf_mib / 1024:.2f} Gio)"
    models_count = report["catalog_model_count"]
    test_count = len(report["cases"])

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><style>
@page {{ size: A4 landscape; margin: 12mm 13mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Arial, Helvetica, sans-serif; color: #172033; font-size: 10pt; line-height: 1.42; }}
h1 {{ font-size: 26pt; margin: 0 0 4mm; color: #132449; }}
h2 {{ font-size: 17pt; margin: 0 0 5mm; color: #132449; }}
h3 {{ font-size: 11pt; margin: 4mm 0 1.5mm; color: #31466c; }}
p {{ margin: 2mm 0; }}
.muted {{ color: #59677e; }}
.small {{ font-size: 8.5pt; }}
.eyebrow {{ color: #5078e3; font-weight: bold; text-transform: uppercase; letter-spacing: 1.3px; }}
.page {{ page-break-after: always; }}
.page:last-child {{ page-break-after: auto; }}
.rule {{ border-top: 1px solid #dfe5ef; margin: 5mm 0; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm; margin: 7mm 0; }}
.card {{ background: #f3f6fb; border: 1px solid #e1e7f0; border-radius: 6px; padding: 4mm; }}
.label {{ font-size: 8pt; color: #59677e; }}
.value {{ font-size: 19pt; color: #1c3c82; font-weight: bold; margin: 1mm 0; }}
.detail {{ font-size: 8pt; color: #59677e; }}
.bar-row {{ display: grid; grid-template-columns: 35mm 1fr 18mm; gap: 4mm; align-items: center; margin: 2.5mm 0; }}
.track {{ height: 6mm; background: #edf0f5; border-radius: 4px; overflow: hidden; }}
.bar {{ height: 100%; }}
.columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10mm; }}
.box {{ border: 1px solid #dfe5ef; border-radius: 5px; padding: 4mm; background: #fbfcfe; }}
table {{ width: 100%; border-collapse: collapse; font-size: 8.2pt; }}
th {{ background: #19336c; color: white; text-align: left; padding: 2mm; }}
td {{ border-bottom: 1px solid #dfe5ef; padding: 1.8mm 2mm; vertical-align: top; overflow-wrap: anywhere; }}
tr:nth-child(even) td {{ background: #f5f7fb; }}
.case-table td:nth-child(1) {{ width: 21%; }}
.case-table td:nth-child(2) {{ width: 27%; }}
.case-table td:nth-child(3), .case-table td:nth-child(4) {{ width: 22%; }}
.case-table td:nth-child(5) {{ width: 8%; }}
.callout {{ padding: 4mm 5mm; border-left: 4px solid #e2a23b; background: #fff8eb; margin: 4mm 0; }}
.footer {{ position: fixed; bottom: -7mm; left: 0; right: 0; color: #7d8899; font-size: 8pt; }}
</style></head><body>
<div class="footer">LLMarchitect — Évaluation locale • {escape(report['date'])}</div>
<section class="page">
  <div class="eyebrow">Évaluation expérimentale</div>
  <h1>Le modèle affiné recommande-t-il mieux ?</h1>
  <p class="muted">Comparaison contrôlée entre l'adaptateur LLMarchitect et le modèle Qwen de base, sur {test_count} scénarios conçus hors du jeu d'entraînement.</p>
  <div class="rule"></div>
  <div class="cards">{scenario_cards}</div>
  <h2>Fidélité au classement du moteur</h2>
  {''.join(bars)}
  <div class="columns" style="margin-top:7mm">
    <div class="box"><h3>Résultat principal</h3>
      <p>Le modèle affiné retrouve le meilleur choix du moteur dans <b>{fine['top1_correct']} cas sur {fine['total']}</b>, contre <b>{baseline['top1_correct']}/{baseline['total']}</b> pour le modèle de base ({escape(delta)}).</p>
      <p>Une égalité de score au premier rang est acceptée. L'intervalle de confiance à 95 % du taux top-1 affiné est <b>{escape(fine_ci)}</b>.</p>
    </div>
    <div class="box"><h3>Ce que mesure ce chiffre</h3>
      <p>La cohérence de la recommandation écrite avec le classement calculé par LLMarchitect, le fait de choisir parmi les candidats présentés et le respect des contraintes vérifiables.</p>
      <p>Ce n'est pas un benchmark général de connaissances, ni une validation indépendante de la qualité des données du catalogue.</p>
    </div>
  </div>
</section>
<section class="page">
  <div class="eyebrow">Protocole & données</div>
  <h2>Comment les résultats ont été obtenus</h2>
  <div class="columns">
    <div>
      <h3>Modèles comparés</h3>
      <p><b>Affiné :</b> <code>{escape(ADVISOR_MODEL_ID)}</code>, servi par LM Studio avec l'export Q4_K_M LLMarchitect ({escape(gguf_text)}).</p>
      <p><b>Référence :</b> <code>{escape(BASE_MODEL_ID)}</code>, Qwen2.5-3B-Instruct exposé par le même serveur.</p>
      <p>Les deux reçoivent les mêmes prompts, le même catalogue, le même prompt système et les paramètres de production : température 0,3, maximum 800 tokens.</p>
      <h3>Jeu de test indépendant</h3>
      <p>{test_count} scénarios rédigés pour ce test : exécution locale, coût API, contexte long, vitesse, code, mathématiques, extraction et RAG. Ce ne sont pas des lignes du fichier d'entraînement ({report['dataset_count']} exemples), et aucun découpage aléatoire du jeu d'entraînement n'est revendiqué.</p>
      <p>Une seule génération a été demandée par modèle et scénario; les résultats sont donc une mesure ponctuelle, sensible à la variabilité de génération.</p>
    </div>
    <div>
      <h3>Référence de décision</h3>
      <p>Pour chaque scénario, le service de recommandation existant calcule le score des modèles depuis le catalogue SQLite local ({models_count} modèles). La sortie attendue est le meilleur modèle présenté dans le top 3; les ex æquo au score maximal comptent comme corrects.</p>
      <p>Le taux « candidat cité » exige que le premier modèle identifiable dans la réponse soit l'un des trois candidats fournis. Le contrôle des contraintes vérifie le contexte, les poids ouverts requis, la VRAM locale Q4 et, pour le cloud, les prix connus face au plafond demandé.</p>
      <div class="callout"><b>Limite importante :</b> le moteur sert d'oracle de cohérence, pas de vérité absolue. Un bon score signifie que le texte suit le classement actuel; il ne prouve pas que le classement, les prix ou les benchmarks sources sont exacts dans le monde réel.</div>
      <div class="callout"><b>Factualité observée :</b> la relecture a relevé des affirmations non étayées, dont un coût API annoncé à zéro pour une option cloud et un score de classification de 94,8 % absent des champs du catalogue. Ces affirmations ne sont pas incluses dans le taux top-1 et nécessitent une vérification humaine.</div>
    </div>
  </div>
  <div class="rule"></div>
  <h3>Entraînement observé dans le journal local</h3>
  <table><tr><th>Base</th><th>Étapes enregistrées</th><th>Perte entraînement</th><th>Matériel journalisé</th></tr>
  <tr><td>{escape(training['base_model'])}</td><td>{escape(training_steps)}</td><td>{escape(train_loss)} ({escape(loss_text)})</td><td>{escape(training['hardware'])}</td></tr></table>
  <p class="small muted">La perte est une mesure sur les données d'entraînement uniquement. Elle ne constitue pas une mesure de généralisation, de précision de recommandation ou de factualité.</p>
</section>
<section class="page">
  <div class="eyebrow">Résultats détaillés</div>
  <h2>Scénario par scénario</h2>
  <table class="case-table"><tr><th>Scénario</th><th>Choix de référence (ex æquo compris)</th><th>Qwen de base</th><th>Qwen affiné</th><th>Fiche candidat (affiné)</th></tr>
  {candidate_table}
  </table>
  <div class="rule"></div>
  <div class="columns">
    <div class="box"><h3>Indicateurs comparatifs</h3>
      <table><tr><th>Mesure</th><th>Base</th><th>Affiné</th></tr>
      <tr><td>Décision top-1</td><td>{baseline['top1_correct']}/{baseline['total']} ({percentage(baseline['top1_accuracy'])})</td><td>{fine['top1_correct']}/{fine['total']} ({percentage(fine['top1_accuracy'])})</td></tr>
      <tr><td>Candidat identifiable dans le top 3</td><td>{baseline['grounded']}/{baseline['total']} ({percentage(baseline['grounded_rate'])})</td><td>{fine['grounded']}/{fine['total']} ({percentage(fine['grounded_rate'])})</td></tr>
      <tr><td>Candidat compatible selon le catalogue</td><td>{baseline['constraint_compliant']}/{baseline['total']} ({percentage(baseline['constraint_compliance_rate'])})</td><td>{fine['constraint_compliant']}/{fine['total']} ({percentage(fine['constraint_compliance_rate'])})</td></tr>
      <tr><td>Latence moyenne</td><td>{baseline['mean_latency_seconds']:.2f} s</td><td>{fine['mean_latency_seconds']:.2f} s</td></tr>
      <tr><td>Débit observé (tokens de réponse / temps d'appel)</td><td>{baseline['mean_generation_tokens_per_second'] or 0:.1f} tok/s</td><td>{fine['mean_generation_tokens_per_second'] or 0:.1f} tok/s</td></tr>
      <tr><td>Tokens de sortie moyens</td><td>{baseline['mean_completion_tokens'] or 0:.1f}</td><td>{fine['mean_completion_tokens'] or 0:.1f}</td></tr>
      </table>
    </div>
    <div class="box"><h3>Conclusion opérationnelle</h3>
      <p>{escape(report['conclusion'])}</p>
      <p><b>Avant production :</b> vérifier manuellement les recommandations et prix, compléter un jeu de vérité terrain annoté par des experts, puis répéter le test sur plusieurs générations et avec un jeu de test plus large.</p>
    </div>
  </div>
</section>
</body></html>"""


def find_browser() -> str:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        str(Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")),
        str(Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")),
        str(Path("C:/Program Files/Google/Chrome/Application/chrome.exe")),
    ]
    browser = next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)
    if not browser:
        raise RuntimeError("Edge/Chrome est requis pour produire le PDF, mais aucun navigateur n'a été trouvé.")
    return browser


def write_pdf(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llmarchitect-pdf-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        html_path = temporary_path / "report.html"
        html_path.write_text(build_report_html(report), encoding="utf-8")
        command = [
            find_browser(),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            f"--user-data-dir={temporary_path / 'profile'}",
            f"--print-to-pdf={output_path.resolve()}",
            html_path.resolve().as_uri(),
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                "Échec de création du PDF avec Edge/Chrome. "
                f"Code={process.returncode}; stderr={process.stderr[-2000:]}"
            )
        with output_path.open("rb") as pdf_file:
            if pdf_file.read(5) != b"%PDF-":
                raise RuntimeError(f"Le fichier produit n'est pas un PDF valide : {output_path}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue l'advisor Qwen local et génère un rapport PDF.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--rebuild-report",
        action="store_true",
        help="Recalcule les métriques et le PDF depuis les réponses JSON existantes, sans appeler LM Studio.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "evaluation_results.json"
    report_path = args.output_dir / "evaluation_report.pdf"
    if args.rebuild_report:
        if not raw_path.is_file():
            raise RuntimeError(f"Résultats existants introuvables : {raw_path}")
        report = json.loads(raw_path.read_text(encoding="utf-8"))
        refresh_report_results(report, ADVISOR_MODEL_ID, BASE_MODEL_ID)
        report["date"] = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M")
        atomic_write_json(raw_path, report)
        write_pdf(report, report_path)
        for model_id, values in report["metrics"].items():
            print(
                f"{model_id}: top-1={values['top1_correct']}/{values['total']} "
                f"({percentage(values['top1_accuracy'])}), "
                f"candidat={percentage(values['grounded_rate'])}, "
                f"contraintes={percentage(values['constraint_compliance_rate'])}"
            )
        print(f"Rapport PDF : {report_path}")
        return

    models, context = load_models()
    cases = build_recommendations(models, context)
    loaded_models = request_json(f"{args.base_url.rstrip('/')}/models")
    loaded_ids = {item["id"] for item in loaded_models.get("data", [])}
    missing_models = {ADVISOR_MODEL_ID, BASE_MODEL_ID} - loaded_ids
    if missing_models:
        raise RuntimeError(
            f"Modèles requis absents de LM Studio : {sorted(missing_models)}. "
            f"Modèles exposés : {sorted(loaded_ids)}"
        )

    from app.services.recommender import SYSTEM_ADVISOR_PROMPT

    results: dict[str, list[dict[str, Any]]] = {
        ADVISOR_MODEL_ID: [],
        BASE_MODEL_ID: [],
    }
    evaluation_order = [ADVISOR_MODEL_ID, BASE_MODEL_ID]
    for index, case in enumerate(cases):
        order = evaluation_order if index % 2 == 0 else list(reversed(evaluation_order))
        for model_id in order:
            result = evaluate_model(
                args.base_url,
                model_id,
                [case],
                SYSTEM_ADVISOR_PROMPT,
            )[0]
            results[model_id].append(result)
            status = "OK" if result["decision_match"] else "DIFF"
            print(
                f"[{model_id}] {index + 1}/{len(cases)} {status} "
                f"{case['scenario']['name']}: "
                f"{result['selected_model_id'] or 'non identifié'}"
            )
            atomic_write_json(
                raw_path,
                {
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "results": results,
                },
            )

    metrics = {model_id: summarize(model_results) for model_id, model_results in results.items()}
    job = context["job"]
    loss_values = [
        float(value)
        for value in re.findall(r"loss=([0-9]+(?:\.[0-9]+)?)", job["logs"] or "")
    ]
    gguf_path = ROOT / "finetuning" / "outputs" / "llm_architect_gguf" / "Qwen2.5-3B-LLMarchitect-Q4_K_M.gguf"
    training = {
        "base_model": job["base_model"],
        "max_steps": job["max_steps"],
        "current_step": job["current_step"],
        "final_loss": job["loss"],
        "initial_loss": loss_values[0] if loss_values else None,
        "hardware": (
            re.search(r"on (NVIDIA .+? GPU \([^)]+\))", job["logs"] or "").group(1)
            if re.search(r"on (NVIDIA .+? GPU \([^)]+\))", job["logs"] or "")
            else "non enregistré"
        ),
    }
    conclusion = (
        f"Sur ce jeu de {len(cases)} cas, le modèle affiné atteint "
        f"{metrics[ADVISOR_MODEL_ID]['top1_correct']}/{len(cases)} décisions conformes au moteur "
        f"({percentage(metrics[ADVISOR_MODEL_ID]['top1_accuracy'])}), contre "
        f"{metrics[BASE_MODEL_ID]['top1_correct']}/{len(cases)} "
        f"({percentage(metrics[BASE_MODEL_ID]['top1_accuracy'])}) pour le modèle de base. "
        f"L'écart observé est de "
        f"{(metrics[ADVISOR_MODEL_ID]['top1_accuracy'] - metrics[BASE_MODEL_ID]['top1_accuracy']) * 100:+.1f} points. "
        "Avec un seul échantillon de génération par scénario, cet écart est indicatif, pas une preuve de supériorité."
    )
    report = {
        "date": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M"),
        "catalog_model_count": len(models),
        "dataset_count": read_dataset_count(),
        "gguf_bytes": gguf_path.stat().st_size if gguf_path.is_file() else 0,
        "training": training,
        "cases": cases,
        "results": results,
        "metrics": metrics,
        "conclusion": conclusion,
    }
    atomic_write_json(raw_path, report)
    write_pdf(report, report_path)

    print("\nRésumé de l'évaluation :")
    for model_id, values in metrics.items():
        print(
            f"{model_id}: top-1={values['top1_correct']}/{values['total']} "
            f"({percentage(values['top1_accuracy'])}), "
            f"candidat={percentage(values['grounded_rate'])}, "
            f"contraintes={percentage(values['constraint_compliance_rate'])}, "
            f"latence moyenne={values['mean_latency_seconds']:.2f}s"
        )
    print(f"Rapport PDF : {report_path}")
    print(f"Résultats bruts : {raw_path}")


if __name__ == "__main__":
    main()
