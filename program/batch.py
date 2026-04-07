from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import os
import re
import shutil
import time

from .model import ModelConfig
from .search import BASE_IMAGE_NAME, FIXED_VOCATION_DIM, render_all_outputs


DEFAULT_SCENARIO_PATH = Path("scenarios/scenario_1.json")


def _sanitize_name(value: str) -> str:
    """Convert free-form names into safe directory names for run folders."""

    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "scenario"


def _resolve_workers(run_count: int, workers: int | None = None) -> int:
    """Cap the worker count by both CPU capacity and number of runs."""

    if run_count <= 0:
        raise ValueError("Scenario must contain at least one run.")
    if workers is not None:
        return max(1, min(int(workers), run_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, run_count))


def _resolve_relative_path(path_value: str, relative_to: Path) -> Path:
    """Resolve scenario-local paths, including the base config path."""

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (relative_to / candidate).resolve()


def load_scenario(path: str | Path) -> tuple[dict[str, object], Path]:
    """Load and validate a scenario JSON description."""

    scenario_path = Path(path).resolve()
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scenario JSON must be an object.")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Scenario JSON must contain a 'runs' list.")
    if not runs:
        raise ValueError("Scenario 'runs' list must not be empty.")
    if "base_config_path" not in payload:
        raise ValueError("Scenario JSON must contain 'base_config_path'.")
    return payload, scenario_path


def build_effective_config(base_config: ModelConfig, overrides: dict[str, object]) -> ModelConfig:
    """Apply only explicit overrides and keep the shared model structure fixed."""

    allowed_keys = set(asdict(base_config))
    unknown_keys = sorted(set(overrides) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown override keys: {', '.join(unknown_keys)}")

    effective = ModelConfig(**asdict(base_config))
    for key, value in overrides.items():
        setattr(effective, key, value)

    # All public experiments keep the same four-vocation interpretation from the article.
    effective.vocation_dim = FIXED_VOCATION_DIM
    return effective


def _prepare_runs(scenario: dict[str, object], scenario_path: Path) -> tuple[ModelConfig, list[dict[str, object]], str]:
    """Expand a scenario JSON into fully materialized run tasks."""

    base_config_path = _resolve_relative_path(str(scenario["base_config_path"]), scenario_path.parent)
    base_config = ModelConfig.load_json(base_config_path)
    base_config.vocation_dim = FIXED_VOCATION_DIM

    prepared_runs: list[dict[str, object]] = []
    for index, raw_run in enumerate(scenario["runs"], start=1):
        if not isinstance(raw_run, dict):
            raise ValueError("Each scenario run must be an object.")

        name = _sanitize_name(str(raw_run.get("name", f"run_{index:02d}")))
        overrides = raw_run.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("Run overrides must be an object.")

        effective = build_effective_config(base_config, overrides)
        prepared_runs.append(
            {
                "index": index,
                "name": name,
                "overrides": overrides,
                "config": effective.to_dict(),
            }
        )

    scenario_name = _sanitize_name(str(scenario.get("scenario_name", scenario_path.stem)))
    return base_config, prepared_runs, scenario_name


def _run_worker(task: dict[str, object]) -> dict[str, object]:
    """Render one isolated run inside its own output directory."""

    run_dir = Path(str(task["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(**task["config"])
    config.save_json(run_dir / "effective_config.json")

    started = time.perf_counter()
    try:
        outputs = render_all_outputs(config, run_dir / BASE_IMAGE_NAME)
        status = "completed"
        error = None
    except Exception as exc:  # pragma: no cover - defensive manifest path
        outputs = {}
        status = "failed"
        error = str(exc)
    duration = time.perf_counter() - started

    return {
        "run_name": task["run_name"],
        "output_dir": str(run_dir),
        "seed": config.seed,
        "overrides": task["overrides"],
        "status": status,
        "duration_seconds": round(duration, 4),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "error": error,
    }


def run_batch_scenario(
    scenario_path: str | Path,
    output_root: str | Path = "scheduler_runs",
    workers: int | None = None,
) -> Path:
    """Execute all scenario runs and collect them into one timestamped package."""

    scenario, resolved_scenario_path = load_scenario(scenario_path)
    _, prepared_runs, scenario_name = _prepare_runs(scenario, resolved_scenario_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    package_dir = Path(output_root) / f"{timestamp}_{scenario_name}"
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy the exact scenario file used so the batch package is self-contained and reproducible.
    shutil.copy2(resolved_scenario_path, package_dir / "scenario.json")

    tasks: list[dict[str, object]] = []
    for run in prepared_runs:
        run_dir = package_dir / f"run_{int(run['index']):02d}_{run['name']}"
        tasks.append(
            {
                "run_name": run["name"],
                "run_dir": str(run_dir),
                "config": run["config"],
                "overrides": run["overrides"],
            }
        )

    resolved_workers = _resolve_workers(len(tasks), workers)
    results: list[dict[str, object]] = []

    # Single-process mode is useful for deterministic debugging and unit tests.
    if resolved_workers == 1:
        for task in tasks:
            results.append(_run_worker(task))
    else:
        # Multi-process mode accelerates independent runs because each run renders its own files.
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_map = {executor.submit(_run_worker, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - defensive manifest path
                    results.append(
                        {
                            "run_name": task["run_name"],
                            "output_dir": task["run_dir"],
                            "seed": task["config"]["seed"],
                            "overrides": task["overrides"],
                            "status": "failed",
                            "duration_seconds": 0.0,
                            "outputs": {},
                            "error": str(exc),
                        }
                    )

    results.sort(key=lambda item: item["run_name"])
    completed = sum(1 for item in results if item["status"] == "completed")
    manifest = {
        "scenario_name": scenario_name,
        "workers": resolved_workers,
        "runs_total": len(results),
        "runs_completed": completed,
        "runs_failed": len(results) - completed,
        "runs": results,
    }
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return package_dir
