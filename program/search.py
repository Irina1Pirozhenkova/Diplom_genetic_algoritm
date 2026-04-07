from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from .model import EnvironmentShockConfig, ModelConfig, SearchResult
from .render import render_curve
from .scoring import score_curve
from .simulator import default_environment_shocks, simulate


DEFAULT_TRIALS = 120
DEFAULT_SEEDS_PER_TRIAL = 5
BASE_IMAGE_NAME = "passionarity_base.png"
SHOCK_LABEL = "\u0420\u0435\u0437\u043a\u043e\u0435 \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0441\u0440\u0435\u0434\u044b"
FIXED_VOCATION_DIM = 4
VARIATION_FACTOR = 3.0

BASE_TITLE = "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u043f\u0430\u0441\u0441\u0438\u043e\u043d\u0430\u0440\u043d\u043e\u0441\u0442\u044c"
ENERGY_TITLE = "\u041f\u0430\u0441\u0441\u0438\u043e\u043d\u0430\u0440\u043d\u043e\u0441\u0442\u044c: \u0432\u0430\u0440\u0438\u0430\u0446\u0438\u044f E"
LOYALTY_TITLE = "\u041f\u0430\u0441\u0441\u0438\u043e\u043d\u0430\u0440\u043d\u043e\u0441\u0442\u044c: \u0432\u0430\u0440\u0438\u0430\u0446\u0438\u044f K"
LATE_SHOCK_TITLE = "\u0420\u0435\u0437\u043a\u043e\u0435 \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0441\u0440\u0435\u0434\u044b: \u043f\u043e\u0437\u0434\u043d\u0438\u0439 \u0443\u0434\u0430\u0440"
EARLY_SHOCK_TITLE = "\u0420\u0435\u0437\u043a\u043e\u0435 \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0441\u0440\u0435\u0434\u044b: \u0440\u0430\u043d\u043d\u0438\u0439 \u0443\u0434\u0430\u0440"
CRITERION_TITLE = "\u041e\u0431\u0449\u0435\u0441\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0439 \u043a\u0440\u0438\u0442\u0435\u0440\u0438\u0439 \u044d\u0444\u0444\u0435\u043a\u0442\u0438\u0432\u043d\u043e\u0441\u0442\u0438"

BASE_COLOR = "black"
UP_COLOR = "#8b1e1e"
DOWN_COLOR = "#1f4e8c"


def sample_config(rng: np.random.Generator) -> ModelConfig:
    # Random search keeps vocation_dim fixed because the public experiments use four vocation axes.
    population_size = int(rng.integers(80, 241))
    return ModelConfig(
        population_size=population_size,
        vocation_dim=FIXED_VOCATION_DIM,
        mutation_rate=float(rng.uniform(0.01, 0.18)),
        mutation_sigma=float(rng.uniform(0.05, 0.9)),
        landscape_energy=float(rng.uniform(0.9 * population_size, 1.8 * population_size)),
        landscape_loyalty=float(rng.uniform(0.15, 1.4)),
        selection_alpha=float(rng.uniform(0.2, 1.8)),
        generations=int(rng.integers(900, 2201)),
        seed=int(rng.integers(1, 1_000_000)),
        wave_strength=float(rng.uniform(0.04, 0.18)),
        wave_count=int(rng.integers(10, 19)),
        wave_irregularity=float(rng.uniform(0.05, 0.30)),
        tail_wave_strength=float(rng.uniform(0.01, 0.08)),
        tail_wave_count=int(rng.integers(2, 5)),
        tail_decay_power=float(rng.uniform(1.4, 3.0)),
    )


def run_search(
    trials: int = DEFAULT_TRIALS,
    seeds_per_trial: int = DEFAULT_SEEDS_PER_TRIAL,
    search_seed: int = 42,
) -> SearchResult:
    # Random search checks several seeds per template so the chosen config is not just a lucky run.
    rng = np.random.default_rng(search_seed)
    best_result: SearchResult | None = None

    for _ in range(trials):
        template = sample_config(rng)
        candidate_best: SearchResult | None = None

        for _ in range(seeds_per_trial):
            config = ModelConfig(**asdict(template))
            config.seed = int(rng.integers(1, 1_000_000))
            simulation = simulate(config)
            score, metrics = score_curve(simulation.passionarity)
            result = SearchResult(
                config=config,
                score=score,
                metrics=metrics,
                passionarity=simulation.passionarity,
            )
            if candidate_best is None or result.score > candidate_best.score:
                candidate_best = result

        if candidate_best is not None and (best_result is None or candidate_best.score > best_result.score):
            best_result = candidate_best

    if best_result is None:
        raise RuntimeError("Search failed to produce a candidate.")
    return best_result


def _derive_output_paths(base_output: str | Path) -> dict[str, Path]:
    output_dir = Path(base_output).parent
    return {
        "base_passionarity": output_dir / "passionarity_base.png",
        "energy_comparison": output_dir / "passionarity_E.png",
        "loyalty_comparison": output_dir / "passionarity_K.png",
        "shock_1072": output_dir / "shock_1072.png",
        "shock_647": output_dir / "shock_647.png",
        "criterion_c_base": output_dir / "criterion_C_base.png",
    }


def _clone_config(config: ModelConfig) -> ModelConfig:
    return ModelConfig(**asdict(config))


def _experimental_base_config(config: ModelConfig) -> ModelConfig:
    # Every rendered experiment uses the article-inspired four-dimensional vocation interpretation.
    base = _clone_config(config)
    base.vocation_dim = FIXED_VOCATION_DIM
    return base


def _legend_base_lines(config: ModelConfig) -> list[tuple[str, str]]:
    return [
        (
            f"I = {config.vocation_dim} "
            "(\u0432\u043e\u0435\u043d\u043d\u0430\u044f \u0441\u0438\u043b\u0430, "
            "\u044d\u043a\u043e\u043d\u043e\u043c\u0438\u043a\u0430, "
            "\u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044f, "
            "\u043a\u0443\u043b\u044c\u0442\u0443\u0440\u0430)",
            "black",
        ),
        (f"n = {config.population_size}", "black"),
        (f"P0 = {config.mutation_rate:.4f}", "black"),
        (f"E = {config.landscape_energy:.2f}", "black"),
        (f"K = {config.landscape_loyalty:.4f}", "black"),
    ]


def _legend_lines_for_energy(config: ModelConfig, up: ModelConfig, down: ModelConfig) -> list[tuple[str, str]]:
    return [
        (f"E = {config.landscape_energy:.2f}", BASE_COLOR),
        (f"E x3 = {up.landscape_energy:.2f}", UP_COLOR),
        (f"E /3 = {down.landscape_energy:.2f}", DOWN_COLOR),
        (f"I = {config.vocation_dim} (\u0432\u043e\u0435\u043d\u043d\u0430\u044f \u0441\u0438\u043b\u0430, \u044d\u043a\u043e\u043d\u043e\u043c\u0438\u043a\u0430, \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044f, \u043a\u0443\u043b\u044c\u0442\u0443\u0440\u0430)", "black"),
        (f"n = {config.population_size}", "black"),
        (f"P0 = {config.mutation_rate:.4f}", "black"),
        (f"K = {config.landscape_loyalty:.4f}", "black"),
    ]


def _legend_lines_for_loyalty(config: ModelConfig, up: ModelConfig, down: ModelConfig) -> list[tuple[str, str]]:
    return [
        (f"K = {config.landscape_loyalty:.4f}", BASE_COLOR),
        (f"K x3 = {up.landscape_loyalty:.4f}", UP_COLOR),
        (f"K /3 = {down.landscape_loyalty:.4f}", DOWN_COLOR),
        (f"I = {config.vocation_dim} (\u0432\u043e\u0435\u043d\u043d\u0430\u044f \u0441\u0438\u043b\u0430, \u044d\u043a\u043e\u043d\u043e\u043c\u0438\u043a\u0430, \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044f, \u043a\u0443\u043b\u044c\u0442\u0443\u0440\u0430)", "black"),
        (f"n = {config.population_size}", "black"),
        (f"P0 = {config.mutation_rate:.4f}", "black"),
        (f"E = {config.landscape_energy:.2f}", "black"),
    ]


def _base_variants(config: ModelConfig) -> dict[str, ModelConfig]:
    # Each comparison changes only one key parameter while keeping the rest identical.
    base_config = _experimental_base_config(config)
    energy_up = _clone_config(base_config)
    energy_up.landscape_energy *= VARIATION_FACTOR
    energy_down = _clone_config(base_config)
    energy_down.landscape_energy /= VARIATION_FACTOR
    loyalty_up = _clone_config(base_config)
    loyalty_up.landscape_loyalty *= VARIATION_FACTOR
    loyalty_down = _clone_config(base_config)
    loyalty_down.landscape_loyalty /= VARIATION_FACTOR
    return {
        "base": base_config,
        "energy_up": energy_up,
        "energy_down": energy_down,
        "loyalty_up": loyalty_up,
        "loyalty_down": loyalty_down,
    }


def _build_shock_outputs() -> dict[str, tuple[EnvironmentShockConfig, str]]:
    outputs: dict[str, tuple[EnvironmentShockConfig, str]] = {}
    for shock in default_environment_shocks():
        if shock.name == "energy_shock_only":
            outputs["shock_1072"] = (shock, LATE_SHOCK_TITLE)
        else:
            outputs["shock_647"] = (shock, EARLY_SHOCK_TITLE)
    return outputs


def _save_experiment_scenarios(
    variants: dict[str, ModelConfig],
    shocks: dict[str, tuple[EnvironmentShockConfig, str]],
    output_dir: Path,
) -> Path:
    # This JSON explains exactly which configurations were rendered for the six output graphs.
    scenario_path = output_dir / "experiment_scenarios.json"
    base_config = variants["base"]
    payload: dict[str, object] = {
        "base_passionarity": {"title": BASE_TITLE, "config": base_config.to_dict()},
        "energy_comparison": {
            "title": ENERGY_TITLE,
            "base": base_config.to_dict(),
            "up": variants["energy_up"].to_dict(),
            "down": variants["energy_down"].to_dict(),
            "factor": VARIATION_FACTOR,
        },
        "loyalty_comparison": {
            "title": LOYALTY_TITLE,
            "base": base_config.to_dict(),
            "up": variants["loyalty_up"].to_dict(),
            "down": variants["loyalty_down"].to_dict(),
            "factor": VARIATION_FACTOR,
        },
        "criterion_c_base": {
            "title": CRITERION_TITLE,
            "config": base_config.to_dict(),
            "rule": "C(t) = mean(survival_score(population_t))",
        },
    }
    for key, (shock, title) in shocks.items():
        payload[key] = {
            "title": title,
            "base_config": base_config.to_dict(),
            "shock": shock.to_dict(),
            "generation": int(round(shock.start_ratio * base_config.generations)),
        }
    scenario_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return scenario_path


def render_all_outputs(config: ModelConfig, base_output: str | Path) -> dict[str, Path]:
    # One configuration produces six deliverables: three baseline comparisons, two shocks, and C(t).
    output_paths = _derive_output_paths(base_output)
    output_paths["base_passionarity"].parent.mkdir(parents=True, exist_ok=True)

    variants = _base_variants(config)
    base_result = simulate(variants["base"])
    render_curve(
        base_result.passionarity,
        output_paths["base_passionarity"],
        title=BASE_TITLE,
        legend_lines=_legend_base_lines(variants["base"]),
        y_min=0.0,
    )

    energy_up_result = simulate(variants["energy_up"])
    energy_down_result = simulate(variants["energy_down"])
    render_curve(
        [
            (f"E = {variants['base'].landscape_energy:.2f}", base_result.passionarity, BASE_COLOR),
            ("E x3", energy_up_result.passionarity, UP_COLOR),
            ("E /3", energy_down_result.passionarity, DOWN_COLOR),
        ],
        output_paths["energy_comparison"],
        title=ENERGY_TITLE,
        legend_lines=_legend_lines_for_energy(variants["base"], variants["energy_up"], variants["energy_down"]),
        y_min=0.0,
    )

    loyalty_up_result = simulate(variants["loyalty_up"])
    loyalty_down_result = simulate(variants["loyalty_down"])
    render_curve(
        [
            (f"K = {variants['base'].landscape_loyalty:.4f}", base_result.passionarity, BASE_COLOR),
            ("K x3", loyalty_up_result.passionarity, UP_COLOR),
            ("K /3", loyalty_down_result.passionarity, DOWN_COLOR),
        ],
        output_paths["loyalty_comparison"],
        title=LOYALTY_TITLE,
        legend_lines=_legend_lines_for_loyalty(variants["base"], variants["loyalty_up"], variants["loyalty_down"]),
        y_min=0.0,
    )

    render_curve(
        base_result.criterion_c,
        output_paths["criterion_c_base"],
        title=CRITERION_TITLE,
        legend_lines=_legend_base_lines(variants["base"]),
        y_label="\u041e\u0431\u0449\u0435\u0441\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0439 \u043a\u0440\u0438\u0442\u0435\u0440\u0438\u0439 \u044d\u0444\u0444\u0435\u043a\u0442\u0438\u0432\u043d\u043e\u0441\u0442\u0438",
    )

    shocks = _build_shock_outputs()
    for key, (shock, title) in shocks.items():
        # Shock graphs reuse the same baseline config so only the shock timing changes.
        shocked_result = simulate(variants["base"], shock=shock)
        marker_x = int(round(shock.start_ratio * variants["base"].generations))
        marker_x = min(max(marker_x, 0), len(shocked_result.passionarity) - 1)
        marker_y = shocked_result.passionarity[marker_x]
        render_curve(
            shocked_result.passionarity,
            output_paths[key],
            title=title,
            marker_x=marker_x,
            marker_y=marker_y,
            marker_label=SHOCK_LABEL,
            shock_axis_label=str(marker_x),
            legend_lines=_legend_base_lines(variants["base"]),
            y_min=0.0,
        )

    _save_experiment_scenarios(variants, shocks, output_paths["base_passionarity"].parent)
    return output_paths


def save_search_result(result: SearchResult, output_dir: str | Path) -> dict[str, Path]:
    # Search persists both the best configuration and the exact visual package built from it.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config_path = output / "best_config.json"
    metadata_path = output / "search_result.json"
    experimental_config = _experimental_base_config(result.config)

    experimental_config.save_json(config_path)
    metadata = {
        "score": result.score,
        "metrics": result.metrics,
        "config": experimental_config.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return render_all_outputs(experimental_config, output / BASE_IMAGE_NAME)
