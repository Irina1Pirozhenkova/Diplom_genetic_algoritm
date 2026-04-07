from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class ModelConfig:
    """Static parameters of the ethnogenesis model and its observable curve."""

    population_size: int
    vocation_dim: int
    mutation_rate: float
    mutation_sigma: float
    landscape_energy: float
    landscape_loyalty: float
    selection_alpha: float
    generations: int
    seed: int
    wave_strength: float = 0.10
    wave_count: int = 12
    wave_irregularity: float = 0.14
    tail_wave_strength: float = 0.03
    tail_wave_count: int = 3
    tail_decay_power: float = 2.2

    def to_dict(self) -> dict[str, int | float]:
        # JSON serialization is used both for the best config and for scheduler run manifests.
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        # Config files are always saved in UTF-8 so scenario files and README examples stay portable.
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelConfig":
        # Loading through one method keeps the scheduler, CLI and tests consistent.
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(slots=True)
class EnvironmentShockConfig:
    """External short-term shock applied to the environment after a chosen generation."""

    name: str
    start_ratio: float
    transition_generations: int
    energy_multiplier_after: float
    loyalty_multiplier_after: float
    recovery_ratio: float
    recovery_rate: float

    def to_dict(self) -> dict[str, str | int | float]:
        # Shock settings are exported into experiment_scenarios.json for transparency.
        return asdict(self)


@dataclass(slots=True)
class SimulationResult:
    """Raw and observed time series collected during a single simulation run."""

    config: ModelConfig
    passionarity: list[float]
    criterion_c: list[float]
    raw_passionarity: list[float]
    population_energy: list[float]
    capacity_ratio: list[float]
    observation_filter: list[float]
    wave_component: list[float]
    tail_component: list[float]
    effective_energy_limit: list[float]
    effective_loyalty: list[float]
    accepted_births: int
    rejected_births: int


@dataclass(slots=True)
class SearchResult:
    """Best configuration found by the random search with its score and curve."""

    config: ModelConfig
    score: float
    metrics: dict[str, float]
    passionarity: list[float]

    def save_json(self, path: str | Path) -> None:
        payload = {
            "score": self.score,
            "metrics": self.metrics,
            "config": self.config.to_dict(),
            "passionarity": self.passionarity,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
