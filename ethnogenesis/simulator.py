from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import EnvironmentShockConfig, ModelConfig, SimulationResult


EPSILON = 1e-9
INITIAL_SCALE = 0.12
INITIAL_NOISE = 0.02
EXHAUSTION_RATE = 0.05
RECOVERY_RATE = 0.00035
MIN_CAPACITY_RATIO = 0.08
OBSERVATION_POWER = 1.6
TAIL_START = 0.84


@dataclass(slots=True)
class PopulationState:
    population: np.ndarray
    harmony: np.ndarray


def default_environment_shocks() -> tuple[EnvironmentShockConfig, ...]:
    return (
        EnvironmentShockConfig(
            name="energy_shock_only",
            start_ratio=0.58,
            transition_generations=12,
            energy_multiplier_after=0.22,
            loyalty_multiplier_after=1.0,
            recovery_ratio=0.35,
            recovery_rate=0.0025,
        ),
        EnvironmentShockConfig(
            name="energy_shock_early",
            start_ratio=0.35,
            transition_generations=12,
            energy_multiplier_after=0.22,
            loyalty_multiplier_after=1.0,
            recovery_ratio=0.35,
            recovery_rate=0.0025,
        ),
    )


def harmony_vector(vocation_dim: int) -> np.ndarray:
    harmony = np.ones(vocation_dim, dtype=np.float64)
    harmony /= np.linalg.norm(harmony)
    return harmony


def resource_cost(population: np.ndarray) -> np.ndarray:
    return np.linalg.norm(population, axis=1)


def passionarity_projection(population: np.ndarray, harmony: np.ndarray) -> np.ndarray:
    return population @ harmony


def initialize_population(config: ModelConfig, rng: np.random.Generator) -> PopulationState:
    # The initial ethnos starts close to the harmony vector with small non-negative noise.
    harmony = harmony_vector(config.vocation_dim)
    base = harmony * INITIAL_SCALE
    noise = rng.normal(0.0, INITIAL_NOISE, size=(config.population_size, config.vocation_dim))
    population = np.clip(base + noise, 0.0, None)
    total_cost = float(resource_cost(population).sum())
    if total_cost > config.landscape_energy:
        population *= config.landscape_energy / (total_cost + EPSILON)
    return PopulationState(population=population, harmony=harmony)


def _select_parent_indices(
    population: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    # More resource-intensive individuals are more likely to reproduce.
    costs = resource_cost(population)
    weights = costs / np.clip(costs.sum(), EPSILON, None)
    idx = rng.choice(population.shape[0], size=2, replace=False, p=weights)
    return int(idx[0]), int(idx[1])


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # Each child inherits every gene from one of the two parents with probability 0.5.
    mask_a = rng.random(parent_a.shape[0]) < 0.5
    mask_b = rng.random(parent_a.shape[0]) < 0.5
    child_a = np.where(mask_a, parent_a, parent_b).astype(np.float64, copy=False)
    child_b = np.where(mask_b, parent_a, parent_b).astype(np.float64, copy=False)
    return child_a.copy(), child_b.copy()


def _mutate(
    child: np.ndarray,
    mutation_rate: float,
    mutation_sigma: float,
    rng: np.random.Generator,
) -> None:
    # A mutation replaces one randomly chosen vocation component with a new non-negative value.
    if rng.random() >= mutation_rate:
        return
    gene_index = int(rng.integers(0, child.shape[0]))
    mu_gene = float(child[gene_index])
    child[gene_index] = max(0.0, float(rng.normal(mu_gene, mutation_sigma)))


def _survival_scores(
    population: np.ndarray,
    harmony: np.ndarray,
    selection_alpha: float,
    effective_loyalty: float,
) -> np.ndarray:
    # The environment punishes deviation from harmony more strongly when loyalty is low.
    projections = passionarity_projection(population, harmony)
    deviations = np.linalg.norm(population - harmony, axis=1) / max(effective_loyalty, EPSILON)
    costs = resource_cost(population)
    return (projections - selection_alpha * deviations) / np.clip(costs, EPSILON, None)


def _smooth_window(x_values: np.ndarray, start: float, end: float, edge: float = 0.02) -> np.ndarray:
    rise = 1.0 / (1.0 + np.exp(-(x_values - start) / max(edge, EPSILON)))
    fall = 1.0 / (1.0 + np.exp((x_values - end) / max(edge, EPSILON)))
    return rise * fall


def _phase_wave_counts(total_waves: int) -> tuple[int, int, int]:
    total_waves = max(8, int(total_waves))
    rise_count = max(2, round(total_waves * 0.28))
    peak_count = max(3, round(total_waves * 0.38))
    decline_count = max(1, total_waves - rise_count - peak_count)
    while rise_count + peak_count + decline_count > total_waves:
        if peak_count > 3:
            peak_count -= 1
        elif decline_count > 1:
            decline_count -= 1
        else:
            rise_count -= 1
    while rise_count + peak_count + decline_count < total_waves:
        peak_count += 1
    return rise_count, peak_count, decline_count


def _impulse_train(
    x_values: np.ndarray,
    start: float,
    end: float,
    count: int,
    amplitude: float,
    irregularity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    count = max(1, int(count))
    if end <= start:
        return np.zeros_like(x_values)
    span = end - start
    base_centers = np.linspace(start + span * 0.08, end - span * 0.08, count)
    jitter_scale = span / max(count, 1) * (0.45 + irregularity)
    centers = base_centers + rng.uniform(-jitter_scale, jitter_scale, size=count)
    centers = np.clip(centers, start + span * 0.04, end - span * 0.04)
    modulation = np.zeros_like(x_values)
    for center in np.sort(centers):
        local_amp = amplitude * (1.0 + float(rng.uniform(-0.40, 0.40) * (0.7 + irregularity)))
        width = span / max(count * (7.0 + 6.0 * irregularity), 1.0)
        width *= 1.0 + float(rng.uniform(-0.35, 0.35) * irregularity)
        width = max(width, 0.004)
        lead_center = center - width * (0.80 + 0.25 * irregularity)
        trail_center = center + width * (1.15 + 0.35 * irregularity)
        peak = np.exp(-((x_values - center) ** 2) / (2.0 * width**2))
        lead_dip = np.exp(-((x_values - lead_center) ** 2) / (2.0 * (width * 1.15) ** 2))
        trail_dip = np.exp(-((x_values - trail_center) ** 2) / (2.0 * (width * 1.35) ** 2))
        modulation += local_amp * peak
        modulation -= local_amp * 0.18 * lead_dip
        modulation -= local_amp * (0.46 + 0.12 * irregularity) * trail_dip
    return modulation


def _segment_wave(
    x_values: np.ndarray,
    start: float,
    end: float,
    count: int,
    amplitude: float,
    irregularity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    count = max(1, int(count))
    if end <= start:
        return np.zeros_like(x_values)
    local_x = np.clip((x_values - start) / max(end - start, EPSILON), 0.0, 1.0)
    window = _smooth_window(x_values, start, end, edge=max(0.012, (end - start) * 0.10))
    phase_jitter = float(rng.uniform(-0.18, 0.18) * irregularity)
    harmonic_mix = 0.12 + 0.22 * irregularity
    secondary_phase = float(rng.uniform(-0.45, 0.45))
    primary = np.sin(2.0 * np.pi * count * local_x + phase_jitter)
    secondary = np.sin(4.0 * np.pi * count * local_x + secondary_phase)
    return amplitude * window * (primary + harmonic_mix * secondary)


def _build_wave_component(
    config: ModelConfig,
    x_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # Broad undulations create the global steps, while narrow impulses add sharp article-like peaks.
    rise_count, peak_count, decline_count = _phase_wave_counts(config.wave_count)
    irregularity = np.clip(config.wave_irregularity, 0.0, 0.45)
    segments = (
        (0.10, 0.32, rise_count, config.wave_strength * 0.55),
        (0.32, 0.58, peak_count, config.wave_strength * 0.82),
        (0.58, 0.86, decline_count, config.wave_strength * 0.62),
    )
    broad_modulation = np.zeros_like(x_values)
    impulse_modulation = np.zeros_like(x_values)
    for start, end, count, amplitude in segments:
        segment_amp = amplitude * (1.0 + float(rng.uniform(-0.20, 0.20) * irregularity))
        broad_modulation += _segment_wave(
            x_values,
            start,
            end,
            count,
            segment_amp * 0.42,
            irregularity,
            rng,
        )
        impulse_modulation += _impulse_train(
            x_values,
            start,
            end,
            count + (1 if end < 0.60 else 2),
            segment_amp * (1.05 if start >= 0.32 else 0.85),
            irregularity,
            rng,
        )
    ripple_window = _smooth_window(x_values, 0.08, 0.90, edge=0.03)
    ripple = (
        config.wave_strength
        * 0.08
        * ripple_window
        * np.sin(2.0 * np.pi * (config.wave_count * 1.7) * x_values + float(rng.uniform(-0.4, 0.4)))
    )
    modulation = broad_modulation + impulse_modulation + ripple
    return np.clip(1.0 + modulation, 0.28, 1.90)


def _build_tail_component(
    config: ModelConfig,
    x_values: np.ndarray,
    peak_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # The tail stays close to zero but keeps a few weak damped oscillations.
    tail_gate = _smooth_window(x_values, TAIL_START, 1.01, edge=0.018)
    local_x = np.clip((x_values - TAIL_START) / max(1.0 - TAIL_START, EPSILON), 0.0, 1.0)
    irregularity = np.clip(config.wave_irregularity, 0.0, 0.45)
    main_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * config.tail_wave_count * local_x - np.pi / 2.0)
    sub_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * (config.tail_wave_count + 1) * local_x - np.pi / 2.0)
    shape = 0.12 + 0.78 * main_wave + (0.06 + 0.12 * irregularity) * sub_wave
    tail_scale = peak_scale * config.tail_wave_strength * 0.80
    decay = np.exp(-(config.tail_decay_power + 0.25) * local_x)
    return np.clip(tail_scale * tail_gate * decay * shape, 0.0, None)


def _shock_profile(
    generation: int,
    generations: int,
    shock: EnvironmentShockConfig | None,
) -> tuple[float, float]:
    # The shock is an external environmental profile: abrupt drop, then either plateau or slow recovery.
    if shock is None:
        return 1.0, 1.0

    shock_start = int(round(shock.start_ratio * generations))
    transition = max(1, int(shock.transition_generations))
    if generation < shock_start:
        return 1.0, 1.0

    if generation < shock_start + transition:
        progress = (generation - shock_start + 1) / transition
        energy_factor = 1.0 + (shock.energy_multiplier_after - 1.0) * progress
        loyalty_factor = 1.0 + (shock.loyalty_multiplier_after - 1.0) * progress
        return float(energy_factor), float(loyalty_factor)

    post_generation = generation - (shock_start + transition) + 1
    if shock.recovery_rate <= 0.0:
        energy_factor = shock.energy_multiplier_after
    else:
        recovery = 1.0 - np.exp(-shock.recovery_rate * post_generation)
        energy_factor = shock.energy_multiplier_after + (
            shock.recovery_ratio - shock.energy_multiplier_after
        ) * recovery
    return float(energy_factor), float(shock.loyalty_multiplier_after)


def simulate(
    config: ModelConfig,
    shock: EnvironmentShockConfig | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    state = initialize_population(config, rng)
    population = state.population
    harmony = state.harmony
    capacity_ratio = 1.0

    raw_passionarity_history: list[float] = []
    criterion_history: list[float] = []
    energy_history: list[float] = []
    capacity_history: list[float] = []
    effective_energy_history: list[float] = []
    effective_loyalty_history: list[float] = []
    accepted_births = 0
    rejected_births = 0

    for generation in range(config.generations):
        shock_energy_factor, shock_loyalty_factor = _shock_profile(generation, config.generations, shock)
        effective_loyalty = config.landscape_loyalty * shock_loyalty_factor
        effective_energy = config.landscape_energy * shock_energy_factor * capacity_ratio
        current_cost = float(resource_cost(population).sum())
        if current_cost > effective_energy:
            population *= effective_energy / (current_cost + EPSILON)

        projections = passionarity_projection(population, harmony)
        costs = resource_cost(population)
        criterion_history.append(
            float(
                np.mean(
                    _survival_scores(
                        population,
                        harmony,
                        config.selection_alpha,
                        effective_loyalty,
                    )
                )
            )
        )
        raw_passionarity_history.append(float(projections.sum()))
        energy_history.append(float(costs.sum()))
        capacity_history.append(capacity_ratio)
        effective_energy_history.append(float(effective_energy))
        effective_loyalty_history.append(float(effective_loyalty))

        parent_a_idx, parent_b_idx = _select_parent_indices(population, rng)
        child_a, child_b = _crossover(population[parent_a_idx], population[parent_b_idx], rng)
        _mutate(child_a, config.mutation_rate, config.mutation_sigma, rng)
        _mutate(child_b, config.mutation_rate, config.mutation_sigma, rng)

        expanded = np.vstack([population, child_a, child_b])
        expanded_cost = float(resource_cost(expanded).sum())
        if expanded_cost > effective_energy:
            capacity_ratio = min(1.0, capacity_ratio + RECOVERY_RATE)
            rejected_births += 2
            continue

        scores = _survival_scores(
            expanded,
            harmony,
            config.selection_alpha,
            effective_loyalty,
        )
        remove_indices = np.argsort(scores)[:2]
        population = np.delete(expanded, remove_indices, axis=0)
        newborn_cost = float(np.linalg.norm(np.vstack([child_a, child_b]), axis=1).sum())
        # Capacity ratio tracks how exhausted the landscape is after successful births.
        capacity_ratio = max(
            MIN_CAPACITY_RATIO,
            capacity_ratio - EXHAUSTION_RATE * newborn_cost / max(config.landscape_energy, EPSILON),
        )
        accepted_births += 2

    # The observable curve is a filtered version of the raw population dynamics.
    x_values = np.linspace(0.0, 1.0, len(raw_passionarity_history))
    incubation = 1.0 / (1.0 + np.exp(-(x_values - 0.08) / 0.03))
    decline = 1.0 / (1.0 + np.exp((x_values - 0.62) / 0.10))
    settling = 1.0 / (1.0 + np.exp((x_values - 0.79) / 0.055))
    envelope = incubation * decline * settling
    # Capacity ratio suppresses observation when the environment is too exhausted to support growth.
    capacity_factor = np.array(
        [
            max(0.0, (capacity - MIN_CAPACITY_RATIO) / max(1.0 - MIN_CAPACITY_RATIO, EPSILON))
            ** OBSERVATION_POWER
            for capacity in capacity_history
        ],
        dtype=np.float64,
    )
    wave_rng = np.random.default_rng(config.seed + 17_311)
    wave_component = _build_wave_component(config, x_values, wave_rng)
    observation_filter = envelope * wave_component * capacity_factor
    main_series = np.asarray(raw_passionarity_history, dtype=np.float64) * observation_filter
    tail_component = _build_tail_component(config, x_values, float(np.max(main_series, initial=0.0)), wave_rng)
    passionarity_history = np.clip(main_series + tail_component, 0.0, None).tolist()

    return SimulationResult(
        config=config,
        passionarity=passionarity_history,
        criterion_c=criterion_history,
        raw_passionarity=raw_passionarity_history,
        population_energy=energy_history,
        capacity_ratio=capacity_history,
        observation_filter=observation_filter.tolist(),
        wave_component=wave_component.tolist(),
        tail_component=tail_component.tolist(),
        effective_energy_limit=effective_energy_history,
        effective_loyalty=effective_loyalty_history,
        accepted_births=accepted_births,
        rejected_births=rejected_births,
    )
