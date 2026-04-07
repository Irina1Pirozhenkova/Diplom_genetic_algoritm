from __future__ import annotations

import numpy as np


EPSILON = 1e-9


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth the curve without changing its overall scale."""

    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _local_maxima(values: np.ndarray) -> np.ndarray:
    # Pure local maxima are enough here because the curve is already smoothed before peak selection.
    if values.size < 3:
        return np.array([], dtype=np.int64)
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1


def _count_shoulders(values: np.ndarray, peak_index: int) -> tuple[int, int]:
    """Estimate how many noticeable bends exist around the main hump."""

    gradient = np.diff(values)
    left_turns = np.where((gradient[:-1] > 0.002) & (gradient[1:] < -0.002))[0]
    right_turns = np.where((gradient[:-1] < -0.002) & (gradient[1:] > 0.002))[0]
    left_count = int(np.count_nonzero(left_turns < peak_index))
    right_count = int(np.count_nonzero(right_turns > peak_index))
    return left_count, right_count


def _closeness(value: float, target: float, width: float) -> float:
    # This helper turns "close to target" into a soft score in [0, 1].
    return max(0.0, 1.0 - abs(value - target) / max(width, EPSILON))


def _peak_prominence(values: np.ndarray, index: int, radius: int) -> float:
    # Prominence helps ignore tiny ripples that are too small to matter visually.
    left = values[max(0, index - radius): index + 1]
    right = values[index: min(values.size, index + radius + 1)]
    if left.size == 0 or right.size == 0:
        return 0.0
    baseline = max(float(left.min()), float(right.min()))
    return float(values[index] - baseline)


def _select_peaks(
    values: np.ndarray,
    min_height: float,
    min_distance: int,
    min_prominence: float,
) -> list[int]:
    """Keep only meaningful peaks, ignoring tiny numerical ripples."""

    candidates = [
        int(idx)
        for idx in _local_maxima(values)
        if values[idx] >= min_height and _peak_prominence(values, int(idx), max(3, min_distance // 2)) >= min_prominence
    ]
    chosen: list[int] = []
    for idx in sorted(candidates, key=lambda item: float(values[item]), reverse=True):
        if all(abs(idx - other) >= min_distance for other in chosen):
            chosen.append(idx)
    chosen.sort()
    return chosen


def score_curve(series: list[float] | np.ndarray) -> tuple[float, dict[str, float]]:
    # The scoring function prefers an early main hump, multiple moderate waves and a low tail.
    values = np.asarray(series, dtype=np.float64)
    if values.size < 20:
        return -1e9, {"reason": -1.0}

    minimum = float(values.min())
    shifted = values - minimum
    amplitude = float(shifted.max())
    if amplitude <= EPSILON:
        return -1e9, {"reason": -2.0}

    normalized = shifted / amplitude
    # Light smoothing keeps medium waves visible; heavy smoothing checks the single main hump.
    light = moving_average(normalized, max(3, values.size // 150))
    heavy = moving_average(normalized, max(9, values.size // 45))

    peak_index = int(np.argmax(heavy))
    peak_position = peak_index / max(values.size - 1, 1)
    start_value = float(heavy[0])
    end_value = float(heavy[-1])
    early_mean = float(heavy[: max(10, values.size // 5)].mean())
    late_mean = float(heavy[-max(10, values.size // 5) :].mean())

    left_count, right_count = _count_shoulders(light, peak_index)
    shoulders_total = left_count + right_count

    heavy_peaks = _select_peaks(
        heavy,
        min_height=0.45,
        min_distance=max(25, values.size // 12),
        min_prominence=0.06,
    )
    dominant_heavy_peaks = [idx for idx in heavy_peaks if heavy[idx] >= 0.58]
    late_dominant_peaks = [idx for idx in dominant_heavy_peaks if idx > peak_index + values.size * 0.08]

    # Count medium peaks over most of the curve and small oscillations near the tail separately.
    wave_peaks = _select_peaks(
        light,
        min_height=0.12,
        min_distance=max(10, values.size // 28),
        min_prominence=0.015,
    )
    tail_start = int(values.size * 0.80)
    mid_wave_peaks = [idx for idx in wave_peaks if idx < tail_start]
    tail_peaks = [
        idx
        for idx in _select_peaks(
            light,
            min_height=0.005,
            min_distance=max(6, values.size // 45),
            min_prominence=0.004,
        )
        if idx >= tail_start and light[idx] <= 0.22
    ]

    rise = float(heavy[peak_index] - heavy[0])
    drop = float(heavy[peak_index] - heavy[-1])
    peak_score = _closeness(peak_position, 0.34, 0.16)
    start_score = 1.0 if start_value <= 0.15 else max(0.0, 1.0 - (start_value - 0.15) * 3.0)
    end_score = 1.0 if end_value <= 0.15 else max(0.0, 1.0 - (end_value - 0.15) * 3.0)
    rise_score = np.clip(rise / 0.75, 0.0, 1.0)
    drop_score = np.clip(drop / 0.75, 0.0, 1.0)
    shoulder_score = _closeness(float(shoulders_total), 4.0, 2.0)
    late_tail_penalty = max(0.0, late_mean - 0.20)
    second_peak_penalty = min(
        1.0,
        len(late_dominant_peaks) * 0.40 + max(0, len(dominant_heavy_peaks) - 2) * 0.20,
    )
    plateau_score = float(np.clip(np.mean(heavy[max(0, peak_index - 20): min(values.size, peak_index + 20)]), 0.0, 1.0))
    early_growth_score = np.clip((float(heavy[max(10, values.size // 8)]) - start_value) / 0.35, 0.0, 1.0)
    wave_count_score = _closeness(float(len(mid_wave_peaks)), 11.0, 5.0)
    tail_wave_score = _closeness(float(len(tail_peaks)), 2.0, 1.5)
    smooth_curve_penalty = max(0.0, (3.0 - len(mid_wave_peaks)) * 0.35)
    missing_tail_penalty = 0.55 if len(tail_peaks) == 0 else 0.0
    late_parasite_penalty = 0.0
    for idx in wave_peaks:
        if idx >= tail_start and light[idx] > 0.22:
            late_parasite_penalty += 0.25
    late_parasite_penalty = min(1.2, late_parasite_penalty)

    # Reward the Gumilev-like silhouette: early rise, single dominant hump, many moderate waves,
    # and a low oscillating tail close to zero.
    score = (
        2.0 * peak_score
        + 1.6 * start_score
        + 1.6 * end_score
        + 1.2 * rise_score
        + 1.2 * drop_score
        + 0.9 * shoulder_score
        + 0.8 * plateau_score
        + 0.8 * early_growth_score
        + 1.5 * wave_count_score
        + 1.2 * tail_wave_score
        - 1.8 * late_tail_penalty
        - 1.4 * second_peak_penalty
        - 1.4 * smooth_curve_penalty
        - 1.0 * missing_tail_penalty
        - 1.2 * late_parasite_penalty
        - 0.9 * abs(early_mean - 0.45)
    )
    metrics = {
        "peak_position": peak_position,
        "start_value": start_value,
        "end_value": end_value,
        "rise": rise,
        "drop": drop,
        "shoulders_total": float(shoulders_total),
        "late_tail_penalty": late_tail_penalty,
        "second_peak_penalty": second_peak_penalty,
        "wave_peak_count": float(len(mid_wave_peaks)),
        "tail_peak_count": float(len(tail_peaks)),
        "smooth_curve_penalty": smooth_curve_penalty,
        "missing_tail_penalty": missing_tail_penalty,
        "late_parasite_penalty": late_parasite_penalty,
    }
    return float(score), metrics
