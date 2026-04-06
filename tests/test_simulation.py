from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ethnogenesis.model import ModelConfig
from ethnogenesis.render import render_curve
from ethnogenesis.scoring import score_curve
from ethnogenesis.search import FIXED_VOCATION_DIM, SHOCK_LABEL, render_all_outputs
from ethnogenesis.simulator import default_environment_shocks, initialize_population, simulate


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            population_size=120,
            vocation_dim=4,
            mutation_rate=0.08,
            mutation_sigma=0.35,
            landscape_energy=170.0,
            landscape_loyalty=0.55,
            selection_alpha=0.95,
            generations=120,
            seed=12345,
            wave_strength=0.10,
            wave_count=12,
            wave_irregularity=0.12,
            tail_wave_strength=0.03,
            tail_wave_count=3,
            tail_decay_power=2.0,
        )
        self.energy_only, self.energy_early = default_environment_shocks()

    def test_simulation_is_reproducible(self) -> None:
        first = simulate(self.config)
        second = simulate(self.config)
        self.assertEqual(first.passionarity, second.passionarity)
        self.assertEqual(first.criterion_c, second.criterion_c)
        self.assertEqual(first.wave_component, second.wave_component)

    def test_population_invariants_hold(self) -> None:
        result = simulate(self.config)
        values = np.asarray(result.passionarity, dtype=np.float64)
        criterion = np.asarray(result.criterion_c, dtype=np.float64)
        self.assertFalse(np.isnan(values).any())
        self.assertFalse(np.isinf(values).any())
        self.assertTrue((values >= 0.0).all())
        self.assertTrue((np.asarray(result.tail_component, dtype=np.float64) >= 0.0).all())
        self.assertFalse(np.isnan(criterion).any())
        self.assertFalse(np.isinf(criterion).any())

    def test_initial_population_respects_energy_limit(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        state = initialize_population(self.config, rng)
        total_cost = np.linalg.norm(state.population, axis=1).sum()
        self.assertLessEqual(total_cost, self.config.landscape_energy + 1e-6)

    def test_scoring_prefers_wavy_hump_curve(self) -> None:
        x = np.linspace(0.0, 1.0, 1000)
        envelope = np.exp(-((x - 0.38) ** 2) / 0.045)
        window = np.exp(-((x - 0.48) ** 2) / 0.11)
        wavy = envelope * (1.0 + 0.17 * np.sin(11 * np.pi * x) * window)
        smooth = envelope
        parasite = (
            envelope
            + 0.34 * np.exp(-((x - 0.84) ** 2) / 0.006)
            + 0.02 * np.sin(15 * np.pi * x) * np.exp(-((x - 0.90) ** 2) / 0.02)
        )
        wavy_score, _ = score_curve(wavy)
        smooth_score, _ = score_curve(smooth)
        parasite_score, _ = score_curve(parasite)
        self.assertGreater(wavy_score, smooth_score)
        self.assertGreater(wavy_score, parasite_score)

    def test_render_creates_png(self) -> None:
        result = simulate(self.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curve.png"
            render_curve(result.passionarity, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_render_uses_readable_russian_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curve.png"
            with mock.patch("matplotlib.axes._axes.Axes.set_xlabel") as set_xlabel:
                with mock.patch("matplotlib.axes._axes.Axes.set_ylabel") as set_ylabel:
                    render_curve([0.0, 1.0, 0.5], path, title="\u0422\u0435\u0441\u0442")
            self.assertEqual(set_xlabel.call_args.args[0], "\u041f\u043e\u043a\u043e\u043b\u0435\u043d\u0438\u0435")
            self.assertEqual(
                set_ylabel.call_args.args[0],
                "\u0421\u0443\u043c\u043c\u0430\u0440\u043d\u0430\u044f \u043f\u0430\u0441\u0441\u0438\u043e\u043d\u0430\u0440\u043d\u043e\u0441\u0442\u044c",
            )

    def test_render_accepts_shock_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curve.png"
            with mock.patch("matplotlib.axes._axes.Axes.scatter") as scatter:
                with mock.patch("matplotlib.axes._axes.Axes.annotate") as annotate:
                    with mock.patch("matplotlib.axes._axes.Axes.set_xticklabels") as set_xticklabels:
                        render_curve(
                            [0.0, 1.0, 0.5],
                            path,
                            marker_x=1,
                            marker_y=1.0,
                            marker_label=SHOCK_LABEL,
                            shock_axis_label="1",
                            y_min=0.0,
                        )
            self.assertTrue(scatter.called)
            self.assertEqual(annotate.call_args.args[0], SHOCK_LABEL)
            self.assertIn("1", set_xticklabels.call_args.args[0])

    def test_render_accepts_parameter_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curve.png"
            render_curve(
                [
                    ("\u0411\u0430\u0437\u0430", [0.0, 1.0, 0.5], "black"),
                    ("E x3", [0.0, 1.2, 0.6], "#8b1e1e"),
                ],
                path,
                legend_lines=[
                    ("\u0411\u0430\u0437\u0430", "black"),
                    ("E x3", "#8b1e1e"),
                    ("I = 4 (...)", "black"),
                ],
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_regression_shape_has_waves_and_low_tail(self) -> None:
        baseline = ModelConfig(
            population_size=132,
            vocation_dim=4,
            mutation_rate=0.013192384320954697,
            mutation_sigma=0.07600102861568155,
            landscape_energy=119.11386955367138,
            landscape_loyalty=0.9106871023717288,
            selection_alpha=1.3481025145231587,
            generations=1267,
            seed=393837,
            wave_strength=0.16992020172814357,
            wave_count=18,
            wave_irregularity=0.296981951471033,
            tail_wave_strength=0.0637801520857464,
            tail_wave_count=4,
            tail_decay_power=2.2359080068284802,
        )
        result = simulate(baseline)
        values = np.asarray(result.passionarity, dtype=np.float64)
        normalized = (values - values.min()) / max(values.max() - values.min(), 1e-9)
        peak_index = int(np.argmax(normalized))
        _, metrics = score_curve(values)
        self.assertLess(peak_index, int(values.size * 0.5))
        self.assertLess(normalized[0], 0.2)
        self.assertLess(normalized[-1], 0.2)
        self.assertGreaterEqual(metrics["wave_peak_count"], 3.0)
        self.assertGreaterEqual(metrics["tail_peak_count"], 1.0)

    def test_base_simulation_is_unchanged_without_shock(self) -> None:
        base = simulate(self.config)
        again = simulate(self.config, shock=None)
        self.assertEqual(base.passionarity, again.passionarity)
        self.assertEqual(base.criterion_c, again.criterion_c)

    def test_energy_shock_only_changes_energy_not_loyalty(self) -> None:
        result = simulate(self.config, shock=self.energy_only)
        start_idx = int(self.energy_only.start_ratio * self.config.generations)
        self.assertLess(
            min(result.effective_energy_limit[start_idx:]),
            self.config.landscape_energy * 0.5,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(result.effective_loyalty[start_idx:], dtype=np.float64),
                self.config.landscape_loyalty,
            )
        )

    def test_default_shocks_differ_only_by_start_ratio(self) -> None:
        self.assertEqual(self.energy_only.energy_multiplier_after, self.energy_early.energy_multiplier_after)
        self.assertEqual(self.energy_only.loyalty_multiplier_after, self.energy_early.loyalty_multiplier_after)
        self.assertEqual(self.energy_only.recovery_ratio, self.energy_early.recovery_ratio)
        self.assertEqual(self.energy_only.recovery_rate, self.energy_early.recovery_rate)
        self.assertNotEqual(self.energy_only.start_ratio, self.energy_early.start_ratio)

    def test_shocked_curve_drops_faster_than_base_after_shock(self) -> None:
        comparison_config = ModelConfig(
            population_size=119,
            vocation_dim=4,
            mutation_rate=0.13827743349708288,
            mutation_sigma=0.13311979782862662,
            landscape_energy=157.24758354978286,
            landscape_loyalty=0.4007030891519425,
            selection_alpha=0.4966004364839771,
            generations=1849,
            seed=601362,
            wave_strength=0.08052015905112908,
            wave_count=14,
            wave_irregularity=0.24901511190646775,
            tail_wave_strength=0.07353234239733376,
            tail_wave_count=3,
            tail_decay_power=1.8256493961946643,
        )
        base = simulate(comparison_config)
        energy = simulate(comparison_config, shock=self.energy_only)
        start_idx = int(self.energy_only.start_ratio * comparison_config.generations)
        base_tail_mean = float(np.mean(base.passionarity[start_idx + 15 : start_idx + 35]))
        energy_tail_mean = float(np.mean(energy.passionarity[start_idx + 15 : start_idx + 35]))
        self.assertLess(energy_tail_mean, base_tail_mean)

    def test_early_shock_hits_curve_earlier_than_late_shock(self) -> None:
        comparison_config = ModelConfig(
            population_size=119,
            vocation_dim=4,
            mutation_rate=0.13827743349708288,
            mutation_sigma=0.13311979782862662,
            landscape_energy=157.24758354978286,
            landscape_loyalty=0.4007030891519425,
            selection_alpha=0.4966004364839771,
            generations=1849,
            seed=601362,
            wave_strength=0.08052015905112908,
            wave_count=14,
            wave_irregularity=0.24901511190646775,
            tail_wave_strength=0.07353234239733376,
            tail_wave_count=3,
            tail_decay_power=1.8256493961946643,
        )
        late = simulate(comparison_config, shock=self.energy_only)
        early = simulate(comparison_config, shock=self.energy_early)
        late_idx = int(round(self.energy_only.start_ratio * comparison_config.generations))
        early_idx = int(round(self.energy_early.start_ratio * comparison_config.generations))
        self.assertEqual(late_idx, 1072)
        self.assertEqual(early_idx, 647)
        window = slice(early_idx + 15, early_idx + 35)
        self.assertLess(float(np.mean(early.passionarity[window])), float(np.mean(late.passionarity[window])))

    def test_render_all_outputs_creates_six_pngs_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_output = Path(tmpdir) / "passionarity_base.png"
            output_paths = render_all_outputs(self.config, base_output)
            self.assertTrue(output_paths["base_passionarity"].exists())
            self.assertTrue(output_paths["energy_comparison"].exists())
            self.assertTrue(output_paths["loyalty_comparison"].exists())
            self.assertTrue(output_paths["shock_1072"].exists())
            self.assertTrue(output_paths["shock_647"].exists())
            self.assertTrue(output_paths["criterion_c_base"].exists())

            scenario_path = Path(tmpdir) / "experiment_scenarios.json"
            self.assertTrue(scenario_path.exists())
            payload = json.loads(scenario_path.read_text(encoding="utf-8"))
            self.assertEqual(
                list(payload.keys()),
                ["base_passionarity", "energy_comparison", "loyalty_comparison", "criterion_c_base", "shock_1072", "shock_647"],
            )
            self.assertEqual(payload["base_passionarity"]["config"]["vocation_dim"], FIXED_VOCATION_DIM)
            self.assertEqual(payload["energy_comparison"]["base"]["vocation_dim"], FIXED_VOCATION_DIM)
            self.assertEqual(payload["loyalty_comparison"]["base"]["vocation_dim"], FIXED_VOCATION_DIM)
            self.assertGreater(
                payload["energy_comparison"]["up"]["landscape_energy"],
                payload["energy_comparison"]["base"]["landscape_energy"],
            )
            self.assertLess(
                payload["energy_comparison"]["down"]["landscape_energy"],
                payload["energy_comparison"]["base"]["landscape_energy"],
            )
            self.assertGreater(
                payload["loyalty_comparison"]["up"]["landscape_loyalty"],
                payload["loyalty_comparison"]["base"]["landscape_loyalty"],
            )
            self.assertLess(
                payload["loyalty_comparison"]["down"]["landscape_loyalty"],
                payload["loyalty_comparison"]["base"]["landscape_loyalty"],
            )

    def test_render_all_outputs_uses_expected_shock_generations_for_best_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_config = ModelConfig(
                population_size=119,
                vocation_dim=4,
                mutation_rate=0.13827743349708288,
                mutation_sigma=0.13311979782862662,
                landscape_energy=157.24758354978286,
                landscape_loyalty=0.4007030891519425,
                selection_alpha=0.4966004364839771,
                generations=1849,
                seed=601362,
                wave_strength=0.08052015905112908,
                wave_count=14,
                wave_irregularity=0.24901511190646775,
                tail_wave_strength=0.07353234239733376,
                tail_wave_count=3,
                tail_decay_power=1.8256493961946643,
            )
            render_all_outputs(comparison_config, Path(tmpdir) / "passionarity_base.png")
            scenario_path = Path(tmpdir) / "experiment_scenarios.json"
            payload = json.loads(scenario_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["shock_1072"]["generation"], 1072)
            self.assertEqual(payload["shock_647"]["generation"], 647)


if __name__ == "__main__":
    unittest.main()
