from __future__ import annotations

import argparse
from pathlib import Path

from .model import ModelConfig
from .search import BASE_IMAGE_NAME, render_all_outputs, run_search, save_search_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ethnogenesis",
        description="\u041c\u043e\u0434\u0435\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u044d\u0442\u043d\u043e\u0433\u0435\u043d\u0435\u0437\u0430 \u043d\u0430 \u0431\u0430\u0437\u0435 \u0433\u0435\u043d\u0435\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c\u0430.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        help="\u041f\u043e\u0434\u043e\u0431\u0440\u0430\u0442\u044c \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0438 \u0441\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043b\u0443\u0447\u0448\u0438\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442.",
    )
    search_parser.add_argument("--trials", type=int, default=120, help="\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u0445 \u043d\u0430\u0431\u043e\u0440\u043e\u0432 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432.")
    search_parser.add_argument(
        "--seeds-per-trial",
        type=int,
        default=5,
        help="\u0421\u043a\u043e\u043b\u044c\u043a\u043e \u043d\u0435\u0437\u0430\u0432\u0438\u0441\u0438\u043c\u044b\u0445 \u0441\u0438\u0434\u043e\u0432 \u043f\u0440\u043e\u0432\u0435\u0440\u044f\u0442\u044c \u0434\u043b\u044f \u043a\u0430\u0436\u0434\u043e\u0433\u043e \u043d\u0430\u0431\u043e\u0440\u0430.",
    )
    search_parser.add_argument("--search-seed", type=int, default=42, help="\u0421\u0438\u0434 \u0433\u0435\u043d\u0435\u0440\u0430\u0442\u043e\u0440\u0430 \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u043e\u0433\u043e \u043f\u043e\u0438\u0441\u043a\u0430.")
    search_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="\u041a\u0430\u0442\u0430\u043b\u043e\u0433 \u0434\u043b\u044f \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u044f \u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u0438 \u0438 PNG-\u0433\u0440\u0430\u0444\u0438\u043a\u043e\u0432.",
    )

    render_parser = subparsers.add_parser(
        "render",
        help="\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0448\u0435\u0441\u0442\u044c PNG: \u0431\u0430\u0437\u0430, E, K, \u0434\u0432\u0430 \u0448\u043e\u043a\u0430 \u0438 \u0433\u0440\u0430\u0444\u0438\u043a C(t).",
    )
    render_parser.add_argument(
        "--config",
        default="artifacts/best_config.json",
        help="\u041f\u0443\u0442\u044c \u043a JSON-\u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u0438 \u043c\u043e\u0434\u0435\u043b\u0438.",
    )
    render_parser.add_argument(
        "--output",
        default=f"artifacts/{BASE_IMAGE_NAME}",
        help="\u041a\u0430\u0442\u0430\u043b\u043e\u0433 \u0434\u043b\u044f \u0441\u0435\u0440\u0438\u0438 PNG; \u0438\u043c\u0435\u043d\u0430 \u0444\u0430\u0439\u043b\u043e\u0432 \u0437\u0430\u0434\u0430\u044e\u0442\u0441\u044f \u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        result = run_search(
            trials=args.trials,
            seeds_per_trial=args.seeds_per_trial,
            search_seed=args.search_seed,
        )
        output_paths = save_search_result(result, args.output_dir)
        print(f"Best score: {result.score:.4f}")
        print(f"Config saved to: {Path(args.output_dir) / 'best_config.json'}")
        for name, path in output_paths.items():
            print(f"{name} PNG saved to: {path}")
        print(f"Experiment config saved to: {Path(args.output_dir) / 'experiment_scenarios.json'}")
        return 0

    if args.command == "render":
        config = ModelConfig.load_json(args.config)
        output_paths = render_all_outputs(config, args.output)
        for name, path in output_paths.items():
            print(f"{name} PNG saved to: {path}")
        print(f"Experiment config saved to: {Path(args.output).parent / 'experiment_scenarios.json'}")
        return 0

    parser.error("Unknown command.")
    return 2
