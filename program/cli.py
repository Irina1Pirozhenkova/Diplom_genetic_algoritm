from __future__ import annotations

import argparse
from pathlib import Path

from .batch import DEFAULT_SCENARIO_PATH, run_batch_scenario
from .model import ModelConfig
from .search import BASE_IMAGE_NAME, render_all_outputs, run_search, save_search_result


BEST_PARAMETERS_DIR = Path("best_parameters")
BEST_CONFIG_PATH = BEST_PARAMETERS_DIR / "best_config.json"
SCHEDULER_RUNS_DIR = Path("scheduler_runs")
SCENARIOS_DIR = Path("scenarios")


def _available_scenario_names() -> list[str]:
    """Return short scenario names that can be entered without a full path."""

    return sorted(path.stem for path in SCENARIOS_DIR.glob("*.json"))


def _resolve_scenario_name(raw_value: str) -> Path:
    """Resolve either a direct path or a short name like `scenario_1`."""

    candidate = Path(raw_value)
    if candidate.suffix == ".json" or candidate.is_absolute() or any(sep in raw_value for sep in ("/", "\\")):
        return candidate

    scenario_candidate = SCENARIOS_DIR / f"{raw_value}.json"
    if scenario_candidate.exists():
        return scenario_candidate

    available = ", ".join(_available_scenario_names())
    raise ValueError(f"Сценарий '{raw_value}' не найден. Доступные сценарии: {available}")


def _prompt_mode() -> str:
    """Ask which top-level execution mode should be used."""

    while True:
        print("Выберите режим запуска:")
        print("1. Лучшие параметры")
        print("2. Сценарий из конфигурационного файла")
        choice = input("Введите 1 или 2: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("Нужно ввести 1 или 2.")


def _prompt_scenario_path(default_name: str) -> Path:
    """Accept either a short scenario name or a full JSON path."""

    available = ", ".join(_available_scenario_names())
    raw_value = input(
        f"Укажите имя сценария или путь к JSON (Enter = {default_name}; доступно: {available}): "
    ).strip()
    return _resolve_scenario_name(raw_value) if raw_value else SCENARIOS_DIR / f"{default_name}.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the main command-line parser for all program modes."""

    parser = argparse.ArgumentParser(
        prog="program",
        description="Моделирование этногенеза на базе генетического алгоритма.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `run` is the main user entry point: it asks whether to use best parameters or a scenario file.
    run_parser = subparsers.add_parser(
        "run",
        help="Интерактивный запуск: лучшие параметры или пакетный сценарий.",
    )
    run_parser.add_argument(
        "--config",
        default=str(BEST_CONFIG_PATH),
        help="Путь к best_config.json для режима лучших параметров.",
    )
    run_parser.add_argument(
        "--scenario",
        default=None,
        help="Имя сценария из папки scenarios или путь к JSON-файлу.",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Число рабочих процессов для пакетного режима.",
    )
    run_parser.add_argument(
        "--runs-dir",
        default=str(SCHEDULER_RUNS_DIR),
        help="Корневая папка для результатов планировщика задач.",
    )

    # `search` is a non-interactive utility for finding a good baseline configuration.
    search_parser = subparsers.add_parser(
        "search",
        help="Подобрать параметры и сохранить лучший результат.",
    )
    search_parser.add_argument("--trials", type=int, default=120, help="Количество случайных наборов параметров.")
    search_parser.add_argument(
        "--seeds-per-trial",
        type=int,
        default=5,
        help="Сколько независимых сидов проверять для каждого набора.",
    )
    search_parser.add_argument("--search-seed", type=int, default=42, help="Сид генератора случайного поиска.")
    search_parser.add_argument(
        "--output-dir",
        default=str(BEST_PARAMETERS_DIR),
        help="Каталог для сохранения конфигурации и PNG-графиков.",
    )

    # `render` quickly rebuilds the full visualization package for one chosen configuration.
    render_parser = subparsers.add_parser(
        "render",
        help="Построить шесть PNG: база, E, K, два шока и график C(t).",
    )
    render_parser.add_argument(
        "--config",
        default=str(BEST_CONFIG_PATH),
        help="Путь к JSON-конфигурации модели.",
    )
    render_parser.add_argument(
        "--output",
        default=str(BEST_PARAMETERS_DIR / BASE_IMAGE_NAME),
        help="Путь к базовому PNG; остальные файлы будут сохранены рядом.",
    )

    return parser


def _run_best_mode(config_path: str | Path) -> int:
    """Rebuild the standard result package from the best saved configuration."""

    config = ModelConfig.load_json(config_path)
    output_paths = render_all_outputs(config, BEST_PARAMETERS_DIR / BASE_IMAGE_NAME)
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print(f"Конфигурация экспериментов: {BEST_PARAMETERS_DIR / 'experiment_scenarios.json'}")
    return 0


def _run_scenario_mode(
    scenario_path: str | Path | None,
    workers: int | None,
    runs_dir: str | Path,
) -> int:
    """Resolve the scenario and execute all runs through the batch scheduler."""

    try:
        if scenario_path:
            scenario = _resolve_scenario_name(str(scenario_path))
        else:
            scenario = _prompt_scenario_path(DEFAULT_SCENARIO_PATH.stem)
    except ValueError as exc:
        print(exc)
        return 2

    package_dir = run_batch_scenario(scenario, output_root=runs_dir, workers=workers)
    print(f"Пакет результатов сохранён в: {package_dir}")
    for run_dir in sorted(path for path in package_dir.iterdir() if path.is_dir()):
        print(f"Прогон: {run_dir}")
    print(f"Сценарий: {package_dir / 'scenario.json'}")
    print(f"Манифест: {package_dir / 'manifest.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected command and keep the CLI surface compact."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        # Explicit command-line scenario should work non-interactively, which is useful for demos and scripts.
        if args.scenario is not None:
            return _run_scenario_mode(args.scenario, args.workers, args.runs_dir)
        mode = _prompt_mode()
        if mode == "1":
            return _run_best_mode(args.config)
        return _run_scenario_mode(args.scenario, args.workers, args.runs_dir)

    if args.command == "search":
        result = run_search(
            trials=args.trials,
            seeds_per_trial=args.seeds_per_trial,
            search_seed=args.search_seed,
        )
        output_paths = save_search_result(result, args.output_dir)
        print(f"Лучший score: {result.score:.4f}")
        print(f"Лучшая конфигурация: {Path(args.output_dir) / 'best_config.json'}")
        for name, path in output_paths.items():
            print(f"{name}: {path}")
        print(f"Конфигурация экспериментов: {Path(args.output_dir) / 'experiment_scenarios.json'}")
        return 0

    if args.command == "render":
        config = ModelConfig.load_json(args.config)
        output_paths = render_all_outputs(config, args.output)
        for name, path in output_paths.items():
            print(f"{name}: {path}")
        print(f"Конфигурация экспериментов: {Path(args.output).parent / 'experiment_scenarios.json'}")
        return 0

    parser.error("Unknown command.")
    return 2
