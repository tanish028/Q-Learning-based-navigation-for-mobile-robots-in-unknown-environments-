#!/usr/bin/env python3
"""
Entry point for the AI Robot Navigation project.
Configures grid size / training length and saves plots under results/.
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

PRESETS = {
    "small": {
        "grid_size": 10,
        "n_episodes": 2000,
        "max_steps": 300,
    },
    "large": {
        "grid_size": 40,
        "n_episodes": 5000,
        "max_steps": 800,
    },
}

COMMANDS = {
    "basic": "src.algorithms.q_learning_robot",
    "improved": "src.algorithms.improved_q_learning",
    "dqn": "src.algorithms.dqn_robot",
    "compare": "src.experiments.comparison",
    "large": "src.experiments.large_grid_experiment",
    "multi": "src.experiments.multiple",
}


def _ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _patch_savefig() -> None:
    """Send all matplotlib saves to results/ without editing algorithm files."""
    import matplotlib.pyplot as plt

    original = plt.savefig

    def save_to_results(fname, *args, **kwargs):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / Path(fname).name
        return original(str(out), *args, **kwargs)

    plt.savefig = save_to_results


def _resolve_preset(args: argparse.Namespace) -> dict:
    if args.preset:
        return dict(PRESETS[args.preset])
    return {}


def apply_config(module, args: argparse.Namespace) -> None:
    """Override module-level hyperparameters before __main__ runs."""
    cfg = _resolve_preset(args)

    grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size")
    n_episodes = args.episodes if args.episodes is not None else cfg.get("n_episodes")
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps")

    if grid_size is not None and hasattr(module, "GRID_SIZE"):
        module.GRID_SIZE = grid_size
    if n_episodes is not None and hasattr(module, "N_EPISODES"):
        module.N_EPISODES = n_episodes
    if max_steps is not None and hasattr(module, "MAX_STEPS"):
        module.MAX_STEPS = max_steps
    if args.seed is not None and hasattr(module, "SEED"):
        module.SEED = args.seed
    if args.obstacle_ratio is not None and hasattr(module, "OBSTACLE_RATIO"):
        module.OBSTACLE_RATIO = args.obstacle_ratio


def run_module(module_path: str, args: argparse.Namespace) -> None:
    _ensure_project_on_path()
    _patch_savefig()

    module = importlib.import_module(module_path)
    apply_config(module, args)

    if module_path == COMMANDS["multi"]:
        comparison = importlib.import_module(COMMANDS["compare"])
        apply_config(comparison, args)
        module.NUM_ENVIRONMENTS = args.num_envs
        if args.seed is not None:
            module.SEED_START = args.seed

    runpy.run_module(module_path, run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare RL agents on grid navigation environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py basic --preset small
  python run.py compare --grid-size 20
  python run.py large --preset large
  python run.py multi --num-envs 5 --preset small
  python run.py dqn --grid-size 40 --episodes 5000 --max-steps 800
        """.strip(),
    )

    parser.add_argument(
        "command",
        choices=sorted(COMMANDS.keys()),
        help="Which script to run",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="small (10x10) or large (40x40) with matching episode limits",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        metavar="N",
        help="Grid width/height (overrides preset)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Training episodes (overrides preset)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Max steps per episode (overrides preset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for a single environment layout",
    )
    parser.add_argument(
        "--obstacle-ratio",
        type=float,
        metavar="R",
        help="Fraction of cells that are obstacles (default in each script: 0.2)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=10,
        metavar="K",
        help="Number of environments for 'multi' (seeds K, K+1, ...)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "large" and args.preset is None and args.grid_size is None:
        args.preset = "large"
    if args.command in ("basic", "improved", "dqn", "compare") and args.preset is None:
        if args.grid_size is None and args.command != "compare":
            pass
        elif args.command == "compare" and args.grid_size is None:
            args.preset = "small"

    module_path = COMMANDS[args.command]
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    run_module(module_path, args)


if __name__ == "__main__":
    main()
