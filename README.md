# AI Robot Navigation — Reinforcement Learning on Grid Worlds

Comparison of **Basic Q-Learning**, **Improved Q-Learning** (8 actions + APF-style rewards), and **Deep Q-Network (DQN)** for autonomous navigation in obstacle grids. Built for coursework / portfolio use (DA221-style robot path planning).

## Features

- Tabular Q-learning (4 and 8 action spaces)
- DQN with PyTorch (CPU or CUDA)
- Side-by-side algorithm comparison and path visualization
- Small (10×10) and large (40×40) grid presets
- Multi-seed evaluation across many random environment layouts
- All training plots saved under `results/`

## Project structure

```
AI_Project/
├── run.py                      # CLI entry point (use this to run experiments)
├── requirements.txt
├── README.md
├── results/                    # Generated PNG plots
└── src/
    ├── algorithms/
    │   ├── q_learning_robot.py      # Basic Q-Learning
    │   ├── improved_q_learning.py   # Improved Q-Learning + APF
    │   └── dqn_robot.py             # DQN
    └── experiments/
        ├── comparison.py            # Compare all three on one grid
        ├── large_grid_experiment.py # Large-grid comparison
        └── multiple.py              # Average metrics over many seeds
```

Your original training logic lives unchanged under `src/`; `run.py` only sets hyperparameters and redirects `matplotlib` output to `results/`.

## Setup

```bash
cd AI_Project
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## How to run

Always run from the project root:

```bash
python run.py <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `basic` | Train basic Q-Learning (4 actions) |
| `improved` | Train improved Q-Learning (8 actions) |
| `dqn` | Train DQN |
| `compare` | Run all three algorithms on the same grid and plot comparison |
| `large` | Full comparison on a large grid (default 40×40) |
| `multi` | Run `compare` across multiple random seeds and print averaged metrics |

### Grid presets

| Preset | Grid | Episodes | Max steps / episode |
|--------|------|----------|---------------------|
| `small` | 10×10 | 2000 | 300 |
| `large` | 40×40 | 5000 | 800 |

```bash
# Small grid — single algorithms
python run.py basic --preset small
python run.py improved --preset small
python run.py dqn --preset small

# Compare all three (small grid)
python run.py compare --preset small

# Large grid experiment
python run.py large --preset large

# Custom grid size and training length
python run.py compare --grid-size 20 --episodes 3000 --max-steps 500

# Multiple environments (10 random layouts by default)
python run.py multi --num-envs 10 --preset small

# 5 environments on a 25×25 grid, custom seed start
python run.py multi --num-envs 5 --grid-size 25 --seed 100
```

### Other options

| Flag | Purpose |
|------|---------|
| `--grid-size N` | Override grid width/height |
| `--episodes N` | Training episodes |
| `--max-steps N` | Step limit per episode |
| `--seed N` | Random seed (single-run commands) or starting seed for `multi` |
| `--obstacle-ratio R` | Obstacle density (e.g. `0.2` = 20%) |
| `--num-envs K` | Number of environments for `multi` (seeds `K`, `K+1`, …) |

### Output files

Plots are written to `results/`:

| Script | Files |
|--------|--------|
| `basic` | `basic_path.png`, `basic_curves.png` |
| `improved` | `improved_path.png`, `improved_curves.png` |
| `dqn` | `dqn_path.png`, `dqn_curves.png` |
| `compare` | `comparison_paths.png`, `comparison_curves.png` |
| `large` | `large_grid_paths.png`, `large_grid_curves.png` |

`multi` prints a summary table only (no extra plots).

## Tech stack

- Python 3.10+
- NumPy, Matplotlib
- PyTorch (DQN and large-grid experiments)


