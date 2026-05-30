import numpy as np

NUM_ENVIRONMENTS = 10
SEED_START = 42


if __name__ == "__main__":
    print("=" * 60)
    print("  MULTI-SEED EXPERIMENT — DA221 Project")
    print("=" * 60)

    SEEDS = list(range(SEED_START, SEED_START + NUM_ENVIRONMENTS))


    all_runs = {
        'Basic QL':      [],
        'Improved QL':   [],
        'DQN':           []
    }

    for seed in SEEDS:
        # Update global seed
        from . import comparison as ca
        ca.SEED = seed
        print(f"\n{'='*60}")
        print(f"  Running seed = {seed}")
        print(f"{'='*60}")

        m1, r1, st1, su1, p1, e1 = ca.run_basic_qlearning()
        m2, r2, st2, su2, p2, e2 = ca.run_improved_qlearning()
        m3, r3, st3, su3, p3, e3 = ca.run_dqn()

        all_runs['Basic QL'].append(m1)
        all_runs['Improved QL'].append(m2)
        all_runs['DQN'].append(m3)

    # Average results across seeds
    print("\n")
    print("=" * 75)
    print(f"  AVERAGED RESULTS ACROSS {NUM_ENVIRONMENTS} ENVIRONMENTS")
    print("=" * 75)

    metrics_to_avg = [
        'success_rate',
        'path_length',
        'path_smoothness',
        'avg_steps',
        'avg_reward',
        'convergence_ep',
        'training_time'
    ]

    labels = {
        'success_rate':    'Success Rate (%)',
        'path_length':     'Path Length (steps)',
        'path_smoothness': 'Direction Changes',
        'avg_steps':       'Avg Steps (last 200)',
        'avg_reward':      'Avg Reward (last 200)',
        'convergence_ep':  'Convergence Episode',
        'training_time':   'Training Time (s)'
    }

    print(f"{'Metric':<25} {'Basic QL':>12} "
          f"{'Improved QL':>14} {'DQN':>12}")
    print("-" * 75)

    averaged = {}
    for key in metrics_to_avg:
        avgs = []
        stds = []
        for algo in ['Basic QL', 'Improved QL', 'DQN']:
            vals = [run[key] for run in all_runs[algo]]
            avgs.append(np.mean(vals))
            stds.append(np.std(vals))

        averaged[key] = avgs

        # Format with std deviation
        cols = [f"{avgs[i]:.1f}±{stds[i]:.1f}"
                for i in range(3)]
        print(f"{labels[key]:<25} {cols[0]:>12} "
              f"{cols[1]:>14} {cols[2]:>12}")

    print("=" * 75)