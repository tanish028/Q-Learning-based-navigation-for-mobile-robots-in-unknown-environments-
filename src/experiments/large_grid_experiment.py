import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


GRID_SIZE       = 40       # larger grid
OBSTACLE_RATIO  = 0.2
SEED            = 42

GAMMA           = 0.9
EPSILON         = 1.0
EPSILON_MIN     = 0.01

REWARD_GOAL     = 100
REWARD_OBSTACLE = -5
REWARD_SCALE    = 2.0
ZETA            = 0.5

N_EPISODES      = 5000     # more episodes for larger grid
MAX_STEPS       = 800      # more steps allowed per episode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS_4 = [(-1,0),(1,0),(0,-1),(0,1)]
ACTIONS_8 = [
    (-1,0),(1,0),(0,-1),(0,1),
    (-1,1),(-1,-1),(1,1),(1,-1)
]


class GridEnvironment:

    def __init__(self):
        self.size  = GRID_SIZE
        self.start = (0, 0)
        self.goal  = (GRID_SIZE-1, GRID_SIZE-1)
        self._build_grid()

    def _build_grid(self):
        np.random.seed(SEED)
        random.seed(SEED)
        self.grid = np.zeros((self.size, self.size), dtype=int)

        all_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r,c) != self.start and (r,c) != self.goal
        ]
        n_obstacles    = int(self.size * self.size * OBSTACLE_RATIO)
        obstacle_cells = random.sample(all_cells, n_obstacles)
        for (r,c) in obstacle_cells:
            self.grid[r][c] = 1

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def manhattan(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def euclidean(self, pos1, pos2):
        return np.sqrt(
            (pos1[0]-pos2[0])**2 +
            (pos1[1]-pos2[1])**2
        )

    def gravity_bonus(self, pos):
        dist = self.euclidean(pos, self.goal)
        if dist == 0:
            return 0
        return ZETA / dist

    def step(self, action, actions_list, use_apf=False):
        r, c     = self.agent_pos
        dr, dc   = actions_list[action]
        nr, nc   = r+dr, c+dc
        old_dist = self.manhattan(self.agent_pos, self.goal)

        if nr<0 or nr>=self.size or nc<0 or nc>=self.size:
            return self.agent_pos, REWARD_OBSTACLE, False

        elif self.grid[nr][nc] == 1:
            return self.agent_pos, REWARD_OBSTACLE, False

        elif (nr,nc) == self.goal:
            self.agent_pos = (nr,nc)
            return (nr,nc), REWARD_GOAL, True

        else:
            next_pos        = (nr,nc)
            new_dist        = self.manhattan(next_pos, self.goal)
            distance_reward = REWARD_SCALE * (old_dist - new_dist)
            gravity         = self.gravity_bonus(next_pos) \
                              if use_apf else 0
            reward          = distance_reward + gravity
            self.agent_pos  = next_pos
            return next_pos, reward, False



def convergence_episode(success_history, threshold=0.9, window=50):
    for i in range(window, len(success_history)):
        if np.mean(success_history[i-window:i]) >= threshold:
            return i
    return len(success_history)   # never converged




def run_basic_qlearning():
    print("\n" + "="*55)
    print(f"  Basic Q-Learning on {GRID_SIZE}×{GRID_SIZE} grid")
    print(f"  4 actions | No APF | Fixed params")
    print("="*55)

    env     = GridEnvironment()
    alpha   = 0.1
    eps     = EPSILON
    q_table = defaultdict(lambda: np.zeros(4))

    rewards_hist = []
    steps_hist   = []
    success_hist = []
    start_time   = time.time()

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done = env.step(
                action, ACTIONS_4, use_apf=False
            )

            curr_q  = q_table[state][action]
            target  = reward if done else \
                      reward + GAMMA * np.max(q_table[next_state])
            q_table[state][action] += alpha * (target - curr_q)

            state        = next_state
            total_reward += reward
            steps        += 1

        eps = max(EPSILON_MIN, eps * 0.998)

        reached = done and state == env.goal
        rewards_hist.append(total_reward)
        steps_hist.append(steps)
        success_hist.append(1 if reached else 0)

        if (episode+1) % 500 == 0:
            sr = np.mean(success_hist[-200:]) * 100
            print(f"  Ep {episode+1:4d} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"Q-table: {len(q_table):5d} states | "
                  f"Success: {sr:.1f}%")

    training_time = time.time() - start_time
    final_sr      = np.mean(success_hist[-200:]) * 100
    conv_ep       = convergence_episode(success_hist)
    coverage      = len(q_table) / (GRID_SIZE*GRID_SIZE) * 100

    print(f"\n  Final Success Rate:  {final_sr:.1f}%")
    print(f"  Q-table entries:     {len(q_table)} / "
          f"{GRID_SIZE*GRID_SIZE} states ({coverage:.1f}% coverage)")
    print(f"  Convergence:         Episode {conv_ep}")
    print(f"  Training time:       {training_time:.1f}s")

    return {
        'name':           'Basic Q-Learning',
        'success_rate':   final_sr,
        'q_table_size':   len(q_table),
        'coverage':       coverage,
        'convergence_ep': conv_ep,
        'training_time':  round(training_time, 1),
        'rewards':        rewards_hist,
        'steps':          steps_hist,
        'success':        success_hist,
        'q_table':        q_table,
        'actions':        ACTIONS_4,
        'env':            env
    }



def run_improved_qlearning():
    print("\n" + "="*55)
    print(f"  Improved Q-Learning on {GRID_SIZE}×{GRID_SIZE} grid")
    print(f"  8 actions | APF gravity | Dynamic params")
    print("="*55)

    env     = GridEnvironment()
    alpha   = 0.5
    eps     = EPSILON
    q_table = defaultdict(lambda: np.zeros(8))

    rewards_hist = []
    steps_hist   = []
    success_hist = []
    start_time   = time.time()

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:
            if random.random() < eps:
                action = random.randint(0, 7)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done = env.step(
                action, ACTIONS_8, use_apf=True
            )

            curr_q  = q_table[state][action]
            target  = reward if done else \
                      reward + GAMMA * np.max(q_table[next_state])
            q_table[state][action] += alpha * (target - curr_q)

            state        = next_state
            total_reward += reward
            steps        += 1

        eps   = max(EPSILON_MIN, eps * 0.998)
        alpha = max(0.01, alpha * 0.999)

        reached = done and state == env.goal
        rewards_hist.append(total_reward)
        steps_hist.append(steps)
        success_hist.append(1 if reached else 0)

        if (episode+1) % 500 == 0:
            sr = np.mean(success_hist[-200:]) * 100
            print(f"  Ep {episode+1:4d} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"Q-table: {len(q_table):5d} states | "
                  f"Success: {sr:.1f}%")

    training_time = time.time() - start_time
    final_sr      = np.mean(success_hist[-200:]) * 100
    conv_ep       = convergence_episode(success_hist)
    coverage      = len(q_table) / (GRID_SIZE*GRID_SIZE) * 100

    print(f"\n  Final Success Rate:  {final_sr:.1f}%")
    print(f"  Q-table entries:     {len(q_table)} / "
          f"{GRID_SIZE*GRID_SIZE} states ({coverage:.1f}% coverage)")
    print(f"  Convergence:         Episode {conv_ep}")
    print(f"  Training time:       {training_time:.1f}s")

    return {
        'name':           'Improved Q-Learning',
        'success_rate':   final_sr,
        'q_table_size':   len(q_table),
        'coverage':       coverage,
        'convergence_ep': conv_ep,
        'training_time':  round(training_time, 1),
        'rewards':        rewards_hist,
        'steps':          steps_hist,
        'success':        success_hist,
        'q_table':        q_table,
        'actions':        ACTIONS_8,
        'env':            env
    }




class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_dqn():
    print("\n" + "="*55)
    print(f"  DQN on {GRID_SIZE}×{GRID_SIZE} grid")
    print(f"  8 actions | APF | Neural Network")
    print("="*55)

    env        = GridEnvironment()
    main_net   = DQNetwork().to(DEVICE)
    target_net = DQNetwork().to(DEVICE)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()
    optimizer  = optim.Adam(main_net.parameters(), lr=0.001)
    memory     = deque(maxlen=20000)

    eps        = EPSILON
    steps_done = 0

    rewards_hist = []
    steps_hist   = []
    success_hist = []
    start_time   = time.time()

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:
            if random.random() < eps:
                action = random.randint(0, 7)
            else:
                with torch.no_grad():
                    st     = torch.FloatTensor(state)\
                                   .unsqueeze(0).to(DEVICE)
                    action = int(main_net(st).argmax().item())

            next_state, reward, done = env.step(
                action, ACTIONS_8, use_apf=True
            )
            memory.append((state, action, reward, next_state, done))
            state        = next_state
            total_reward += reward
            steps        += 1
            steps_done   += 1

            # Train on batch
            if len(memory) >= 128:
                batch      = random.sample(memory, 128)
                s,a,r,ns,d = zip(*batch)

                s_t  = torch.FloatTensor(s).to(DEVICE)
                a_t  = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
                r_t  = torch.FloatTensor(r).to(DEVICE)
                ns_t = torch.FloatTensor(ns).to(DEVICE)
                d_t  = torch.FloatTensor(d).to(DEVICE)

                curr_q = main_net(s_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_t).max(1).values
                    target = r_t + GAMMA * next_q * (1 - d_t)

                loss = F.mse_loss(curr_q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % 200 == 0:
                target_net.load_state_dict(main_net.state_dict())

        eps = max(EPSILON_MIN, eps * 0.998)

        reached = done and state == env.goal
        rewards_hist.append(total_reward)
        steps_hist.append(steps)
        success_hist.append(1 if reached else 0)

        if (episode+1) % 500 == 0:
            sr = np.mean(success_hist[-200:]) * 100
            print(f"  Ep {episode+1:4d} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"Memory: {len(memory):5d} | "
                  f"Success: {sr:.1f}%")

    training_time = time.time() - start_time
    final_sr      = np.mean(success_hist[-200:]) * 100
    conv_ep       = convergence_episode(success_hist)

    print(f"\n  Final Success Rate:  {final_sr:.1f}%")
    print(f"  Convergence:         Episode {conv_ep}")
    print(f"  Training time:       {training_time:.1f}s")

    return {
        'name':           'DQN',
        'success_rate':   final_sr,
        'coverage':       'N/A',
        'convergence_ep': conv_ep,
        'training_time':  round(training_time, 1),
        'rewards':        rewards_hist,
        'steps':          steps_hist,
        'success':        success_hist,
        'main_net':       main_net,
        'actions':        ACTIONS_8,
        'env':            env
    }




def get_path(result):
    env     = result['env']
    state   = env.reset()
    path    = [state]
    visited = set()
    done    = False
    steps   = 0

    while not done and steps < MAX_STEPS:
        if state in visited:
            break
        visited.add(state)

        if 'q_table' in result:
            action = int(np.argmax(result['q_table'][state]))
            next_state, _, done = env.step(
                action, result['actions'],
                use_apf=('Improved' in result['name'])
            )
        else:
            with torch.no_grad():
                st     = torch.FloatTensor(state)\
                               .unsqueeze(0).to(DEVICE)
                action = int(result['main_net'](st).argmax().item())
            next_state, _, done = env.step(
                action, result['actions'], use_apf=True
            )

        path.append(next_state)
        state = next_state
        steps += 1

    return path




def plot_paths(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors    = ['#0984e3', '#00b894', '#6c5ce7']

    for ax, result, color in zip(axes, results, colors):
        env  = result['env']
        size = env.size

        for r in range(size):
            for c in range(size):
                if env.grid[r][c] == 1:
                    fc = '#2d3436'
                elif (r,c) == env.start:
                    fc = '#00b894'
                elif (r,c) == env.goal:
                    fc = '#d63031'
                else:
                    fc = '#dfe6e9'
                rect = patches.Rectangle(
                    (c, size-1-r), 1, 1,
                    linewidth=0.2,
                    edgecolor='grey',
                    facecolor=fc
                )
                ax.add_patch(rect)

        path = get_path(result)
        if len(path) > 1:
            px = [c+0.5 for (r,c) in path]
            py = [size-0.5-r for (r,c) in path]
            ax.plot(px, py, color=color,
                    linewidth=1.5, marker='o',
                    markersize=2,
                    label=f'{len(path)-1} steps')

        sr = result['success_rate']
        ax.set_title(
            f"{result['name']}\nSuccess: {sr:.1f}%",
            fontweight='bold', fontsize=10
        )
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)

    plt.suptitle(
        f'Path Comparison — {GRID_SIZE}×{GRID_SIZE} Grid '
        f'(20% obstacles)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('large_grid_paths.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Saved: large_grid_paths.png")


def plot_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors    = ['#0984e3', '#00b894', '#6c5ce7']
    window    = 100
    smooth    = lambda x: np.convolve(
                    x, np.ones(window)/window, mode='valid')

    for result, color in zip(results, colors):
        name = result['name']

        axes[0].plot(smooth(result['rewards']),
                     color=color, label=name, linewidth=1.5)

        axes[1].plot(smooth(result['steps']),
                     color=color, label=name, linewidth=1.5)

        sr = [np.mean(result['success']
                      [max(0,i-window):i+1])*100
              for i in range(len(result['success']))]
        axes[2].plot(sr, color=color,
                     label=name, linewidth=1.5)

    titles  = ['Reward per Episode',
               'Steps per Episode',
               'Success Rate (%)']
    ylabels = ['Total Reward', 'Steps', 'Success Rate (%)']

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_ylim(0, 105)

    plt.suptitle(
        f'Training Curves — {GRID_SIZE}×{GRID_SIZE} Grid',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('large_grid_curves.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Saved: large_grid_curves.png")



def print_summary(results):
    print("\n")
    print("=" * 70)
    print(f"  LARGE GRID RESULTS ({GRID_SIZE}×{GRID_SIZE}, 20% obstacles)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Basic QL':>14} "
          f"{'Improved QL':>14} {'DQN':>10}")
    print("-" * 70)

    metrics = [
        ('success_rate',   'Success Rate (%)',    '{:.1f}%'),
        ('coverage',       'Q-table Coverage',    '{:.1f}%'),
        ('convergence_ep', 'Convergence Episode', 'Ep {}'),
        ('training_time',  'Training Time (s)',   '{}s'),
    ]

    for key, label, fmt in metrics:
        vals = []
        for r in results:
            v = r[key]
            if v == 'N/A':
                vals.append('N/A (net)')
            else:
                vals.append(fmt.format(v))
        print(f"{label:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>10}")

    print("=" * 70)




if __name__ == "__main__":
    print("=" * 55)
    print(f"  LARGE GRID COMPARISON ({GRID_SIZE}×{GRID_SIZE})")
    print(f"  All 3 Algorithms")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    r1 = run_basic_qlearning()
    r2 = run_improved_qlearning()
    r3 = run_dqn()

    results = [r1, r2, r3]

    print_summary(results)
    plot_paths(results)
    plot_curves(results)

    print("\n  All done!")
    print("  Files: large_grid_paths.png, large_grid_curves.png")