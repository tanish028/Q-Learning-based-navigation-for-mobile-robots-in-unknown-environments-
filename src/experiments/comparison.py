#


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

GRID_SIZE       = 10
OBSTACLE_RATIO  = 0.2
SEED            = 42

GAMMA           = 0.9
EPSILON         = 1.0
EPSILON_MIN     = 0.01

REWARD_GOAL     = 100
REWARD_OBSTACLE = -5
REWARD_SCALE    = 2.0
ZETA            = 0.5

N_EPISODES      = 2000
MAX_STEPS       = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic uses 4 actions, Improved+DQN use 8
ACTIONS_4 = [(-1,0),(1,0),(0,-1),(0,1)]
ACTIONS_8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(-1,-1),(1,1),(1,-1)]

class GridEnvironment:

    def __init__(self, actions):
        self.size    = GRID_SIZE
        self.start   = (0, 0)
        self.goal    = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.actions = actions
        self._build_grid()

    def _build_grid(self):
        np.random.seed(SEED)
        random.seed(SEED)
        self.grid = np.zeros((self.size, self.size), dtype=int)

        all_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) != self.start and (r, c) != self.goal
        ]
        n_obstacles    = int(self.size * self.size * OBSTACLE_RATIO)
        obstacle_cells = random.sample(all_cells, n_obstacles)
        for (r, c) in obstacle_cells:
            self.grid[r][c] = 1

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def manhattan(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def euclidean(self, pos1, pos2):
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def gravity_bonus(self, pos):
        dist = self.euclidean(pos, self.goal)
        if dist == 0:
            return 0
        return ZETA / dist

    def step(self, action, use_apf=False):
        r, c     = self.agent_pos
        dr, dc   = self.actions[action]
        nr, nc   = r + dr, c + dc
        old_dist = self.manhattan(self.agent_pos, self.goal)

        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            self.agent_pos = self.agent_pos
            return self.agent_pos, REWARD_OBSTACLE, False

        elif self.grid[nr][nc] == 1:
            self.agent_pos = self.agent_pos
            return self.agent_pos, REWARD_OBSTACLE, False

        elif (nr, nc) == self.goal:
            self.agent_pos = (nr, nc)
            return (nr, nc), REWARD_GOAL, True

        else:
            next_pos        = (nr, nc)
            new_dist        = self.manhattan(next_pos, self.goal)
            distance_reward = REWARD_SCALE * (old_dist - new_dist)
            gravity         = self.gravity_bonus(next_pos) if use_apf else 0
            reward          = distance_reward + gravity
            self.agent_pos  = next_pos
            return next_pos, reward, False

def convergence_episode(success_history, threshold=0.9, window=50):
    """First episode where rolling success rate >= threshold."""
    for i in range(window, len(success_history)):
        if np.mean(success_history[i-window:i]) >= threshold:
            return i
    return len(success_history)   # never converged


def path_smoothness(path):
    """
    Count direction changes in path.
    Fewer changes = smoother path = better quality.
    A straight diagonal has 0 changes.
    A zigzag path has many changes.
    """
    if len(path) < 3:
        return 0
    changes = 0
    prev_dir = (path[1][0]-path[0][0], path[1][1]-path[0][1])
    for i in range(2, len(path)):
        curr_dir = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        if curr_dir != prev_dir:
            changes += 1
        prev_dir = curr_dir
    return changes


def collect_metrics(name, success_history, rewards, steps, path,
                    training_time):
    """Package all metrics into one dictionary."""
    return {
        'name':             name,
        'success_rate':     round(np.mean(success_history[-200:])*100, 1),
        'avg_steps':        round(np.mean(steps[-200:]), 1),
        'avg_reward':       round(np.mean(rewards[-200:]), 1),
        'convergence_ep':   convergence_episode(success_history),
        'path_length':      len(path) - 1,
        'path_smoothness':  path_smoothness(path),
        'training_time':    round(training_time, 1),
    }


def run_basic_qlearning():
    print("\n" + "="*50)
    print(" Running: Basic Q-Learning (4 actions, no APF)")
    print("="*50)

    env   = GridEnvironment(actions=ACTIONS_4)
    alpha = 0.1
    eps   = EPSILON

    q_table = defaultdict(lambda: np.zeros(len(ACTIONS_4)))

    rewards_hist = []
    steps_hist   = []
    success_hist = []

    start_time = time.time()

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:

            if random.random() < eps:
                action = random.randint(0, len(ACTIONS_4)-1)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done = env.step(action, use_apf=False)


            curr_q  = q_table[state][action]
            target  = reward if done else \
                      reward + GAMMA * np.max(q_table[next_state])
            q_table[state][action] += alpha * (target - curr_q)

            state        = next_state
            total_reward += reward
            steps        += 1


        eps = max(EPSILON_MIN, eps * 0.995)

        reached = done and state == env.goal
        rewards_hist.append(total_reward)
        steps_hist.append(steps)
        success_hist.append(1 if reached else 0)

        if (episode+1) % 200 == 0:
            sr = np.mean(success_hist[-200:])*100
            print(f"  Ep {episode+1:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"Success: {sr:.1f}%")

    training_time = time.time() - start_time


    state   = env.reset()
    path    = [state]
    visited = set()
    done    = False
    steps   = 0
    while not done and steps < MAX_STEPS:
        if state in visited:
            break
        visited.add(state)
        action             = int(np.argmax(q_table[state]))
        next_state, _, done = env.step(action, use_apf=False)
        path.append(next_state)
        state = next_state
        steps += 1

    metrics = collect_metrics(
        "Basic Q-Learning",
        success_hist, rewards_hist, steps_hist,
        path, training_time
    )

    return metrics, rewards_hist, steps_hist, success_hist, path, env



def run_improved_qlearning():
    print("\n" + "="*50)
    print(" Running: Improved Q-Learning (8 actions + APF)")
    print("="*50)

    env      = GridEnvironment(actions=ACTIONS_8)
    alpha    = 0.5
    eps      = EPSILON
    q_table  = defaultdict(lambda: np.zeros(len(ACTIONS_8)))

    rewards_hist = []
    steps_hist   = []
    success_hist = []

    start_time = time.time()

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:
            if random.random() < eps:
                action = random.randint(0, len(ACTIONS_8)-1)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done = env.step(action, use_apf=True)

            curr_q  = q_table[state][action]
            target  = reward if done else \
                      reward + GAMMA * np.max(q_table[next_state])
            q_table[state][action] += alpha * (target - curr_q)

            state        = next_state
            total_reward += reward
            steps        += 1

        eps   = max(EPSILON_MIN, eps * 0.998)
        alpha = max(0.01,        alpha * 0.999)

        reached = done and state == env.goal
        rewards_hist.append(total_reward)
        steps_hist.append(steps)
        success_hist.append(1 if reached else 0)

        if (episode+1) % 200 == 0:
            sr = np.mean(success_hist[-200:])*100
            print(f"  Ep {episode+1:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"Success: {sr:.1f}%")

    training_time = time.time() - start_time

    state   = env.reset()
    path    = [state]
    visited = set()
    done    = False
    steps   = 0
    while not done and steps < MAX_STEPS:
        if state in visited:
            break
        visited.add(state)
        action              = int(np.argmax(q_table[state]))
        next_state, _, done = env.step(action, use_apf=True)
        path.append(next_state)
        state = next_state
        steps += 1

    metrics = collect_metrics(
        "Improved Q-Learning",
        success_hist, rewards_hist, steps_hist,
        path, training_time
    )

    return metrics, rewards_hist, steps_hist, success_hist, path, env




class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(ACTIONS_8))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_dqn():
    print("\n" + "="*50)
    print(" Running: DQN (8 actions + APF + Neural Network)")
    print("="*50)

    env            = GridEnvironment(actions=ACTIONS_8)
    main_net       = DQNetwork().to(DEVICE)
    target_net     = DQNetwork().to(DEVICE)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()
    optimizer      = optim.Adam(main_net.parameters(), lr=0.001)
    memory         = deque(maxlen=10000)

    eps        = EPSILON
    steps_done = 0

    rewards_hist = []
    steps_hist   = []
    success_hist = []

    start_time = time.time()

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:

            if random.random() < eps:
                action = random.randint(0, len(ACTIONS_8)-1)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = int(main_net(st).argmax().item())

            next_state, reward, done = env.step(action, use_apf=True)
            memory.append((state, action, reward, next_state, done))
            state        = next_state
            total_reward += reward
            steps        += 1
            steps_done   += 1


            if len(memory) >= 64:
                batch  = random.sample(memory, 64)
                s, a, r, ns, d = zip(*batch)

                s_t  = torch.FloatTensor(s).to(DEVICE)
                a_t  = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
                r_t  = torch.FloatTensor(r).to(DEVICE)
                ns_t = torch.FloatTensor(ns).to(DEVICE)
                d_t  = torch.FloatTensor(d).to(DEVICE)

                curr_q  = main_net(s_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_t).max(1).values
                    target = r_t + GAMMA * next_q * (1 - d_t)

                loss = F.mse_loss(curr_q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if steps_done % 100 == 0:
                target_net.load_state_dict(main_net.state_dict())

        eps = max(EPSILON_MIN, eps * 0.998)

        reached = done and state == env.goal
        rewards_hist.append(total_reward)
        steps_hist.append(steps)
        success_hist.append(1 if reached else 0)

        if (episode+1) % 200 == 0:
            sr = np.mean(success_hist[-200:])*100
            print(f"  Ep {episode+1:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {eps:.3f} | "
                  f"Memory: {len(memory):5d} | "
                  f"Success: {sr:.1f}%")

    training_time = time.time() - start_time


    state   = env.reset()
    path    = [state]
    visited = set()
    done    = False
    steps   = 0
    while not done and steps < MAX_STEPS:
        if state in visited:
            break
        visited.add(state)
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = int(main_net(st).argmax().item())
        next_state, _, done = env.step(action, use_apf=True)
        path.append(next_state)
        state = next_state
        steps += 1

    metrics = collect_metrics(
        "DQN",
        success_hist, rewards_hist, steps_hist,
        path, training_time
    )

    return metrics, rewards_hist, steps_hist, success_hist, path, env


def print_results_table(all_metrics):
    print("\n")
    print("=" * 75)
    print("  FINAL COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Metric':<25} {'Basic QL':>14} {'Improved QL':>14} {'DQN':>14}")
    print("-" * 75)

    metrics_to_show = [
        ('success_rate',    'Success Rate (%)',      '{}%'),
        ('path_length',     'Path Length (steps)',   '{}'),
        ('path_smoothness', 'Direction Changes',     '{}'),
        ('avg_steps',       'Avg Steps (last 200)',  '{}'),
        ('avg_reward',      'Avg Reward (last 200)', '{}'),
        ('convergence_ep',  'Convergence Episode',   'Ep {}'),
        ('training_time',   'Training Time (sec)',   '{}s'),
    ]

    for key, label, fmt in metrics_to_show:
        vals = [fmt.format(m[key]) for m in all_metrics]
        print(f"{label:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    print("=" * 75)


def plot_combined_curves(all_rewards, all_steps, all_success, names):
    """Plot all three algorithms on same axes for direct comparison."""
    colors = ['#0984e3', '#00b894', '#6c5ce7']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    window = 50
    smooth = lambda x: np.convolve(
                x, np.ones(window)/window, mode='valid')

    for i, (rewards, steps, success, name, color) in enumerate(
            zip(all_rewards, all_steps, all_success, names, colors)):

        axes[0].plot(smooth(rewards), color=color, label=name, linewidth=1.5)
        axes[1].plot(smooth(steps),   color=color, label=name, linewidth=1.5)

        sr = [np.mean(success[max(0,j-window):j+1])*100
              for j in range(len(success))]
        axes[2].plot(sr, color=color, label=name, linewidth=1.5)

    titles  = ['Reward per Episode', 'Steps per Episode', 'Success Rate (%)']
    ylabels = ['Total Reward',       'Steps',             'Success Rate (%)']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_ylim(0, 105)
    plt.suptitle('Algorithm Comparison — Training Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(" Saved: comparison_curves.png")


def plot_all_paths(envs, paths, names):
    """Show all three paths side by side on same grid."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors_path = ['#0984e3', '#00b894', '#6c5ce7']

    for ax, env, path, name, color in zip(
            axes, envs, paths, names, colors_path):

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
                    linewidth=0.3, edgecolor='grey', facecolor=fc)
                ax.add_patch(rect)

        if path and len(path) > 1:
            px = [c+0.5 for (r,c) in path]
            py = [size-0.5-r for (r,c) in path]
            ax.plot(px, py, color=color, linewidth=2,
                    marker='o', markersize=4,
                    label=f'{name}\n({len(path)-1} steps)')

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Path Comparison — All Three Algorithms',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_paths.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(" Saved: comparison_paths.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  FULL ALGORITHM COMPARISON — DA221 Project")
    print(f"  Device: {DEVICE}")
    print("=" * 60)


    m1, r1, st1, su1, p1, e1 = run_basic_qlearning()
    m2, r2, st2, su2, p2, e2 = run_improved_qlearning()
    m3, r3, st3, su3, p3, e3 = run_dqn()

    all_metrics = [m1, m2, m3]
    names       = [m['name'] for m in all_metrics]


    print_results_table(all_metrics)


    plot_combined_curves(
        [r1, r2, r3], [st1, st2, st3], [su1, su2, su3], names
    )
    plot_all_paths(
        [e1, e2, e3], [p1, p2, p3], names
    )

    print("\n All results saved!")
    print(" Files: comparison_curves.png, comparison_paths.png")