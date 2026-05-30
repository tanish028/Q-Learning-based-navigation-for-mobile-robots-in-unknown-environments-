import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from collections import defaultdict

GRID_SIZE       = 10
OBSTACLE_RATIO  = 0.2
SEED            = 42

ALPHA           = 0.1
GAMMA           = 0.9
EPSILON         = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.995

REWARD_GOAL     = 100
REWARD_OBSTACLE = -5
REWARD_SCALE    = 2.0

N_EPISODES      = 2000
MAX_STEPS       = 300


ACTIONS = [
    (-1,  0),   # 0: Up
    ( 1,  0),   # 1: Down
    ( 0, -1),   # 2: Left
    ( 0,  1),   # 3: Right
]
N_ACTIONS = len(ACTIONS)




class GridEnvironment:
    """
    Grid world:
    0 = free cell
    1 = obstacle
    S = start (0,0)
    G = goal  (9,9)
    """

    def __init__(self):
        self.size  = GRID_SIZE
        self.start = (0, 0)
        self.goal  = (GRID_SIZE - 1, GRID_SIZE - 1)
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
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def euclidean(self, pos1, pos2):
        return np.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2
        )

    def step(self, action):

        r, c     = self.agent_pos
        dr, dc   = ACTIONS[action]
        nr, nc   = r + dr, c + dc
        old_dist = self.manhattan(self.agent_pos, self.goal)


        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            next_pos = self.agent_pos
            reward   = REWARD_OBSTACLE
            done     = False


        elif self.grid[nr][nc] == 1:
            next_pos = self.agent_pos
            reward   = REWARD_OBSTACLE
            done     = False


        elif (nr, nc) == self.goal:
            next_pos = (nr, nc)
            reward   = REWARD_GOAL
            done     = True


        else:
            next_pos        = (nr, nc)
            new_dist        = self.manhattan(next_pos, self.goal)
            distance_reward = REWARD_SCALE * (old_dist - new_dist)
            reward          = distance_reward

            done            = False

        self.agent_pos = next_pos
        return next_pos, reward, done


class QLearningAgent:
    """
    Basic Q-Learning agent.
    Uses Q-table.
    4 actions.
    """

    def __init__(self):

        self.q_table = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPSILON

    def choose_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)   # explore
        else:
            return int(np.argmax(self.q_table[state])) # exploit

    def update(self, state, action, reward, next_state, done):

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
# Training

def train(env, agent):
    rewards_per_episode = []
    steps_per_episode   = []
    success_history     = []

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:
            action                   = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state                    = next_state
            total_reward            += reward
            steps                   += 1

        agent.decay_epsilon()

        reached_goal = done and state == env.goal
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        success_history.append(1 if reached_goal else 0)

        if (episode + 1) % 200 == 0:
            recent_sr = np.mean(success_history[-200:]) * 100
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Success: {recent_sr:.1f}%")

    return rewards_per_episode, steps_per_episode, success_history


def get_path(env, agent):

    state   = env.reset()
    path    = [state]
    done    = False
    steps   = 0
    visited = set()

    while not done and steps < MAX_STEPS:
        if state in visited:
            print("  ⚠ Loop detected — agent stuck")
            break
        visited.add(state)
        action               = int(np.argmax(agent.q_table[state]))
        next_state, _, done  = env.step(action)
        path.append(next_state)
        state                = next_state
        steps               += 1

    if state == env.goal:
        print(f"Goal reached in {len(path)-1} steps")
    else:
        print(f"Goal not reached — ends at {state}")

    return path

def plot_grid(env, path, title="Basic Q-Learning Path"):
    fig, ax = plt.subplots(figsize=(7, 7))
    size    = env.size

    for r in range(size):
        for c in range(size):
            if env.grid[r][c] == 1:
                color = '#2d3436'
            elif (r, c) == env.start:
                color = '#00b894'
            elif (r, c) == env.goal:
                color = '#d63031'
            else:
                color = '#dfe6e9'
            rect = patches.Rectangle(
                (c, size - 1 - r), 1, 1,
                linewidth=0.5, edgecolor='grey', facecolor=color
            )
            ax.add_patch(rect)

    if path and len(path) > 1:
        px = [c + 0.5 for (r, c) in path]
        py = [size - 0.5 - r for (r, c) in path]
        ax.plot(px, py, 'b-o', linewidth=2,
                markersize=5,
                label=f'Robot path ({len(path)-1} steps)',
                zorder=5)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('basic_path.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_curves(rewards, steps, success):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    window    = 50
    smooth    = lambda x: np.convolve(
                    x, np.ones(window)/window, mode='valid')

    axes[0].plot(smooth(rewards), color='#0984e3')
    axes[0].set_title('Reward per Episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(smooth(steps), color='#e17055')
    axes[1].set_title('Steps to Goal')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True, alpha=0.3)

    sr = [np.mean(success[max(0, i-window):i+1]) * 100
          for i in range(len(success))]
    axes[2].plot(sr, color='#00b894')
    axes[2].set_title('Success Rate (%)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Basic Q-Learning Training Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('basic_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("  Basic Q-Learning ")
    print("=" * 60)

    env   = GridEnvironment()
    agent = QLearningAgent()

    print(f"\n Grid:      {GRID_SIZE}×{GRID_SIZE}")
    print(f" Obstacles: {int(GRID_SIZE*GRID_SIZE*OBSTACLE_RATIO)} cells")
    print(f" Actions:   {N_ACTIONS} directions (Up/Down/Left/Right only)")
    print(f" No APF gravity")
    print(f"\n Training...\n")

    rewards, steps, success = train(env, agent)

    final_sr = np.mean(success[-200:]) * 100
    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f" Final Success Rate:  {final_sr:.1f}%")
    print(f" Final ε:             {agent.epsilon:.4f}")
    print(f" Q-table size:        {len(agent.q_table)} states learned")

    path = get_path(env, agent)
    print(f" Path length:         {len(path)-1} steps")
    print(f"{'='*60}")

    plot_grid(env, path)
    plot_curves(rewards, steps, success)
    print("\n Saved: basic_path.png, basic_curves.png")

