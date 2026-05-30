import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from collections import defaultdict

# GRID

GRID_SIZE = 10
OBSTACLE_RATIO =  0.2
SEED = 42

ALPHA = 0.5
ALPHA_MIN = 0.01
ALPHA_DECAY = 0.999

GAMMA = 0.9

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998

ZETA = 0.5

REWARD_GOAL = 100
REWARD_OBSTACLE = -5
REWARD_SCALE = 2.0

N_EPISODES = 2000
MAX_STEPS = 300

ACTIONS = [
    (-1,0),(1,0),(0,-1),(0,1),(-1,1),(-1,-1),(1,1),(1,-1)
]
N_ACTIONS = len(ACTIONS)


class Grid_Environment:

    def __init__(self):
        self.size = GRID_SIZE
        self.start = (0,0)
        self.goal = (GRID_SIZE-1,GRID_SIZE-1)
        self.reset_environment()

    def reset_environment(self):
            # for creating new grid

            np.random.seed(SEED)
            random.seed(SEED)
            self.grid = np.zeros((self.size,self.size),dtype = int)

            num_obstacles = int(GRID_SIZE*GRID_SIZE*OBSTACLE_RATIO)
            obstacle_cells = random.sample([(r,c) for r in range(self.size) for c in range(self.size)
            if (r,c) not in [(0,0), (self.size-1,self.size-1)]],num_obstacles)

            for (r,c) in obstacle_cells:
                self.grid[r][c] = 1


            self.grid[self.start] = 0
            self.grid[self.goal] = 0

    def reset(self):
        # Reset agent to start position for a new episode.
        self.agent_pos = self.start
        return self.agent_pos

    def manhattan(self,pos1,pos2):
        return abs(pos1[0]-pos2[0])+ abs(pos1[1]-pos2[1])

    def euclidean(self,pos1,pos2):
        return np.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2
        )


    def gravity_bonus(self,pos):
        """
        Formula: ZETA / distance

        if Far from goal,small bonus  (e.g. 0.5/10 = 0.05)
        if Close to goal,large bonus  (e.g. 0.5/1  = 0.50)

        This will help in reaching the goal much faster when compared to normal reward function
        """
        dist = self.euclidean(pos,self.goal)
        if dist ==0:
            return 0
        return ZETA/dist

    def step(self,action):

        r,c = self.agent_pos

        dr,dc = ACTIONS[action]
        nr,nc = r+dr, c+dc

        # we will record this for calculating reward from the new reward function
        old_dist = self.manhattan(self.agent_pos,self.goal)

        if nr<0 or nr >= self.size or nc<0 or nc >= self.size:
            # If Hit wall — stay in place, small penalty
            next_pos = self.agent_pos
            reward = REWARD_OBSTACLE
            done = False
        elif self.grid[nr][nc] == 1:
            # if hit obstacle - penalty
            next_pos = self.agent_pos # stay in place
            reward = REWARD_OBSTACLE
            done = False
        elif (nr, nc) == self.goal:
            # Reached goal
            next_pos = (nr, nc)
            reward = REWARD_GOAL
            done = True
        else:
            next_pos = (nr, nc)

            new_dist = self.manhattan(next_pos,self.goal)

            distance_reward = REWARD_SCALE* (old_dist-new_dist)

            gravity_bonus = self.gravity_bonus(next_pos)

            #new reward function
            reward = distance_reward+ gravity_bonus
            done = False

        self.agent_pos = next_pos
        return next_pos, reward, done

class ImprovedQLearningAgent:

    def __init__(self):

        self.q_table = defaultdict(lambda : np.zeros(N_ACTIONS))

        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def choose_action(self,state):

        if random.random() < self.epsilon:
            return random.randint(0,N_ACTIONS-1) # choose random action
        else:
            return int(np.argmax(self.q_table[state])) # otherwise we will choose the best possible action


    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]

        if done:
            target = reward  # no future rewards possible

        else:
            target = reward + self.gamma* np.max(self.q_table[next_state])

        td_error = target - current_q

        self.q_table[state][action] += self.alpha * td_error


    def decay_parameters(self):
        """
        Called after each episode.
        Dynamically reduce alpha and epsilon over time.
        """

        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


        if self.alpha > ALPHA_MIN:
            self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)



# Training

def train(env,agent):

    rewards_per_episode = []
    steps_per_episode = []
    success_history = []


    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward,done = env.step(action)
            agent.update(state,action,reward,next_state,done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_parameters()

        reached_goal = done and state == env.goal
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        success_history.append(1 if reached_goal else 0)

        if(episode +1)% 200 == 0 :
            recent_sr = np.mean(success_history[-200:]) * 100
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Steps: {steps:4d} | "
                  f"α: {agent.alpha:.4f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Success: {recent_sr:.1f}%")

    return rewards_per_episode,steps_per_episode,success_history

def get_path(env, agent):
    """
    Extract learned path using greedy policy.
    No exploration — always picks best known action.
    """
    state   = env.reset()      # reset MUST be called first
    path    = [state]
    done    = False
    steps   = 0
    visited = set()

    while not done and steps < MAX_STEPS:
        if state in visited:
            print("Loop detected — agent stuck")
            break
        visited.add(state)

        # Always pick best action — no randomness
        action               = int(np.argmax(agent.q_table[state]))
        next_state, _, done  = env.step(action)
        path.append(next_state)
        state                = next_state
        steps               += 1


    if state == env.goal:
        print(f"Goal reached in {len(path)-1} steps")
    else:
        print(f"Goal not reached — path ends at {state}")

    return path

def plot_grid(env, path, title="Improved Q-Learning Path"):


    print(f"\n Plotting path with {len(path)} points: {path[:5]}...")

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
                markersize=5, label=f'Robot path ({len(path)} steps)',
                zorder=5)
        print(f" Path drawn successfully.")
    else:
        print(f" WARNING: Path too short to draw ({len(path)} points)")
        print(f"          Agent may not have learned a valid path yet.")

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('improved_path.png', dpi=150, bbox_inches='tight')
    print(f" Saved to improved_path.png")
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

    plt.suptitle('Improved Q-Learning Training Curves',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('improved_curves.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("  Improved Q-Learning — 8 Actions + APF + Dynamic Params")
    print("=" * 60)

    env   = Grid_Environment()
    agent = ImprovedQLearningAgent()

    print(f"\n Grid:      {GRID_SIZE}×{GRID_SIZE}")
    print(f" Obstacles: {int(GRID_SIZE*GRID_SIZE*OBSTACLE_RATIO)} cells")
    print(f" Actions:   {N_ACTIONS} directions")
    print(f" APF ζ:     {ZETA}")
    print(f"\n Training...\n")

    rewards, steps, success = train(env, agent)

    final_sr = np.mean(success[-200:]) * 100
    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f" Final Success Rate:  {final_sr:.1f}%")
    print(f" Final α:             {agent.alpha:.4f}")
    print(f" Final ε:             {agent.epsilon:.4f}")
    print(f" Q-table size:        {len(agent.q_table)} states learned")

    path = get_path(env, agent)
    print(f" Path length:         {len(path)} steps")
    print(f"{'='*60}")

    plot_grid(env, path)
    plot_curves(rewards, steps, success)
    print("\n Saved: improved_path.png, improved_curves.png")