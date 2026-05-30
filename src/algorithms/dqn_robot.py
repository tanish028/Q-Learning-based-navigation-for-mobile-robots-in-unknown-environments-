import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GRID_SIZE = 10
OBSTACLE_RATIO = 0.2
SEED = 42

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

# Neural network part
BATCH_SIZE = 64  #we will sample 64 random experiences
MEMORY_SIZE = 10000 # we store the last 10000 exp
TARGET_UPDATE = 100 # target network is updated evry 100 steps
LEARNING_RATE = 0.001 # for gradient descent
HIDDEN_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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


class Memory :
    # Stores past experiences (s, a, r, s', done)

    def __init__ (self,capacity = MEMORY_SIZE):
        self.memory = deque(maxlen = capacity)

    def push(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def sample(self, batch_size= BATCH_SIZE):
        # this will return 5 sample lists per element

        batch = random.sample(self.memory,batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)



class DQNetwork(nn.Module):

    """
    Takes robot position [row, col] as input.
    Outputs Q-value for each of 8 possible actions.
    Input(2) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(8)
    """
    def __init__ (self,input_size=2,hidden_size =HIDDEN_SIZE,ouput_size =N_ACTIONS):

        super(DQNetwork,self).__init__()

        #layers
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,ouput_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))  # layer 1
        x = F.relu(self.fc2(x))  # layer 2
        x = self.fc3(x)  # output (we will not perfrom relu here as it removes the negative values)

        return x


class DQNAgent:

    """
    We will use two neural networks:
    main(updates evry step) and target(updates evry 100 steps)
    """

    def __init__(self):

        self.main_network = DQNetwork().to(DEVICE)
        self.target_network = DQNetwork().to(DEVICE)

        #copy from main
        self.target_network.load_state_dict(
            self.main_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.main_network.parameters(),lr=LEARNING_RATE
        )

        self.memory = Memory()
        self.epsilon = EPSILON
        self.steps_done = 0

    def state_to_tensor(self,state):
        # converting state tuple to tensor(pytorch)
        return torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

    def choose_action(self,state):

        if random.random() < self.epsilon:
            return random.randint(0,N_ACTIONS-1)

        else:
            with torch.no_grad():

                state_tensor = self.state_to_tensor(state)
                q_values = self.main_network(state_tensor)
                return int(q_values.argmax().item())


    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):

        if len(self.memory)< BATCH_SIZE:
            return

        states,actions, rewards,next_states, dones = self.memory.sample(BATCH_SIZE)


        states_t = torch.FloatTensor(states).to(DEVICE)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)

        rewards_t = torch.FloatTensor(rewards).to(DEVICE)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)

        dones_t = torch.FloatTensor(dones).to(DEVICE)

        current_q = self.main_network(states_t).gather(1,actions_t).squeeze(1)

        with torch.no_grad():

            next_q = self.target_network(next_states_t).max(1).values

            target = rewards_t + GAMMA *next_q *(1- dones_t)

        loss = F.mse_loss(current_q,target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

    def decay_epsilon(self):

        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN,self.epsilon*EPSILON_DECAY)

# Training



def train_dqn(env, agent):
    rewards_per_episode = []
    steps_per_episode   = []
    success_history     = []

    for episode in range(N_EPISODES):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:

            action = agent.choose_action(state)


            next_state, reward, done = env.step(action)


            agent.store_experience(state, action, reward,
                                   next_state, done)


            agent.train_step()

            state        = next_state
            total_reward += reward
            steps        += 1

        # Decay exploration after each episode
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
                  f"Memory: {len(agent.memory):5d} | "
                  f"Success: {recent_sr:.1f}%")

    return rewards_per_episode, steps_per_episode, success_history


def get_path_dqn(env, agent):

    state   = env.reset()
    path    = [state]
    done    = False
    steps   = 0
    visited = set()

    while not done and steps < MAX_STEPS:
        if state in visited:
            print("Loop detected")
            break
        visited.add(state)

        # Always exploit — no exploration
        with torch.no_grad():
            state_tensor = agent.state_to_tensor(state)
            q_values     = agent.main_network(state_tensor)
            action       = int(q_values.argmax().item())

        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1

    if state == env.goal:
        print(f"Goal reached in {len(path)-1} steps")
    else:
        print(f"Goal not reached — ends at {state}")

    return path


def plot_grid_dqn(env, path, title="DQN Path"):
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
                label=f'DQN path ({len(path)-1} steps)',
                zorder=5)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('dqn_path.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_curves_dqn(rewards, steps, success):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    window    = 50
    smooth    = lambda x: np.convolve(
                    x, np.ones(window)/window, mode='valid')

    axes[0].plot(smooth(rewards), color='#6c5ce7')
    axes[0].set_title('Reward per Episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(smooth(steps), color='#fd79a8')
    axes[1].set_title('Steps to Goal')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True, alpha=0.3)

    sr = [np.mean(success[max(0, i-window):i+1]) * 100
          for i in range(len(success))]
    axes[2].plot(sr, color='#00cec9')
    axes[2].set_title('Success Rate (%)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('DQN Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dqn_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  DQN Robot Path Planning")
    print("=" * 60)
    print(f"\n Device:    {DEVICE}")
    print(f" Grid:      {GRID_SIZE}×{GRID_SIZE}")
    print(f" Obstacles: {int(GRID_SIZE*GRID_SIZE*OBSTACLE_RATIO)} cells")
    print(f" Actions:   {N_ACTIONS} directions")
    print(f" Memory:    {MEMORY_SIZE} experiences")
    print(f" Batch:     {BATCH_SIZE} experiences per update")
    print(f"\n Training...\n")

    env   = Grid_Environment()
    agent = DQNAgent()

    rewards, steps, success = train_dqn(env, agent)

    final_sr = np.mean(success[-200:]) * 100
    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f" Final Success Rate:  {final_sr:.1f}%")
    print(f" Final ε:             {agent.epsilon:.4f}")
    print(f" Total steps done:    {agent.steps_done}")
    print(f"{'='*60}")

    path = get_path_dqn(env, agent)
    print(f" Path length: {len(path)-1} steps")

    plot_grid_dqn(env, path)
    plot_curves_dqn(rewards, steps, success)
    print("\n Saved: dqn_path.png, dqn_curves.png")