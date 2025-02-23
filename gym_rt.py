import gym
from gym import spaces
import numpy as np
import pygame
import random

class ComplicatedRaceTrackEnvPygame(gym.Env):
    """A Gym environment for a grid race track with obstacles rendered with Pygame."""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_size=(10, 10), cell_size=50):
        super().__init__()
        self.grid_size = grid_size  # (rows, columns)
        self.cell_size = cell_size  # pixel size of each grid cell
        self.width = grid_size[1] * cell_size
        self.height = grid_size[0] * cell_size
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right.
        self.action_space = spaces.Discrete(4)
        # Observation: a grid of shape (rows, cols) with values:
        #   0.0 for free space, 0.5 for obstacles, 1.0 for the agent.
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(grid_size[0], grid_size[1]), 
            dtype=np.float32
        )
        
        # Initialize Pygame.
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Complicated Race Track")
        self.clock = pygame.time.Clock()
        
        # Define obstacles: 
        # Create a wall on row 3 with a gap at column 4.
        # Create a wall on row 6 with a gap at column 7.
        self.obstacles = set()
        for j in range(grid_size[1]):
            if j != 4:
                self.obstacles.add((3, j))
        for j in range(grid_size[1]):
            if j != 7:
                self.obstacles.add((6, j))
        # Ensure the start (0,0) and goal (last cell) are free.
        self.obstacles.discard((0, 0))
        self.obstacles.discard((grid_size[0]-1, grid_size[1]-1))
        
        self.agent_pos = (0, 0)
        self.done = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (0, 0)
        self.done = False
        return self._get_obs(), {}
    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}, {}
        
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if action == 0:  # Up
            new_x = max(x - 1, 0)
        elif action == 1:  # Down
            new_x = min(x + 1, self.grid_size[0] - 1)
        elif action == 2:  # Left
            new_y = max(y - 1, 0)
        elif action == 3:  # Right
            new_y = min(y + 1, self.grid_size[1] - 1)
        
        # If the new cell is an obstacle, penalize and keep the agent in place.
        if (new_x, new_y) in self.obstacles:
            reward = -0.5
            new_x, new_y = x, y
        else:
            reward = -0.01  # Small step penalty.
        
        self.agent_pos = (new_x, new_y)
        
        # If the goal (bottom-right) is reached, give a reward and finish the episode.
        if self.agent_pos == (self.grid_size[0] - 1, self.grid_size[1] - 1):
            reward = 1.0
            self.done = True
        
        return self._get_obs(), reward, self.done, {}, {}
    
    def render(self, mode="human"):
        # Process Pygame events so the window remains responsive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
        
        # Clear the screen.
        self.screen.fill((0, 0, 0))
        
        # Draw grid lines.
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
        
        # Draw obstacles.
        for (i, j) in self.obstacles:
            rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (150, 150, 150), rect)
        
        # Draw the agent as a green square.
        i, j = self.agent_pos
        agent_rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), agent_rect)
        
        pygame.display.flip()
        self.clock.tick(15)  # Limit to 15 FPS.
    
    def _get_obs(self):
        """
        Returns a grid observation:
          - 0.0 for free space,
          - 0.5 for obstacles,
          - 1.0 for the agent's cell.
        """
        obs = np.zeros(self.grid_size, dtype=np.float32)
        for (i, j) in self.obstacles:
            obs[i, j] = 0.5
        i, j = self.agent_pos
        obs[i, j] = 1.0
        return obs
    
    def close(self):
        pygame.quit()

# Q-Learning integration with the Pygame environment.
if __name__ == "__main__":
    # Q-learning hyperparameters.
    num_episodes = 1500
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_rate = 0.99
    
    # Exploration parameters (epsilon-greedy).
    epsilon = 1.0          # Exploration rate.
    max_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.005
    
    env = ComplicatedRaceTrackEnvPygame(grid_size=(10, 10), cell_size=50)
    
    # Initialize the Q-table with keys ((row, col), action).
    q_table = {}
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            for action in range(env.action_space.n):
                q_table[((i, j), action)] = 0.0
    
    rewards_all_episodes = []
    
    # Training loop.
    for episode in range(num_episodes):
        state, _ = env.reset()
        state_pos = env.agent_pos
        done = False
        rewards_current_episode = 0
        
        for step in range(max_steps_per_episode):
            # Choose action using epsilon-greedy policy.
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [q_table[(state_pos, a)] for a in range(env.action_space.n)]
                action = int(np.argmax(q_values))
            
            new_state, reward, done, _, _ = env.step(action)
            new_state_pos = env.agent_pos
            
            # Q-learning update.
            old_value = q_table[(state_pos, action)]
            next_max = max([q_table[(new_state_pos, a)] for a in range(env.action_space.n)])
            new_value = old_value + learning_rate * (reward + discount_rate * next_max - old_value)
            q_table[(state_pos, action)] = new_value
            
            state_pos = new_state_pos
            rewards_current_episode += reward
            
            if done:
                break
        
        # Decay epsilon after each episode.
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        rewards_all_episodes.append(rewards_current_episode)
    
    print("Training finished.\n")
    
    # Test the learned policy.
    state, _ = env.reset()
    state_pos = env.agent_pos
    done = False
    print("Trained Q-Table Policy Execution:")
    while not done:
        env.render()
        q_values = [q_table[(state_pos, a)] for a in range(env.action_space.n)]
        action = int(np.argmax(q_values))
        state, reward, done, _, _ = env.step(action)
        state_pos = env.agent_pos
        print(f"Action: {action}, Position: {state_pos}, Reward: {reward}")
    
    # Final render and cleanup.
    env.render()
    env.close()
