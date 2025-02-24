import gym
from gym import spaces
import numpy as np
import pygame
import random

TROPHY_SPRITE = pygame.image.load("./assets/trophy.png")
BACKGROUND_SPRITE = pygame.image.load("./assets/background.png")

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
        
         # Draw the background.
        rect = pygame.Rect(0, 0, self.width, self.height)
        scaled_background = pygame.transform.scale(BACKGROUND_SPRITE, (self.width, self.height))
        self.screen.blit(scaled_background, rect)
        
        # Draw grid lines.
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (15, 15, 15), rect, 1)
        
        # Draw obstacles.
        for (i, j) in self.obstacles:
            rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (100, 100, 100), rect)
        
        # Draw the agent as a green square.
        i, j = self.agent_pos
        agent_rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), agent_rect)
        
        # Draw the goal as a trophy.
        i, j = self.grid_size[0] - 1, self.grid_size[1] - 1
        trophy_rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                  self.cell_size, self.cell_size)
        scaled_trophy = pygame.transform.scale(TROPHY_SPRITE, (self.cell_size, self.cell_size))
        self.screen.blit(scaled_trophy, trophy_rect)

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

def find_dfs_path(env):
        start = (0, 0)
        goal = (env.grid_size[0]-1, env.grid_size[1]-1)
        obstacles = env.obstacles
        grid_size = env.grid_size

        stack = [(start, [])]  # Each element is (current_pos, path_actions)
        visited = set()

        while stack:
            current_pos, path_actions = stack.pop()
            if current_pos == goal:
                return path_actions
            if current_pos in visited:
                continue
            visited.add(current_pos)
            for action in range(4):
                x, y = current_pos
                if action == 0:  # Up
                    new_x = max(x - 1, 0)
                    new_y = y
                elif action == 1:  # Down
                    new_x = min(x + 1, grid_size[0] - 1)
                    new_y = y
                elif action == 2:  # Left
                    new_y = max(y - 1, 0)
                    new_x = x
                elif action == 3:  # Right
                    new_y = min(y + 1, grid_size[1] - 1)
                    new_x = x
                new_pos = (new_x, new_y)
                # Determine next_state based on obstacle check
                if new_pos in obstacles:
                    next_state = current_pos
                else:
                    next_state = new_pos
                if next_state not in visited:
                    stack.append((next_state, path_actions + [action]))
        return None

if __name__ == "__main__":
    env = ComplicatedRaceTrackEnvPygame(grid_size=(10, 10), cell_size=50)

    

    path = find_dfs_path(env)

    if path is None:
        print("No path found!")
    else:
        print(f"Path found with {len(path)} steps. Executing...")
        state, _ = env.reset()
        done = False
        for action in path:
            if done:
                break
            # Render and step
            env.render()
            state, reward, done, _, _ = env.step(action)
            print(f"Action: {action}, Position: {env.agent_pos}, Reward: {reward}")
            pygame.time.wait(500)  # Half-second delay to visualize steps

        # Final render and close
        env.render()
        pygame.time.wait(2000)  # Wait to show final state
        env.close()
        
