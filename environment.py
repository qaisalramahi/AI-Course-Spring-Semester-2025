import gym
from gym import spaces
import numpy as np
import pygame

TROPHY_SPRITE = pygame.image.load("./assets/trophy.png")
MAP_SPRITE = pygame.image.load("./assets/Map.png")
CAR_SPRITE = pygame.image.load("./assets/Car.png")
GRID_SIZE = (15, 15)
CELL_SIZE = 32
START_POS = (0, 0)
GOAL_POS = (14, 14)
PATH_TILES = {
    (0,0), (1,0), (2,0), 
    (2,1), 
    (2,2), 
    (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3), (10,3), (11,3), (12,3),
    (9,4), (12,4),
    (9,5), (12,5),
    (2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (8,6), (9,6), (10,6), (11,6), (12,6),
    (2,7), (5,7), (7,7),
    (2,8), (5,8), (6,8), (7,8), (10,8), (11,8), (12,8),
    (2,9), (5,9), (7,9), (10,9), (12,9),
    (2,10), (3,10), (4,10), (5,10), (6,10), (7,10), (8,10), (9,10), (10,10), (12,10),
    (8,11), (12,11),
    (6,12), (7,12), (8,12), (9,12), (10,12), (11,12), (12,12),
    (6,13),
    (6,14), (7,14), (8,14), (9,14), (10,14), (11,14), (12,14), (13,14), (14,14)
}
GAS_TILES = {
    (11,3),
    (2,8), (6,8),
    (10,12)
}
GAS_MAX = 20

class ComplicatedRaceTrackEnvPygame(gym.Env):
    """A Gym environment for a grid race track with obstacles rendered with Pygame."""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_size=GRID_SIZE, cell_size=CELL_SIZE):
        super().__init__()
        self.grid_size = grid_size  # (rows, columns)
        self.cell_size = cell_size  # pixel size of each grid cell
        self.width = grid_size[1] * cell_size
        self.height = grid_size[0] * cell_size
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right.
        self.action_space = spaces.Discrete(4)
        # Observation: a grid of shape (rows, cols) with values:
        #   0.0 for free space, 0.5 for obstacles, 1.0 for the agent, 0.75 for gas
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
        
        self.agent_pos = START_POS
        self._prev_pos = self.agent_pos
        self.gas = GAS_MAX
        self.done = False
        self.visited = set({START_POS})
    
    def get_possible_actions(self, pos):
        # print(f"Getting possible actions for position {pos}")
        actions = []
        x, y = pos
        if (x - 1, y) in PATH_TILES: # Left
            actions.append(0)
        if (x + 1, y) in PATH_TILES: # Right
            actions.append(1)
        if (x, y - 1) in PATH_TILES: # Up
            actions.append(2)
        if (x, y + 1) in PATH_TILES: # Down
            actions.append(3)
        # print(f"Possible actions: {actions}")
        return actions
    
    def check(self, path_actions):
        path_gas = GAS_MAX
        path_pos = START_POS
        for action in path_actions:
            if path_gas <= 0:
                print("Invalid path: out of gas")
                return False, 0, None

            if action == 0: # Left
                new_pos = (path_pos[0] - 1, path_pos[1])
            elif action == 1: # Right
                new_pos = (path_pos[0] + 1, path_pos[1])
            elif action == 2: # Up
                new_pos = (path_pos[0], path_pos[1] - 1)
            elif action == 3: # Down
                new_pos = (path_pos[0], path_pos[1] + 1)
            
            if new_pos not in PATH_TILES:
                print("Invalid path: not a path tile")
                return False, 0, None
            if new_pos in GAS_TILES:
                path_gas = GAS_MAX
            path_gas -= 1
            path_pos = new_pos

        if path_gas <= 0:
            print("Invalid path: out of gas")
            return False, 0, None
        return True, path_gas, path_pos

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = START_POS
        self.done = False
        self.gas = GAS_MAX
        return self._get_obs(), {}
    
    def step(self, action):

        if self.done:
            return self._get_obs(), 0.0, True, {}, {}
        
        # If the agent is out of gas, penalize and finish the episode.
        if self.gas <= 0:
            self.done = True
            return self._get_obs(), -100.0, True, {}, {}
        
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if action == 0:  # Left
            new_x = max(x - 1, 0)
        elif action == 1:  # Right
            new_x = min(x + 1, self.grid_size[0] - 1)
        elif action == 2:  # Up
            new_y = max(y - 1, 0)
        elif action == 3:  # Down
            new_y = min(y + 1, self.grid_size[1] - 1)
        
        # If the new cell is not a path, penalize and keep the agent in place.
        if (new_x, new_y) not in PATH_TILES:
            reward = -10.0
            new_x, new_y = x, y
        
        # If the new cell is a gas tile, refill the gas and give a small reward.
        elif (new_x, new_y) in GAS_TILES:
            self.gas = GAS_MAX
            reward = 5.0 # Small reward for finding gas.
        else:
            self.gas -= 1   # Consume gas.
            reward = 0.1 # Small reward for moving.

        # Give a big negative reward if the agent revisits a cell.
        if (new_x, new_y) in self.visited:
            # print("BAD: Revisited cell")
            reward = -1.0
        
        self._prev_pos = self.agent_pos
        self.agent_pos = (new_x, new_y)
        self.visited.add((new_x, new_y))
        
        # If the goal (bottom-right) is reached, give a reward and finish the episode.
        if self.agent_pos == (self.grid_size[0] - 1, self.grid_size[1] - 1):
            reward = 100.0
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
        scaled_background = pygame.transform.scale(MAP_SPRITE, (self.width, self.height))
        self.screen.blit(scaled_background, rect)
        
        # Draw the agent as a car.
        x, y = self.agent_pos
        agent_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
        transformed_agent = pygame.transform.scale(CAR_SPRITE, (self.cell_size, self.cell_size))
        # Rotate the car based on the previous position.
        if self._prev_pos[0] < self.agent_pos[0]:
            transformed_agent = pygame.transform.rotate(transformed_agent, 90)
        elif self._prev_pos[0] > self.agent_pos[0]:
            transformed_agent = pygame.transform.rotate(transformed_agent, -90)
        elif self._prev_pos[1] < self.agent_pos[1]:
            transformed_agent = pygame.transform.rotate(transformed_agent, 180)
        
        self.screen.blit(transformed_agent, agent_rect)

        
        # Draw the goal as a trophy.
        x, y = GOAL_POS
        trophy_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                  self.cell_size, self.cell_size)
        scaled_trophy = pygame.transform.scale(TROPHY_SPRITE, (self.cell_size, self.cell_size))
        self.screen.blit(scaled_trophy, trophy_rect)

        # Print the gas level.
        font = pygame.font.Font(None, 36)
        text = font.render(f"Gas: {self.gas}", True, (255, 0, 0))
        self.screen.blit(text, (380, 5))

        pygame.display.flip()
        self.clock.tick(15)  # Limit to 15 FPS.
    
    def _get_obs(self):
        """
        Returns a grid observation:
          - 0.0 for non-path cells.
          - 0.25 for path cells.
          - 0.75 for gas cells.
          - 1.0 for the agent.
        """
        obs = np.zeros(self.grid_size, dtype=np.float32)
        for (i, j) in PATH_TILES:
            obs[i, j] = 0.25
        for (i, j) in GAS_TILES:
            obs[i, j] = 0.75
        i, j = self.agent_pos
        obs[i, j] = 1.0
        return obs
    
    def close(self):
        pygame.quit()