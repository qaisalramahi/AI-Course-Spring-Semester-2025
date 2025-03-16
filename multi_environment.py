import pygame
import numpy as np
import random

# Environment constants
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

# Define player colors for visualization
PLAYER_COLORS = [
    (255, 0, 0),    # Red for player 1
    (0, 0, 255),    # Blue for player 2
    (0, 255, 0),    # Green for additional players
    (255, 255, 0)   # Yellow for additional players
]

# Define action names for debugging
ACTION_NAMES = ["Left", "Right", "Up", "Down"]

# Define the actions and their effects
ACTIONS = [
    (-1, 0),  # Left: dx, dy
    (1, 0),   # Right: dx, dy
    (0, -1),  # Up: dx, dy
    (0, 1)    # Down: dx, dy
]

# Helper functions
def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_path(start, goal, path_tiles, occupied_positions=None):
    """
    Find a path from start to goal using breadth-first search.
    Avoids occupied positions if provided.
    Returns a list of positions forming the path.
    """
    if occupied_positions is None:
        occupied_positions = set()
    
    if start == goal:
        return [start]
    
    queue = [(start, [])]  # (pos, path_so_far)
    visited = {start}
    
    while queue:
        (x, y), path = queue.pop(0)
        
        # Try all four directions
        for i, (dx, dy) in enumerate(ACTIONS):
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            # Check if valid move
            if (new_pos in path_tiles and 
                new_pos not in visited and 
                new_pos not in occupied_positions):
                
                new_path = path + [new_pos]
                
                # Found the goal
                if new_pos == goal:
                    return [start] + new_path
                
                # Add to queue
                queue.append((new_pos, new_path))
                visited.add(new_pos)
    
    # No path found
    return None

class MultiAgentState:
    """Class to represent the complete state of the multi-agent environment"""
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.positions = [START_POS for _ in range(num_agents)]
        self.prev_positions = [START_POS for _ in range(num_agents)]
        self.gas_levels = [GAS_MAX for _ in range(num_agents)]
        self.done = [False for _ in range(num_agents)]
        self.visited = [set([START_POS]) for _ in range(num_agents)]
        self.visit_counts = [{START_POS: 1} for _ in range(num_agents)]
        self.current_turn = 0  # Index of the agent whose turn it is
        self.gas_station_status = {pos: GAS_MAX for pos in GAS_TILES}
        self.step_count = 0
        self.max_steps = 200
        self.last_action = [-1 for _ in range(num_agents)]
        self.scores = [0 for _ in range(num_agents)]
        
        # For enhanced pathfinding and state evaluation
        self.path_distances = {}  # Will be set by the environment
        self.alternate_paths = [[] for _ in range(num_agents)]  # Store different path options

    def copy(self):
        """Create a deep copy of the state"""
        new_state = MultiAgentState(self.num_agents)
        new_state.positions = self.positions.copy()
        new_state.prev_positions = self.prev_positions.copy()
        new_state.gas_levels = self.gas_levels.copy()
        new_state.done = self.done.copy()
        new_state.visited = [s.copy() for s in self.visited]
        new_state.visit_counts = [{k: v for k, v in vc.items()} for vc in self.visit_counts]
        new_state.current_turn = self.current_turn
        new_state.gas_station_status = self.gas_station_status.copy()
        new_state.step_count = self.step_count
        new_state.max_steps = self.max_steps
        new_state.last_action = self.last_action.copy()
        new_state.scores = self.scores.copy()
        new_state.path_distances = self.path_distances
        new_state.alternate_paths = [p.copy() if p else [] for p in self.alternate_paths]
        return new_state
    
    def get_positions_as_set(self, exclude_agent=None):
        """Get the set of positions occupied by agents (optionally excluding one)"""
        if exclude_agent is None:
            return set(self.positions)
        else:
            return {pos for i, pos in enumerate(self.positions) if i != exclude_agent}
    
    def is_game_over(self):
        """Check if the game is over"""
        return all(self.done) or self.step_count >= self.max_steps
    
    def get_winner(self):
        """Return the index of the winning agent or -1 for a tie"""
        goal_reached = [i for i, pos in enumerate(self.positions) if pos == GOAL_POS]
        if len(goal_reached) == 1:
            return goal_reached[0]  # Clear winner
        elif len(goal_reached) > 1:
            # Tiebreaker: who got there with more gas
            best_gas = -1
            best_agent = -1
            for i in goal_reached:
                if self.gas_levels[i] > best_gas:
                    best_gas = self.gas_levels[i]
                    best_agent = i
            return best_agent
        
        # If no one reached the goal, check who made the most progress
        if self.step_count >= self.max_steps:
            best_progress = -float('inf')
            best_agent = -1
            for i in range(self.num_agents):
                # Score is negative distance to goal plus gas level as a tiebreaker
                progress = -manhattan_distance(self.positions[i], GOAL_POS) + self.gas_levels[i] * 0.01
                if progress > best_progress:
                    best_progress = progress
                    best_agent = i
            return best_agent
            
        return -1  # No winner yet
    
    def get_next_turn(self):
        """Get the index of the next agent whose turn it is"""
        # Skip done agents
        next_turn = (self.current_turn + 1) % self.num_agents
        while self.done[next_turn] and not all(self.done):
            next_turn = (next_turn + 1) % self.num_agents
        return next_turn
    
    def update_visit_count(self, agent_idx, position):
        """Update the visit count for a position"""
        if position in self.visit_counts[agent_idx]:
            self.visit_counts[agent_idx][position] += 1
        else:
            self.visit_counts[agent_idx][position] = 1
        
        # Also add to visited set if not already there
        if position not in self.visited[agent_idx]:
            self.visited[agent_idx].add(position)

class MultiAgentEnvironment:
    """A multi-agent environment for the race track problem"""
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.gas_tiles = GAS_TILES
        self.goal_pos = GOAL_POS
        self.path_tiles = PATH_TILES
        self.start_pos = START_POS
        self.gas_max = GAS_MAX
        self.width = self.grid_size[1] * self.cell_size
        self.height = self.grid_size[0] * self.cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Multi-Agent Race Track")
        self.clock = pygame.time.Clock()
        
        # Precompute path distances for more efficient evaluation
        self.path_distances = self.compute_path_distances()
        
        # Precompute all path tiles in each vertical and horizontal row for alternate path finding
        self.path_grid = self.compute_path_grid()
        
        # Initialize state
        self.state = MultiAgentState(num_agents)
        self.state.path_distances = self.path_distances
        
        # Find an initial path for each agent
        for i in range(num_agents):
            occupied = self.state.get_positions_as_set(exclude_agent=i)
            path = find_path(self.start_pos, self.goal_pos, self.path_tiles, occupied)
            if path:
                self.state.alternate_paths[i] = path
        
        # Load sprites with different colors
        self.car_sprites = []
        for i in range(num_agents):
            # Create a copy of the car sprite with a different color
            colored_sprite = CAR_SPRITE.copy()
            # Apply color filter
            colored_surface = pygame.Surface(colored_sprite.get_size(), pygame.SRCALPHA)
            colored_surface.fill(PLAYER_COLORS[i % len(PLAYER_COLORS)])
            colored_sprite.blit(colored_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.car_sprites.append(colored_sprite)
    
    def compute_path_distances(self):
        """
        Precompute shortest path distances from every path tile to the goal
        using breadth-first search
        """
        distances = {}
        queue = [(GOAL_POS, 0)]  # (position, distance)
        visited = {GOAL_POS}
        
        while queue:
            pos, dist = queue.pop(0)
            distances[pos] = dist
            
            # Explore neighbors
            x, y = pos
            for dx, dy in ACTIONS:
                neighbor = (x + dx, y + dy)
                if neighbor in self.path_tiles and neighbor not in visited:
                    queue.append((neighbor, dist + 1))
                    visited.add(neighbor)
        
        return distances
    
    def compute_path_grid(self):
        """Create a grid representation of path tiles for faster lookup"""
        grid = {}
        for x in range(self.grid_size[0]):
            grid[x] = {}
            for y in range(self.grid_size[1]):
                grid[x][y] = (x, y) in self.path_tiles
        return grid
    
    def get_possible_actions(self, pos):
        """Get possible actions from a position, considering path tiles only"""
        actions = []
        x, y = pos
        for i, (dx, dy) in enumerate(ACTIONS):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                if (new_x, new_y) in self.path_tiles:
                    actions.append(i)
        return actions
    
    def check_move_validity(self, agent_idx, action):
        """Check if a move is valid, considering other agents' positions"""
        if self.state.done[agent_idx] or self.state.gas_levels[agent_idx] <= 0:
            return False, None
        
        pos = self.state.positions[agent_idx]
        x, y = pos
        dx, dy = ACTIONS[action]
        new_x, new_y = x + dx, y + dy
        
        # Check bounds
        if not (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]):
            return False, None
        
        new_pos = (new_x, new_y)
        
        # Check if the new position is a valid path tile
        if new_pos not in self.path_tiles:
            return False, None
        
        # Check if the new position is occupied by another agent
        for i, other_pos in enumerate(self.state.positions):
            if i != agent_idx and other_pos == new_pos:
                return False, None
        
        return True, new_pos
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.state = MultiAgentState(self.num_agents)
        self.state.path_distances = self.path_distances
        
        # Find an initial path for each agent
        for i in range(self.num_agents):
            occupied = self.state.get_positions_as_set(exclude_agent=i)
            path = find_path(self.start_pos, self.goal_pos, self.path_tiles, occupied)
            if path:
                self.state.alternate_paths[i] = path
                
        return self.state
    
    def step(self, action):
        """Execute one step in the environment"""
        agent_idx = self.state.current_turn
        
        # Store the action for debugging
        self.state.last_action[agent_idx] = action
        
        if self.state.done[agent_idx] or self.state.is_game_over():
            # Skip turn if agent is already done
            self.state.current_turn = self.state.get_next_turn()
            self.state.step_count += 1
            return self.state, 0.0, self.state.is_game_over()
        
        # Check if agent has gas
        if self.state.gas_levels[agent_idx] <= 0:
            self.state.done[agent_idx] = True
            self.state.current_turn = self.state.get_next_turn()
            self.state.step_count += 1
            return self.state, -50.0, self.state.is_game_over()
        
        # Check move validity
        is_valid, new_pos = self.check_move_validity(agent_idx, action)
        reward = 0.0
        
        if not is_valid:
            # Invalid move, stay in place and get a penalty
            print(f"Invalid move! Agent {agent_idx+1} tried {ACTION_NAMES[action]}")
            reward = -5.0
            
            # Try to find a new path since the current one might be blocked
            current_pos = self.state.positions[agent_idx]
            occupied = self.state.get_positions_as_set(exclude_agent=agent_idx)
            path = find_path(current_pos, self.goal_pos, self.path_tiles, occupied)
            if path:
                self.state.alternate_paths[agent_idx] = path
                print(f"Found new path for agent {agent_idx+1}")
        else:
            # Update position
            old_pos = self.state.positions[agent_idx]
            self.state.prev_positions[agent_idx] = old_pos
            self.state.positions[agent_idx] = new_pos
            print(f"Agent {agent_idx+1} moved from {old_pos} to {new_pos}")
            
            # Update visit count for this position
            self.state.update_visit_count(agent_idx, new_pos)
            
            # Consume gas
            self.state.gas_levels[agent_idx] -= 1
            
            # Check for gas station
            if new_pos in self.gas_tiles and self.state.gas_station_status[new_pos] > 0:
                # Refill gas and reduce gas station capacity
                refill_amount = min(self.gas_max - self.state.gas_levels[agent_idx], 
                                   self.state.gas_station_status[new_pos])
                self.state.gas_levels[agent_idx] += refill_amount
                self.state.gas_station_status[new_pos] -= refill_amount
                reward += 2.0  # Small reward for finding gas
                print(f"Agent {agent_idx+1} refilled gas at {new_pos}, new level: {self.state.gas_levels[agent_idx]}")
            
            # Penalty if revisiting
            if self.state.visit_counts[agent_idx][new_pos] > 1:
                # Higher penalty for frequently visited positions
                visit_count = self.state.visit_counts[agent_idx][new_pos]
                penalty = 1.0 * visit_count
                reward -= penalty
                print(f"Agent {agent_idx+1} revisited position {new_pos} ({visit_count} times), penalty: {penalty}")
            
            # Check if goal reached
            if new_pos == self.goal_pos:
                reward += 100.0  # Big reward for reaching goal
                self.state.done[agent_idx] = True
                print(f"Agent {agent_idx+1} reached the goal!")
            else:
                # Calculate progress reward using path distance if available
                old_dist = self.path_distances.get(old_pos, 
                                                  manhattan_distance(old_pos, self.goal_pos))
                new_dist = self.path_distances.get(new_pos, 
                                                 manhattan_distance(new_pos, self.goal_pos))
                
                # Significant reward for getting closer to the goal using true path distance
                progress = old_dist - new_dist
                progress_reward = 2.0 * progress
                reward += progress_reward
                print(f"Agent {agent_idx+1} distance change: {progress:+d} ({old_dist} -> {new_dist}), reward: {progress_reward}")
                
                # If we made negative progress, try to find a new path
                if progress < 0:
                    occupied = self.state.get_positions_as_set(exclude_agent=agent_idx)
                    path = find_path(new_pos, self.goal_pos, self.path_tiles, occupied)
                    if path:
                        self.state.alternate_paths[agent_idx] = path
                        print(f"Found new path for agent {agent_idx+1} after negative progress")
            
            # Update score for this agent
            self.state.scores[agent_idx] += reward
        
        # Update turn
        old_turn = self.state.current_turn
        self.state.current_turn = self.state.get_next_turn()
        print(f"Turn changed from Player {old_turn+1} to Player {self.state.current_turn+1}")
        
        # Update step count
        self.state.step_count += 1
        
        # Check for game over
        done = self.state.is_game_over()
        
        return self.state, reward, done
    
    def render(self):
        """Render the current state of the environment"""
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
        
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw the background
        rect = pygame.Rect(0, 0, self.width, self.height)
        scaled_background = pygame.transform.scale(MAP_SPRITE, (self.width, self.height))
        self.screen.blit(scaled_background, rect)
        
        # Draw path tiles (lightly shaded)
        for pos in self.path_tiles:
            x, y = pos
            path_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (200, 200, 200, 50), path_rect, 1)
        
        # Optionally draw the precomputed paths for each agent
        for i in range(self.num_agents):
            if not self.state.done[i] and self.state.alternate_paths[i]:
                color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
                alpha_color = (*color[:3], 80)  # Add transparency
                for pos in self.state.alternate_paths[i]:
                    x, y = pos
                    path_rect = pygame.Rect(x * self.cell_size + 10, y * self.cell_size + 10,
                                           self.cell_size - 20, self.cell_size - 20)
                    pygame.draw.rect(self.screen, alpha_color, path_rect, 1)
        
        # Draw the goal
        x, y = self.goal_pos
        trophy_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                 self.cell_size, self.cell_size)
        scaled_trophy = pygame.transform.scale(TROPHY_SPRITE, (self.cell_size, self.cell_size))
        self.screen.blit(scaled_trophy, trophy_rect)
        
        # Draw gas stations with remaining capacity indicator
        for gas_pos, capacity in self.state.gas_station_status.items():
            x, y = gas_pos
            if capacity > 0:
                # Draw gas indicator
                gas_rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5,
                                      self.cell_size - 10, self.cell_size - 10)
                capacity_pct = capacity / self.gas_max
                color_intensity = int(255 * capacity_pct)
                pygame.draw.rect(self.screen, (0, color_intensity, 0), gas_rect)
        
        # Draw the agents
        for i in range(self.num_agents):
            if self.state.done[i]:
                continue  # Skip rendering done agents
                
            x, y = self.state.positions[i]
            agent_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                    self.cell_size, self.cell_size)
            
            # Transform sprite based on movement direction
            transformed_agent = pygame.transform.scale(self.car_sprites[i], 
                                                     (self.cell_size, self.cell_size))
            
            # Rotate based on previous position
            prev_x, prev_y = self.state.prev_positions[i]
            if prev_x < x:
                transformed_agent = pygame.transform.rotate(transformed_agent, 90)
            elif prev_x > x:
                transformed_agent = pygame.transform.rotate(transformed_agent, -90)
            elif prev_y < y:
                transformed_agent = pygame.transform.rotate(transformed_agent, 180)
            
            self.screen.blit(transformed_agent, agent_rect)
        
        # Draw turn indicator and gas levels
        font = pygame.font.Font(None, 24)
        
        # Draw current turn indicator
        turn_text = font.render(f"Turn: Player {self.state.current_turn + 1}", True, 
                               PLAYER_COLORS[self.state.current_turn % len(PLAYER_COLORS)])
        self.screen.blit(turn_text, (10, 10))
        
        # Draw gas levels, scores, and positions for each agent
        for i in range(self.num_agents):
            gas_text = font.render(f"P{i+1} Gas: {self.state.gas_levels[i]}", True, 
                                  PLAYER_COLORS[i % len(PLAYER_COLORS)])
            self.screen.blit(gas_text, (10, 30 + i * 40))
            
            # Draw score
            score_text = font.render(f"Score: {self.state.scores[i]:.1f}", True,
                                    PLAYER_COLORS[i % len(PLAYER_COLORS)])
            self.screen.blit(score_text, (10, 50 + i * 40))
            
            # Draw position
            pos_text = font.render(f"Pos: {self.state.positions[i]}", True,
                                  PLAYER_COLORS[i % len(PLAYER_COLORS)])
            self.screen.blit(pos_text, (100, 30 + i * 40))
            
            # Draw last action
            if self.state.last_action[i] != -1:
                action_text = font.render(f"Action: {ACTION_NAMES[self.state.last_action[i]]}", True, 
                                        PLAYER_COLORS[i % len(PLAYER_COLORS)])
                self.screen.blit(action_text, (200, 30 + i * 40))
        
        # Draw step counter
        step_text = font.render(f"Step: {self.state.step_count}/{self.state.max_steps}", True, (255, 255, 255))
        self.screen.blit(step_text, (self.width - 120, 10))
        
        pygame.display.flip()
        self.clock.tick(15)  # Limit to 15 FPS
    
    def close(self):
        """Clean up resources"""
        pygame.quit()

# Used for evaluating states in MCTS
def evaluate_state(state, agent_idx, path_distances=None):
    """
    Evaluate a state from the perspective of agent_idx
    Returns higher score for states favorable to that agent
    """
    # If game is over, check winner
    if state.is_game_over():
        winner = state.get_winner()
        if winner == agent_idx:
            return 1000  # Big positive score if this agent won
        elif winner == -1:
            return 0  # Neutral score for tie
        else:
            return -1000  # Big negative score if opponent won
    
    # Basic evaluation with better weights and priorities
    score = 0
    
    # Get my position and distance to goal
    my_pos = state.positions[agent_idx]
    
    # Use path distance if available, otherwise Manhattan distance
    if path_distances and my_pos in path_distances:
        my_dist = path_distances[my_pos]
    else:
        my_dist = manhattan_distance(my_pos, GOAL_POS)
    
    # Distance to goal is the MOST important factor
    score -= my_dist * 20  # Very high weight for distance to goal
    
    # Gas level (higher is better) but only to the extent needed
    needed_gas = min(my_dist + 5, GAS_MAX)  # Only value gas that's needed
    gas_value = min(state.gas_levels[agent_idx], needed_gas)
    score += gas_value * 3
    
    # Critical gas warning - make this a priority if we're low on gas
    if state.gas_levels[agent_idx] < my_dist / 2:
        # We need to find gas urgently
        min_gas_dist = float('inf')
        for gas_pos, capacity in state.gas_station_status.items():
            if capacity > 0:
                gas_dist = manhattan_distance(my_pos, gas_pos)
                min_gas_dist = min(min_gas_dist, gas_dist)
        
        if min_gas_dist < float('inf'):
            score -= min_gas_dist * 10  # High priority to reach gas
    
    # Heavy penalty for revisiting positions (prevents loops)
    if my_pos in state.visit_counts[agent_idx]:
        visit_count = state.visit_counts[agent_idx][my_pos]
        if visit_count > 1:  # Only penalize if visited more than once
            score -= (visit_count * visit_count) * 10  # Quadratic penalty
    
    return score