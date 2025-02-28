import time
import pygame
import heapq
from environment import ComplicatedRaceTrackEnvPygame, START_POS, GOAL_POS, GAS_MAX, GAS_TILES

def heuristic(position, goal, gas_level):
    """
    Gas-aware Manhattan distance heuristic.
    Considers both distance to goal and gas management.
    """
    # Base Manhattan distance
    manhattan_dist = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
    
    # If we have enough gas to reach the goal, just use Manhattan distance
    if gas_level >= manhattan_dist:
        return manhattan_dist
    
    # If gas is low, consider distance to nearest gas station
    nearest_gas_dist = float('inf')
    for gas_tile in GAS_TILES:
        dist_to_gas = abs(position[0] - gas_tile[0]) + abs(position[1] - gas_tile[1])
        nearest_gas_dist = min(nearest_gas_dist, dist_to_gas)
    
    # Return a weighted combination to guide toward gas stations when needed
    # The weight is less than 1 to keep the heuristic admissible
    return manhattan_dist + (nearest_gas_dist * 0.5)

def find_astar_path(env: ComplicatedRaceTrackEnvPygame):
    """
    Find path using A* algorithm with a gas-aware heuristic.
    """
    start = START_POS
    goal = GOAL_POS
    
    # Priority queue: (f_score, entry_count, current_pos, path_actions, gas)
    open_set = [(heuristic(start, goal, GAS_MAX), 0, start, [], GAS_MAX)]
    entry_count = 1  # For tie-breaking
    
    # Track best g_score for each (position, gas) state
    best_g_scores = {(start, GAS_MAX): 0}
    
    while open_set:
        # Get node with lowest f_score
        _, _, current_pos, path_actions, current_gas = heapq.heappop(open_set)
        
        # Check if we've reached the goal
        if current_pos == goal:
            print(f"Goal reached with actions {path_actions}")
            return path_actions
        
        # Verify that the current path remains valid
        valid, gas, pos = env.check(path_actions)
        if not valid:
            continue
        
        # Current g_score (path length so far)
        current_g = len(path_actions)
        
        # Skip if we already found a better path to this state
        current_state = (current_pos, current_gas)
        if current_state in best_g_scores and best_g_scores[current_state] < current_g:
            continue
        
        # Explore possible actions from the current position
        possible_actions = env.get_possible_actions(current_pos)
        for action in possible_actions:
            # Check if new path is valid
            new_valid, new_gas, new_pos = env.check(path_actions + [action])
            if not new_valid:
                continue
                
            # Calculate new g_score
            new_g = current_g + 1
            
            # Create new state
            new_state = (new_pos, new_gas)
            
            # Skip if we already found a better path to this state
            if new_state in best_g_scores and best_g_scores[new_state] <= new_g:
                continue
            
            # This is a better path, so update best_g_scores
            best_g_scores[new_state] = new_g
            
            # Calculate f_score = g_score + h_score
            h_score = heuristic(new_pos, goal, new_gas)
            f_score = new_g + h_score
            
            # Add promising node to open set to find the best path by f_score
            heapq.heappush(open_set, (f_score, entry_count, new_pos, path_actions + [action], new_gas))
            entry_count += 1
    
    return None  # No path found

if __name__ == "__main__":
    env = ComplicatedRaceTrackEnvPygame()

    print("Finding path using A*...")
    start_time = time.time()
    path = find_astar_path(env)
    end_time = time.time()
    print(f"Path found. Time taken: {end_time - start_time:.3f} seconds")

    if path is None:
        print("No path found!")
    else:
        print(f"Path found with {len(path)} steps. Executing...")
        state, _ = env.reset()
        done = False
        for action in path:
            if done:
                break
            env.render()
            state, reward, done, _, _ = env.step(action)
            pygame.time.wait(500)  # Half-second delay to visualize steps

        # Final render and close
        env.render()
        pygame.time.wait(2000)  # Wait to show final state
        env.close()