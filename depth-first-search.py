import time
import pygame
from environment import ComplicatedRaceTrackEnvPygame, START_POS, GOAL_POS, GAS_MAX

def find_dfs_path(env: ComplicatedRaceTrackEnvPygame):
    start = START_POS
    goal = GOAL_POS
    # Using a list as a stack: each element is (current_pos, path_actions, visited, gas)
    stack = [(start, [], set({start}), GAS_MAX)]
    
    while stack:
        current_pos, path_actions, visited, current_gas = stack.pop()  # DFS: pop from the end
        # Check if we've reached the goal
        if current_pos == goal:
            print(f"Goal reached with actions {path_actions}")
            return path_actions
        
        # Verify that the current path remains valid
        valid, gas, pos = env.check(path_actions)
        if not valid:
            continue

        # Explore possible actions from the current position
        possible_actions = env.get_possible_actions(current_pos)
        for action in possible_actions:
            new_valid, new_gas, new_pos = env.check(path_actions + [action])
            if not new_valid:
                continue

            if new_pos not in visited:
                new_visited = visited.copy()
                new_visited.add(new_pos)
                # Push the new state onto the stack
                stack.append((new_pos, path_actions + [action], new_visited, new_gas))
    return None


#  <in breadth_first_search.py>
def solve(env):
    path = find_dfs_path(env)
    return path


if __name__ == "__main__":
    env = ComplicatedRaceTrackEnvPygame()

    print("Finding path using DFS...")
    start_time = time.time()
    path = find_dfs_path(env)
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
