import time
import timeit
import pygame
from environment import ComplicatedRaceTrackEnvPygame

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

    # Measure time taken for pathfinding
    print("Measuring time taken for 100 runs...")
    res = timeit.timeit("find_dfs_path(env)", globals=globals(), number=100)
    print(f"Time taken for 100 runs: {res:.3f} seconds")

    print("Finding path using DFS...")
    start = time.time()
    path = find_dfs_path(env)
    end = time.time()
    print(f"Path found. Time taken: {end - start:.3f} seconds")

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
    
        
