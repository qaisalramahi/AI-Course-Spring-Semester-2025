import time
import pygame
from environment import ComplicatedRaceTrackEnvPygame, START_POS, GOAL_POS, GAS_MAX
from collections import deque

def find_bfs_path(env: ComplicatedRaceTrackEnvPygame):
        start = START_POS
        goal = GOAL_POS

        queue = deque([(start, [], set({start}), GAS_MAX)]) # Is (current_pos, path_actions, visited, gas)
        
        while queue:
            current_pos, path_actions, visited, current_gas = queue.popleft()
            # print(f"Current pos: {current_pos}, Path: {path_actions}. Visited: {visited}. Gas: {current_gas}")

            if current_pos == goal:
                print(f"Goal reached with actions {path_actions}")
                return path_actions
            
            # If path is not valid, terminate branch
            valid, gas, pos = env.check(path_actions)
            assert(current_pos == pos, f"Current pos {current_pos} does not match checked pos {pos}")
            assert(current_gas == gas, f"Current gas {current_gas} does not match checked gas {gas}")
            if not valid:
                # print(f"Path {path_actions} is not valid, terminating branch (gas: {gas})")
                continue

            possible_actions = env.get_possible_actions(current_pos)
            # print(f"Possible actions: {possible_actions}")
            for action in possible_actions:
                new_valid, new_gas, new_pos = env.check(path_actions + [action])
                # print(f"Checking new pos {new_pos} with action {action} (gas: {new_gas})")
                if not new_valid:
                    # print(f"Path {path_actions + [action]} is not valid, skipping")
                    continue

                # print(f"New pos {new_pos} with action {action} is valid. Visited: {visited}")

                if new_pos not in visited:
                    new_visited = visited.copy()
                    new_visited.add(new_pos)
                    # print(f"New pos has not been visited. Adding: {new_visited}")
                    queue.append((new_pos, path_actions + [action], new_visited, new_gas))
        return None

if __name__ == "__main__":
    env = ComplicatedRaceTrackEnvPygame()

    print("Finding path using BFS...")
    start = time.time()
    path = find_bfs_path(env)
    end = time.time()
    print(f"Path found. Time taken: {end - start:.3f} seconds")

    if path is None:
        print("No path found!")
    else:
        print(f"Path found with {len(path)} steps. Executing...")
        # print(f"Path: {path}")
        state, _ = env.reset()
        done = False
        for action in path:
            if done:
                break
            # Render and step
            env.render()
            state, reward, done, _, _ = env.step(action)
            # print(f"Action: {action}, Position: {env.agent_pos}, Reward: {reward}")
            pygame.time.wait(500)  # Half-second delay to visualize steps

        # Final render and close
        env.render()
        pygame.time.wait(2000)  # Wait to show final state
        env.close()
    