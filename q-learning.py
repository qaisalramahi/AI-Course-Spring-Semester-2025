import numpy as np
import random
import time
import pygame
from environment import ComplicatedRaceTrackEnvPygame, START_POS, GOAL_POS, GAS_MAX

ACTIONS = ["Left", "Right", "Up", "Down"]

def find_qlearning_path(env: ComplicatedRaceTrackEnvPygame):
    # hyperparameters
    num_episodes = 5000
    max_steps_per_episode = 300
    learning_rate = 0.1
    discount_rate = 0.99
    
    # Exploration parameters (increased decay for less random oscillation)
    epsilon = 1.0          
    max_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001  # Increased decay rate
    
    # Initialize the Q-table with keys ((row, col, gas_level), action).
    q_table = {}
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            for gas in range(GAS_MAX + 1):  # Include gas level in state
                for action in range(env.action_space.n):
                    q_table[((i, j, gas), action)] = 0.0
    
    rewards_all_episodes = []
    success_count = 0
    
    # Begin training
    for episode in range(num_episodes):
        state, _ = env.reset()
        state_pos = START_POS
        gas_level = GAS_MAX
        done = False
        rewards_current_episode = 0
        
        # Compute initial Manhattan distance to goal.
        prev_distance = abs(state_pos[0] - GOAL_POS[0]) + abs(state_pos[1] - GOAL_POS[1])
        
        for step in range(max_steps_per_episode):
            # Get valid actions for current position
            valid_actions = env.get_possible_actions(state_pos)
            if not valid_actions:
                break
            
            # Choose action using epsilon-greedy policy.
            if random.uniform(0, 1) < epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = [q_table[(state_pos + (gas_level,), a)] for a in valid_actions]
                action = valid_actions[int(np.argmax(q_values))]
            
            # print(f"Taking action {ACTIONS[action]} at position {state_pos} with gas {gas_level}")
            new_state, reward, done, _, _ = env.step(action)
            new_state_pos = env.agent_pos
            new_gas_level = env.gas
            
            # Compute the Manhattan distance to the goal after moving.
            new_distance = abs(new_state_pos[0] - GOAL_POS[0]) + abs(new_state_pos[1] - GOAL_POS[1])
            # Shaping reward: positive if the agent moves closer, negative otherwise.
            shaping_reward = 0.2 * (prev_distance - new_distance)
            reward += shaping_reward
            prev_distance = new_distance
            
            # Q-learning update
            old_value = q_table[(state_pos + (gas_level,), action)]
            # Only consider valid actions from the new state.
            valid_next_actions = env.get_possible_actions(new_state_pos)
            if valid_next_actions:
                next_max = max([q_table[(new_state_pos + (new_gas_level,), a)] for a in valid_next_actions])
            else:
                next_max = 0.0
            new_value = old_value + learning_rate * (reward + discount_rate * next_max - old_value)
            q_table[(state_pos + (gas_level,), action)] = new_value
            
            state_pos = new_state_pos
            gas_level = new_gas_level
            rewards_current_episode += reward
            
            # print(f"New state: {new_state_pos}, Gas: {new_gas_level}, Reward: {reward:.2f}. Total reward: {rewards_current_episode:.2f}")
            
            if done:
                if reward > 0:  # Successfully reached the goal
                    success_count += 1
                break
            
            # time.sleep(0.5)  # Slow down for visualization
        
        # Decay epsilon after each episode.
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        rewards_all_episodes.append(rewards_current_episode)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_all_episodes[-100:]) if episode >= 100 else np.mean(rewards_all_episodes)
            print(f"Episode {episode}: Avg reward = {avg_reward:.2f}, Success rate = {success_count/(episode+1):.2%}")
            print(f"Position: {state_pos}, Gas: {gas_level}, Action: {action}, Reward: {reward:.2f}")
    
    print(f"Final success rate: {success_count/num_episodes:.2%}")
    return q_table


def test_training(env, q_table):
    state, _ = env.reset()
    state_pos = START_POS
    gas_level = GAS_MAX
    done = False
    print("Trained Q-Table Policy Execution:")
    steps = 0
    total_reward = 0
    
    # For visualization, track the path
    path = [state_pos]
    
    while not done and steps < 100:  # Add step limit to prevent infinite loops
        env.render()
        
        # Get valid actions and their Q-values
        x, y = state_pos
        valid_actions = env.get_possible_actions(state_pos)
        
        # Get Q-values for all actions, prioritizing valid ones
        q_values = [q_table[(state_pos + (gas_level,), a)] 
                   if a in valid_actions else -float('inf') 
                   for a in range(env.action_space.n)]
        action = int(np.argmax(q_values))
        
        # Show the action being taken
        print(f"Step {steps}: Position {state_pos}, Gas: {gas_level}, Taking action: {ACTIONS[action]}")
        
        # Take the action
        state, reward, done, _, _ = env.step(action)
        state_pos = env.agent_pos
        gas_level = env.gas
        total_reward += reward
        steps += 1
        path.append(state_pos)
        
        pygame.time.wait(500)  # Slow down visualization
    
    # Show results
    print(f"Test completed: Steps taken = {steps}, Total reward = {total_reward:.2f}")
    if done and total_reward > 0:
        print("Successfully reached the goal!")
    elif env.gas <= 0:
        print("Failed: Ran out of gas!")
    else:
        print("Failed: Didn't reach the goal within step limit")
    
    print("Path taken:", path)
    return steps, total_reward

# Q-Learning RL algorithm integrated within pygame environment
if __name__ == "__main__":
    env = ComplicatedRaceTrackEnvPygame()
    
    print("Training started...")
    start = time.time()
    q_table = find_qlearning_path(env)
    end = time.time()
    print(f"Training finished. Training time: {end - start:.3f} seconds")
    
    # Test the learned policy
    print("Testing trained policy...")
    steps, reward = test_training(env, q_table)
    
    # Final render and cleanup
    env.render()
    pygame.time.wait(2000)
    env.close()