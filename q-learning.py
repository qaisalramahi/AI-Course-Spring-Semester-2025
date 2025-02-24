import numpy as np
import random
import time
import timeit
import pygame
from environment import ComplicatedRaceTrackEnvPygame

def find_qlearning_path(env):
    # hyperparameters
    num_episodes = 1500
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_rate = 0.99
    
    # Exploration parameters
    epsilon = 1.0          
    max_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.005

    # Initialize the Q-table with keys ((row, col), action).
    q_table = {}
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            for action in range(env.action_space.n):
                q_table[((i, j), action)] = 0.0
    
    rewards_all_episodes = []
    
    # Begin training
    
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
        
    return q_table

def test_training(env, q_table):
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
        pygame.time.wait(500)
        print(f"Action: {action}, Position: {state_pos}, Reward: {reward}")


# Q-Learning RL algorithm integrated within pygame environment
if __name__ == "__main__":
    env = ComplicatedRaceTrackEnvPygame(grid_size=(10, 10), cell_size=50)


    ### This takes a long time (about 60s) to run for q-learning, showcasing how slow it is compared to BFS and DFS
    ### Comment it out if you don't want to wait for it to finish
    # # Measure time taken for pathfinding
    # print("Measuring time taken for 100 runs...")
    # res = timeit.timeit("find_qlearning_path(env)", globals=globals(), number=100)
    # print(f"Time taken for 100 runs: {res:.3f} seconds")
    
    print("Training started...")
    start = time.time()
    q_table = find_qlearning_path(env)
    end = time.time()
    print(f"Training finished. Training time: {end - start:.3f} seconds")
    
    # Test the learned policy.
    test_training(env, q_table)
    
    # Final render and cleanup.
    env.render()
    pygame.time.wait(2000)
    env.close()

