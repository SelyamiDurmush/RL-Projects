import gym
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42) # Set random seed for reproducibility

env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)  # "ansi" for text output
#env = gym.make("FrozenLake-v1", render_mode="human,", is_slippery=False) # # "human" for visual rendering, 
env.reset(seed=42)  # Initialize the environment

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n)) # (16 states, 4 actions) 

""" 

0: left, 1: down, 2: right, 3: up, 
states are numbered from 0 to 15 in a 4x4 grid, where 0 is the top-left corner and 15 is the bottom-right corner.

"""
# Hyperparameters
alpha = 0.3   # Learning rate
gamma = 0.95   # Discount factor
epsilon = 0.25 # Exploration rate, 0.1 = 10% exploration, 90% exploitation


# Plotting matrices
reward_list = [] # Having the reward of 1 for reaching the goal and 0 otherwise
episode_num =  50001

for episode in range(1, episode_num):

  # Initialize environment
  state, _ = env.reset()
  done = False

  reward_count = 0
  time_steps = 0

  while True:

    time_steps += 1
    # explore vs exploit
    if random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()  # Explore: choose a random action
    else:
      action = np.argmax(q_table[state])  # Exploit: choose the best action from Q-table

    # action process and take reward/observation
    next_state, reward, terminated, truncated, _ = env.step(action) 
    #print(f"action: {action},next_state: {next_state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, _: {_}")
    done = terminated or truncated

    # Q learning function
    next_value = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

    # update Q-table
    q_table[state, action] = next_value

    # update state
    state = next_state

    reward_count += reward

    if done:
      print(f"Episode {episode} finished after {time_steps} time steps with reward {reward_count}")
      break
    
  if episode % 100 == 0:
    reward_list.append(reward_count)
    print(f"Episode {episode}: Reward: {reward_count}, action: {action}, time_steps: {time_steps}")
    
# env.render()
# env.close()

# the best action for each state as a 4x4 grid
policy = np.argmax(q_table, axis=1)
print("Extract learned policy: ")
print(policy.reshape(4, 4))

fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(reward_list)
axs.set_title("Reward over Time")
axs.set_xlabel("Episode (x100)")
axs.set_ylabel("Reward")
plt.show()