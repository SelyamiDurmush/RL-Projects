import gym
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="ansi").env # Create the Taxi environment
env.reset()  # Initialize the environment 

""""
States: The state is represented by a tuple of 4 values:
    1. Taxi row (0-4)
    2. Taxi column (0-4)
    3. Passenger location (0-4, where 0 is the taxi's location and 1-4 are the locations of the passengers)
    4. Destination (0-3, where 0 is the taxi's location and 1-3 are the locations of the passengers)
    The total number of states is 5*5*5*4 = 500.

Actions: There are 6 discrete deterministic actions: 
0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff
Rewards: The reward is +20 for a successful dropoff, -10 for an illegal pickup or dropoff, and -1 for each time step taken.

# The taxi environment is a grid world where the taxi has to pick up and drop off passengers at different locations.
"""

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n)) # (500 states, 6 actions)
# 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate, 0.1 = 10% exploration, 90% exploitation


# Plotting matrices
reward_list = []
dropoff_list = []


episode_num =  2
for episode in range(1, episode_num):

  # Initialize environment
  state, _ = env.reset()
  done = False

  reward_count = 0
  dropoff_count = 0
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
    print(f"next_state: {next_state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, _: {_}")
    done = terminated or truncated

    # Q learning function
    next_value = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])


    # update Q-table
    q_table[state, action] = next_value

    # update state
    state = next_state

    # reward counting
    reward_count += reward

    # find the number of wrong dropoffs
    if action == 5 and reward == -10:  # Action 5 is Dropoff
      dropoff_count += 1
    
   # Program is terminated when the taxi reaches the destination or the passenger is dropped off at the wrong location.
    if done:
      break
    
    env.render()

  if episode % 1 == 0:
    reward_list.append(reward_count)
    dropoff_list.append(dropoff_count)
    print(f"Episode {episode}: Total Reward: {reward_count}, Wrong Dropoffs: {dropoff_count}, action: {action}, time_steps: {time_steps}")


# %% Plotting the results
# fig, axs = plt.subplots(nrows=2, figsize=(10, 6))
# axs[0].plot(reward_list)
# axs[0].set_xlabel('Episode')
# axs[0].set_ylabel('Reward')


# axs[1].plot(dropoff_list)
# axs[1].set_xlabel('Episode')
# axs[1].set_ylabel('Dropoff')

# axs[0].grid(True)
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()



