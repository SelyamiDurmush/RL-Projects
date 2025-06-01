# %% Test
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ["OMP_NUM_THREADS"] = "12"  # Match logical/physical cores
os.environ["KMP_BLOCKTIME"] = "1"     # Reduce idle wait time
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0" #
import gym
import numpy as np
if not hasattr(np, "bool8"):  # Fix for compatibility with older numpy versions
    np.bool8 = np.bool_
import tensorflow as tf
from keras.utils import Progbar
from tensorflow.keras.models import load_model
import keras.utils
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import random

# Set number of CPU threads used by TensorFlow (helps on local machines with multi-core CPUs)
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Disable training progress bar (Keras outputs) for performance/clean output
class NoOpProgbar:
    def __init__(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass
    def set_description(self, *args, **kwargs): pass

keras.utils.Progbar = NoOpProgbar

class DQNAgent:

    def __init__(self, env):
        # Environment info
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay per episode
        self.epsilon_min = 0.01  # Minimum exploration
        self.memory = deque(maxlen = 2000)  # Experience replay buffer
        self.model = self.build_model()  # Online model
        self.target_model = self.build_model()  # Target model (for Double DQN)

    def build_model(self):
        # Simple 2-layer dense neural network
        model = Sequential()
        model.add(Input(shape=(self.state_size,))) # Input layer
        model.add(Dense(64, activation='tanh'))  # Tanh activation works well for CartPole
        model.add(Dense(self.action_size, activation='linear'))  # Output Q-values for each action
        model.compile(loss='mse', optimizer = Adam(learning_rate = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store transition (s, a, r, s', done)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # ε-greedy policy: explore or exploit
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()  # Explore: random action
        else:
          q_values = self.model.predict(state, verbose=0)
          return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = np.array(random.sample(self.memory, batch_size), dtype=object)

        # Unpack minibatch
        states = np.vstack(minibatch[:, 0])
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2].astype(float)
        next_states = np.vstack(minibatch[:, 3])
        dones = minibatch[:, 4].astype(bool)

        # Predict Q-values for current and next states
        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        # Double DQN
        best_actions = np.argmax(q_next, axis=1)
        target_q_values = rewards.copy()
        target_q_values[~dones] += self.gamma * q_next_target[~dones, best_actions[~dones]]

        q_values[np.arange(batch_size), actions] = target_q_values

        # Batch training
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def adaptiveEGreedy(self):
        # Decay ε after each episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def targetModelUpdate(self):
        # Copy online model weights to target model
        self.target_model.set_weights(self.model.get_weights())

if __name__ == "__main__":
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    env = gym.make("CartPole-v1", render_mode="human")
    agent = DQNAgent(env)

    batch_size = 64
    episode_num = 100

    for episode in range(1, episode_num + 1):
        
        state, _ = env.reset(seed = seed) # set seed for reproducibility
        state = np.reshape(state, [1, agent.state_size]) # Reshape state for the model input
        time_step = 0
        
        while True:
          
          action = agent.act(state) # Choose action based on ε-greedy policy

          next_state, reward, terminated, truncated, _ = env.step(action)
          done = terminated or truncated
          next_state = np.reshape(next_state, [1, agent.state_size]) # Reshape next_state for the model input

          # Store transition
          agent.remember(state, action, reward, next_state, done)
          # Update state
          state = next_state
          # Train the agent
          agent.replay(batch_size)
          
          # Update ε-greedy policy
          # agent.adaptiveEGreedy()
          
          time_step += 1

          if done:
            print(f"Episode: {episode}, time step {time_step}")
            agent.adaptiveEGreedy()  # ε-decay once per episode

            if episode % 5 == 0:
                agent.targetModelUpdate()  # Periodic target network update

            if time_step >= 500:
              agent.model.save("dqn_cartpole_solved.h5")
              print("✅ Environment solved! Model saved as dqn_cartpole_solved.h5")
            
            break


# %% # Test the trained model 

model = load_model("dqn_cartpole_solved.h5", compile=False) 
env = gym.make("CartPole-v1", render_mode="human")

# Initialize environment
state, _ = env.reset()
state = np.reshape(state, [1, 4])  # CartPole state has size 4
time_t = 0

while True:
    env.render()
    
    # Predict the action using the trained model
    q_values = model.predict(state, verbose=0)
    action = np.argmax(q_values[0])
    
    # Take the action
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 4])
    state = next_state
    time_t += 1

    if terminated or truncated:
        break

env.close()
print(f"Test finished after {time_t} time steps.")