# Root Project README.md
# RL Projects

This repository contains implementations of Reinforcement Learning algorithms on classic OpenAI Gym environments.

## Projects

- [CartPole DQN](./CartPole-DQN)
- [FrozenLake Q-learning](./FrozenLake-Qlearning)
- [Taxi Q-learning](./Taxi-Qlearning)

## Setup
```bash
pip install -r requirements.txt
```

---

# /CartPole-DQN/README.md
## CartPole - Deep Q-Learning

This is a DQN agent to solve the CartPole-v1 environment using Keras and TensorFlow.

### Run
```bash
python dqn_cartpole.py
```

- Model is saved to `dqn_cartpole_solved.h5` when solved.

---

# /FrozenLake-Qlearning/README.md
## FrozenLake - Q-learning

A simple Q-learning agent to solve FrozenLake-v1 (non-slippery).

### Run
```bash
python frozenlake_qlearning.py
```

- Final policy is printed as a 4x4 grid.
- Reward progression is plotted.

---

# /Taxi-Qlearning/README.md
## Taxi-v3 - Q-learning

A Q-learning agent solving the Taxi-v3 environment.

### Run
```bash
python taxi_qlearning.py
```

- Reward and dropoff stats printed every episode.

---

# requirements.txt (already exists)
gym==0.26.2
keras==3.10.0
matplotlib==3.10.3
numpy==1.23.5
tensorflow==2.19.0