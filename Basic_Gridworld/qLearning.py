import numpy as np
from grid import Grid
import random

# Initiated the environment
env = Grid()


num_eps = 1000
T = 1000
discount_rate = 0.99
learning_rate = 0.1

# Parameters for epsilon-greedy policy improvement
epsilon = 1.0 # exploration rate
epsilon_decay_rate = 0.003

# Initializing Q Value for all state action
Q = np.zeros((len(env.stateSpace), len(env.actionSpace)))

for ep in range(num_eps):

    env.reset()
    state = env.currentState


    for t in range(T):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[env.currentState])
        else:
            action = random.choice(env.actionSpace)

        next_state, reward, done = env.step(action)

        # Update action-value function for current (s,a) pairs
        Q[state, action] += learning_rate * (reward + discount_rate * Q[next_state].max() - Q[state, action])

        state = next_state

        # Terminal state reached, end of episode
        if done:
            break

        epsilon = np.exp(-epsilon_decay_rate * ep)


print('Action-Value function for all states and actions:')
print(Q)


env.startGrid(Q)