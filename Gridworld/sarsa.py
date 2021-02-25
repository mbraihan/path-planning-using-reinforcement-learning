import numpy as np
import random
from grid import Grid

# Create environment
env = Grid()


num_eps = 1000
T = 1000
discount_rate = 0.99
learning_rate = 0.1
decay_rate = 0.8


epsilon = 1.0
epsilon_decay_rate = 0.003


Q = np.zeros((len(env.stateSpace), len(env.actionSpace))) # action-value function for all (s,a) pairs

for ep in range(num_eps):

    env.reset()
    state = env.currentState


    E = np.zeros((len(env.stateSpace), len(env.actionSpace)))


    if random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[env.currentState])
    else:
        action = random.choice(env.actionSpace)


    for t in range(T):
        next_state, reward, done = env.step(action)

        if random.uniform(0, 1) > epsilon:
            next_action = np.argmax(Q[env.currentState])
        else:
            next_action = random.choice(env.actionSpace)


        td_error = reward + discount_rate * Q[next_state, next_action] - Q[state, action]


        E[state, action] += 1

        # Update action-value function and eligibility trace for all (s,a) pairs
        for s in env.stateSpace:
            for a in env.actionSpace:
                Q[s, a] += learning_rate * td_error * E[s, a]
                E[s, a] = discount_rate * decay_rate * E[s, a]

        state = next_state
        action = next_action

        # Terminal state reached, end of episode
        if done:
            break

        epsilon = np.exp(-epsilon_decay_rate * ep)

# Display the action-value function
print('Action-Value function for all states and actions:')
print(Q)


env.startGrid(Q)
