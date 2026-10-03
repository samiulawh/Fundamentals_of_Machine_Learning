# Sami Ullah
# Roll no: 22013122-022

import numpy as np

rewards = np.array([
    [0, 4, 8, 0, 0, 0],
    [0, 0, 8, 0, 0, 0],
    [0, 2, 0, 0, 8, 0],
    [0, 1, 1, 0, 2, 10],
    [0, 1, 3, 8, 0, 0],
    [0, 0, 1, 0, 0, 0]
])

alpha = 0.1
gamma = 0.9
q_table = np.zeros(rewards.shape)
episodes = 100

for _ in range(episodes):
    state = np.random.randint(rewards.shape[0])
    valid_actions = np.where(rewards[state] > 0)[0]
    if len(valid_actions) == 0:
        continue
    action = np.random.choice(valid_actions)

    next_state = action
    reward = rewards[state, action]

    next_actions = np.where(rewards[next_state] > 0)[0]
    if len(next_actions) == 0:
        Q_next = 0
    else:
        next_action = np.random.choice(next_actions)
        Q_next = q_table[next_state, next_action]

    q_table[state, action] += alpha * (reward + gamma * Q_next - q_table[state, action])

print(np.round(q_table, 2))
