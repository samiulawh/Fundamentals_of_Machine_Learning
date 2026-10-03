# Sami Ullah
# Roll no: 22013122-022

import numpy as np

rewards = np.array([
    [0,4,8,0,0,0],
    [0,0,8,0,0,0],
    [0,2,0,0,8,0],
    [0,1,1,0,2,10],
    [0,1,3,8,0,0],
    [0,0,1,0,0,0]
])

alpha = 0.1
gamma = 0.9

num_states = rewards.shape[0]
num_actions = rewards.shape[1]

Q = np.zeros((num_states, num_actions))

episodes = 1000

for _ in range(episodes):
    state = np.random.randint(num_states)

    actions = np.whre(rewards[state] > 0)[0]
    if len(actions) == 0:
        continue

    action = np.random.choice(actions)
    next_state = action
    reward = rewards[state, action]

    Q[state, action] += alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )

print(np.round(Q, 2))
