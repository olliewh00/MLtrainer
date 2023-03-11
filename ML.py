import numpy as np
import gym
import random

env = gym.make('FrozenLake-v1', render_mode='ansi')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))

# or

# q_table = np.full((state_space_size, action_space_size), 0.5)


n_ep = 10000
max_step = 1000
l_rate = 0.1
disc_rate = 0.99
ex_rate = 1
max_ex_rate = 1
min_ex_rate = 0.01
ex_decay = 0.001
rew = []

for episode in range(n_ep):
    state = env.reset()[0]
    done = False
    rewards_current_episode = 0

    for step in range(max_step):
        ex_rate_treshold = random.uniform(0, 1)
        if ex_rate_treshold > ex_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, truncated, info = env.step(action)
        q_table[state, action] = q_table[state, action] * (1 - l_rate) + l_rate * (
                    reward + disc_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward

        if done:
            break
    ex_rate = min_ex_rate + (max_ex_rate - min_ex_rate) * np.exp(-ex_decay * episode)
    rew.append(rewards_current_episode)
total_rewards = sum(rew)
average_reward = total_rewards / n_ep * 1000
print(f'Average reward per thousand episodes: {average_reward:.2f}')
print(q_table)



