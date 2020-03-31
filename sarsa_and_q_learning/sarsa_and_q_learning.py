import gym
from windy_gridworld import WindyGridworldEnv
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def create_state_action_dictionary(env):
    Q = {}
    for key in range(env.nS):
         Q[key] = {a: 0.0 for a in range(env.nA)}
    return Q

def epsilon_greedy_action(env, epsilon, s, Q):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        greedy_action = np.argmax(list(Q[s].values()))
        return greedy_action

def sarsa(env, n_episodes, epsilon, alpha, gamma):
    Q = create_state_action_dictionary(env)
    episode_timesteps = [[0,0]]
    timestep = 0

    for i in range(1, n_episodes):
        s = env.reset()
        a = epsilon_greedy_action(env, epsilon, s, Q)
        ''' Uncomment the lines related to timestep to visualize the env '''
        while True:
            next_state, r, done, info = env.step(a)
            #env.render()
            timestep += 1
            #print("Timestep: ", timestep)
            if done: 
                episode_timesteps.append([i, timestep])
                break
            #time.sleep(0.01)
            #os.system('cls')
            next_action = epsilon_greedy_action(env, epsilon, next_state, Q)
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[next_state][next_action] - Q[s][a])
            s = next_state
            a = next_action
    return Q, episode_timesteps

def q_learning(env, n_episodes, epsilon, alpha, gamma):
    Q = create_state_action_dictionary(env)
    q_learning_episode_timesteps = [[0,0]]
    timestep = 0
    for i in range(1, n_episodes):
        s = env.reset()
        while True:
            a = epsilon_greedy_action(env, epsilon, s, Q)
            next_state, r, done, info = env.step(a)
            timestep += 1
            if done: 
                q_learning_episode_timesteps.append([i, timestep])
                break
            Q[s][a] = Q[s][a] + alpha * (r + gamma * max(Q[next_state].values()) - Q[s][a])
            s = next_state
    return Q, q_learning_episode_timesteps

n_episodes = 200
epsilon = 0.1
alpha = 0.5
gamma = 1.0

# SARSA Without King's Moves
env_standard = WindyGridworldEnv()
Q, episode_timesteps = sarsa(env_standard, n_episodes, epsilon, alpha, gamma)

# SARSA With King's Moves
env_kings_move = WindyGridworldEnv(kings_move=True)
Q_kings, episode_timesteps_kings = sarsa(env_kings_move, n_episodes, epsilon, alpha, gamma)

# Q-Learning Without King's Moves
Q_q_learning, q_learning_episode_timesteps = q_learning(env_standard, n_episodes, epsilon, alpha, gamma)

episodes = [x[0] for x in episode_timesteps]
timesteps = [x[1] for x in episode_timesteps]
episodes_kings = [x[0] for x in episode_timesteps_kings]
timesteps_kings = [x[1] for x in episode_timesteps_kings]
q_learning_episodes = [x[0] for x in q_learning_episode_timesteps]
q_learning_timesteps = [x[1] for x in q_learning_episode_timesteps]

plt.plot(timesteps, episodes)
plt.plot(timesteps_kings, episodes_kings)
plt.plot(q_learning_timesteps, q_learning_episodes)
plt.title('Windy Gridworld - Results')
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.legend(['SARSA Without King s Moves', 'SARSA With King s Moves', 'Q-Learning Without King s Moves'], loc='upper left')
plt.show()