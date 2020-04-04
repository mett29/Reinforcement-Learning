import gym
import numpy as np
import matplotlib.pyplot as plt
from windy_gridworld import WindyGridworldEnv
import time, os

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

def n_step_sarsa(env, n_episodes, n, epsilon, alpha, gamma):
    Q = create_state_action_dictionary(env)

    visits_per_state = np.zeros(env.nS)
    for _ in range(n_episodes):
        s_0 = env.reset()
        T = np.inf
        t = 0
        a_t = epsilon_greedy_action(env, epsilon, s_0, Q)
        states = [s_0]
        actions = [a_t]
        rewards = [0]
        while True:
            #env._render()
            #time.sleep(0.05)
            #os.system('cls')
            if t < T:
                next_state, r, done, info = env.step(actions[t])
                visits_per_state[next_state] += 1
                rewards.append(r)
                states.append(next_state)
                if done:
                    T = t + 1
                else:
                    next_action = epsilon_greedy_action(env, epsilon, next_state, Q)
                    actions.append(next_action)
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+n+1,T+1)):
                    G += np.power(gamma, i-tau-1) * rewards[i]
                if tau + n < T:
                    G += np.power(gamma, n) * Q[states[tau+n]][actions[tau+n]]
                Q[states[tau]][actions[tau]] = Q[states[tau]][actions[tau]] + alpha * (G - Q[states[tau]][actions[tau]])
            if tau == T - 1: 
                break
            t += 1
    return Q, visits_per_state.reshape(env.shape[0], env.shape[1])

n_episodes = 100
epsilon = 0.1
alpha = 0.5
gamma = 0.9
n = [1, 2, 8, 30]

# Try to (dis)able the king's moves to see the impact
env = WindyGridworldEnv(kings_move=True)

total_visits_per_state = []
avg_total_visit = []
for v in n:
    Q, visits_per_state = n_step_sarsa(env, n_episodes, v, epsilon, alpha, gamma)
    #avg_q_values_per_state = np.array([np.mean(list(el.values())) for el in Q.values()]).reshape((env.shape[0], env.shape[1]))
    avg_total_visit.append(np.mean(visits_per_state))
    total_visits_per_state.append(visits_per_state)

# --------------------
#  VISUALIZE RESULTS
# --------------------
fig, ax = plt.subplots(2, 2)

im = ax[0,0].imshow(total_visits_per_state[0])
ax[0,0].set_xticks(np.arange(env.shape[1]))
ax[0,0].set_yticks(np.arange(env.shape[0]))
for i in range(env.shape[0]):
    for j in range(env.shape[1]):
        text = ax[0,0].text(j, i, total_visits_per_state[0][i, j], ha="center", va="center", color="w")
ax[0,0].set_title('Visits Per State With n=1\nAverage Visits = %d' % avg_total_visit[0])

im = ax[0,1].imshow(total_visits_per_state[1])
ax[0,1].set_xticks(np.arange(env.shape[1]))
ax[0,1].set_yticks(np.arange(env.shape[0]))
for i in range(env.shape[0]):
    for j in range(env.shape[1]):
        text = ax[0,1].text(j, i, total_visits_per_state[1][i, j], ha="center", va="center", color="w")
ax[0,1].set_title('Visits Per State With n=2\nAverage Visits = %d' % avg_total_visit[1])

im = ax[1,0].imshow(total_visits_per_state[2])
ax[1,0].set_xticks(np.arange(env.shape[1]))
ax[1,0].set_yticks(np.arange(env.shape[0]))
for i in range(env.shape[0]):
    for j in range(env.shape[1]):
        text = ax[1,0].text(j, i, total_visits_per_state[2][i, j], ha="center", va="center", color="w")
ax[1,0].set_title('Visits Per State With n=8\nAverage Visits = %d' % avg_total_visit[2])

im = ax[1,1].imshow(total_visits_per_state[3])
ax[1,1].set_xticks(np.arange(env.shape[1]))
ax[1,1].set_yticks(np.arange(env.shape[0]))
for i in range(env.shape[0]):
    for j in range(env.shape[1]):
        text = ax[1,1].text(j, i, total_visits_per_state[3][i, j], ha="center", va="center", color="w")
ax[1,1].set_title('Visits Per State With n=30\nAverage Visits = %d' % avg_total_visit[3])

fig.tight_layout()
plt.show()