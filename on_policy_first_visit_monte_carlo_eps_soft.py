import gym
import numpy as np
import matplotlib.pyplot as plt
import random

# action mapping to display the final result
action_mapping = {
    3: '\u2191', # UP
    2: '\u2192', # RIGHT
    1: '\u2193', # DOWN
    0: '\u2190' # LEFT
}

def create_random_policy(env):
     policy = {}
     for key in range(env.nS):
          p = {}
          for action in range(env.nA):
               p[action] = 1 / env.nA
          policy[key] = p
     return policy
    
def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
         Q[key] = {a: 0.0 for a in range(0, env.nA)}
    return Q

def play_episode(env, P):
    env.reset()
    state_action_reward = []
    done = False

    while not done:
        s = env.env.s
        timestep = []
        timestep.append(s)
        probs = list(P[s].values())
        a = np.random.choice(np.arange(len(probs)), p=probs)
        s, r, done, _ = env.step(a)
        timestep.append(a)
        timestep.append(r)

        state_action_reward.append(timestep)
        
    return state_action_reward

def test_policy(env, policy):
    wins = 0
    nb_play = 100
    for i in range(nb_play):
        r = play_episode(env, policy)[-1][-1]
        if r == 1:
            wins += 1
    return wins / nb_play

def monte_carlo_eps_soft(env, n_episodes=5000, P=None, epsilon=0.01, gamma=0.9):
    
    if P == None:
        P = create_random_policy(env)
    Q = create_state_action_dictionary(env, P)

    returns = {}
    avg_wins = []

    for ep_counter in range(n_episodes):

        if ep_counter % 500 == 0 and ep_counter is not 0:
            print("Episode n. %d" % ep_counter)

        # Epsilon decreasing
        epsilon = max(0, epsilon - ep_counter/n_episodes)
        # Generate an episode using P
        episode = play_episode(env, P)
        actual_r = episode[-1][-1]
        avg_wins.append((sum(avg_wins[-30:])+actual_r) / (len(avg_wins[-30:]) + 1))
        G = 0

        for i in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            G = r_t + gamma * G
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action])

                Q_values = list(map(lambda x: x[1], Q[s_t].items()))
                best_actions = [a for a, q in enumerate(Q_values) if q == max(Q_values)]
                max_Q = random.choice(best_actions) # with ties broken arbitrarily

                for a in P[s_t].items():
                    if a[0] == max_Q:
                        P[s_t][a[0]] = 1 - epsilon + (epsilon / env.nA)
                    else:
                        P[s_t][a[0]] = (epsilon / env.nA)
    return P, avg_wins

env = gym.make('FrozenLake-v0', is_slippery=True)
P, avg_wins = monte_carlo_eps_soft(env)
#print(P)
policy = []
for s in range(env.nS):
    probs = list(P[s].values())
    policy.append(np.argmax(probs))
arrows = np.array([action_mapping[int(action)] for action in policy])
print(arrows.reshape((4,4)))

print("Wins %: ", test_policy(env, P))

plt.plot(avg_wins)
plt.title('Avg Rewards Over Time')
plt.xlabel('Episodes')
plt.ylabel('Avg Rewards')
plt.show()