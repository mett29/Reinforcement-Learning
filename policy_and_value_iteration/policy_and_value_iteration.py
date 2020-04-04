import gym
import numpy as np
import matplotlib.pyplot as plt

# action mapping to display the final result
action_mapping = {
    3: '\u2191', # UP
    2: '\u2192', # RIGHT
    1: '\u2193', # DOWN
    0: '\u2190' # LEFT
}

def value_iteration(env, gamma=0.9, epsilon=0.001, V=None, verbose=True):
    env.reset()
    env.render()

    if V == None:
        V = np.zeros([env.nS])
    max_iterations = 1000

    for i in range(max_iterations):
        delta = 0
        for s in range(env.nS):
            v_prev = V[s]
            Q_value = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, r, done in env.P[s][a]:
                    Q_value[a] += prob * (r + gamma * V[next_state])
            V[s] = np.max(Q_value)
            delta = max(delta, np.abs(v_prev - V[s]))
            
        if delta < epsilon:
            print('\nValue converged at iteration %d\n' % (i+1))
            break
    return V

def get_policy(env, V, gamma=0.9):
    P = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        Q_value = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, r, done in env.P[s][a]:
                Q_value[a] += prob * (r + gamma * V[next_state])
        P[s] = np.argmax(Q_value)
    return P

def policy_iteration(env, gamma=0.9, epsilon=0.001, P=None, verbose=True):
    env.reset()

    V = np.zeros([env.nS])

    if P == None:
        P = np.random.choice(env.nA, size=env.nS)
    max_iterations = 1000

    for i in range(max_iterations):

        # POLICY EVALUATION
        while True:
            delta = 0
            for s, a in enumerate(P):
                v = 0
                for prob, next_state, r, done in env.P[s][a]:
                    v += prob * (r + gamma * V[next_state])
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v 
            if delta < epsilon: 
                break

        # POLICY IMPROVEMENT
        policy_stable = True
        for s in range(env.nS):
            old_action = P[s]
            Q_value = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, r, done in env.P[s][a]:
                    Q_value[a] += prob * (r + gamma * V[next_state])
            best_action = np.argmax(Q_value)
            if old_action != best_action:
                policy_stable = False
            P[s] = best_action
        
        if policy_stable: break

    return V, P 
    

def play_episodes(env, n_episodes, policy):
    rewards = []

    for episode in range(n_episodes):
        s = env.reset()
        while True:
            a = policy[s]
            next_state, r, done, info = env.step(a)
            s = next_state
            if done:
                rewards.append(r)
                break
    return rewards

env = gym.make('FrozenLake-v0', is_slippery=True)

#-----------------
# VALUE ITERATION
#-----------------
print("VALUE ITERATION:")
V = value_iteration(env)
P = get_policy(env, V)
print(V.reshape(4,4))
print(P)
policy = np.array([action_mapping[int(action)] for action in P])
print(policy.reshape((4,4)))

n_episodes = 1000
rewards1 = play_episodes(env, n_episodes, P)
print("\nAverage reward (VI): ", np.mean(rewards1))

# -----------------
# POLICY ITERATION
# -----------------
print("POLICY ITERATION:\n")
V2, P2 = policy_iteration(env)
print(V2.reshape(4,4))
print(P2)
policy2 = np.array([action_mapping[int(action)] for action in P2])
print(policy2.reshape((4,4)))
n_episodes = 1000
rewards2 = play_episodes(env, n_episodes, P2)
print("\nAverage reward (PI): ", np.mean(rewards2))


# Rewards Graphs
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax1.title.set_text('Value Iteration Rewards')
ax2.title.set_text('Policy Iteration Rewards')
ax1.plot(rewards1)
ax2.plot(rewards2)
plt.show()