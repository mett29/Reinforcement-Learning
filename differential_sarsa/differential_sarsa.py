import numpy as np
import matplotlib.pyplot as plt
from tile_coding import *

'''
Except for some changes, the code is mainly taken from:
https://github.com/MJeremy2017/Reinforcement-Learning-Implementation
'''

REJECT = 0
ACCEPT = 1
ACTIONS = [REJECT, ACCEPT]

NUM_SERVERS = 10

class ValueFunction:

    def __init__(self, alpha=0.01, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings
        # Exactly one feature is present in each tiling, so the total number
        # of features present is always the same as the number of tilings.
        # This allows to set the step-size parameter alpha in an intuitive way.
        self.alpha = alpha / numOfTilings
        self.hashTable = IHT(maxSize)
        self.weights = np.zeros(maxSize)
        self.serverScale = self.numOfTilings / NUM_SERVERS  # 10 servers
        self.priorityScale = self.numOfTilings / 3.0  # 4 kinds of priorities

    # Get the indices of the active tiles for a given state and action
    def getActiveTiles(self, n_server, priority, action):
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.serverScale * n_server, self.priorityScale * priority],
                            [action])
        return activeTiles

    # Estimate the value of a given state and action
    def value(self, state, action):
        n_server, priority = state
        activeTiles = self.getActiveTiles(n_server, priority, action)
        return np.sum(self.weights[activeTiles])

    # Learn with a given state, action and target:
    # because each component is either 0 or 1, the weighted sum making up the
    # approximate value function can be computed efficiently by taking the
    # indices of the active features and then adding up the n corresponding
    # components of the weight vector.
    def update(self, state, action, delta):
        n_server, priority = state
        activeTiles = self.getActiveTiles(n_server, priority, action)
        delta_alpha = delta * self.alpha
        for activeTile in activeTiles:
            self.weights[activeTile] += delta_alpha

    def stateValue(self, state):
        if state[0] == 0:
            # No server available
            return self.value(state, 0)
        values = [self.value(state, a) for a in ACTIONS]
        return max(values)

class ServerAccess:
    def __init__(self, exp_rate=0.1, lr=0.1, beta=0.01):
        self.n_server = NUM_SERVERS
        self.free_prob = 0.06
        self.priorities = range(4)
        self.actions = ACTIONS  # 0: reject; 1: accept
        self.state = (0, 0)  # (num_servers, priority of the customer at the head of the queue)

        self.exp_rate = exp_rate
        self.lr = lr
        self.beta = beta

    def getFreeServers(self):
        n = 0
        n_free_server = self.state[0]
        n_busy_server = self.n_server - n_free_server
        # Each busy server becomes free with p = 0.06 on each time step
        for _ in range(n_busy_server):
            if np.random.uniform(0, 1) <= 0.06:
                n += 1
        n_free_server += n
        self.state = (n_free_server, self.state[1])
        return n_free_server

    def chooseAction(self, valueFunc):
        n_free_server = self.getFreeServers()
        # A customer cannot be served if there are no free servers
        if n_free_server == 0:
            return 0
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            values = {}
            for a in self.actions:
                v = valueFunc.value(self.state, a)
                values[a] = v
            action = np.random.choice([k for k, v in values.items() if v == max(values.values())])
        return action

    def nextState(self, action):
        if action == 1:
            n_free_server = self.state[0] - 1
        else:
            n_free_server = self.state[0]
        priority = np.random.choice(self.priorities)
        self.state = (n_free_server, priority)
        return self.state

    def giveReward(self, action):
        if action == 1:
            priority = self.state[1]
            return np.power(2, priority)
        return 0
    
    def differential_semi_gradient_sarsa(self, value_function, steps):
        avg_reward = 0
        # Initialize state S and action A
        self.state = (NUM_SERVERS, np.random.choice(self.priorities))
        current_state = self.state
        current_action = self.chooseAction(value_function)

        total_reward = 0
        for i in range(steps):
            # Take action A, observe R,S'
            r = self.giveReward(current_action)
            total_reward += r
            next_state = self.nextState(current_action)
            # Choose A' eps-greedily
            next_action = self.chooseAction(value_function)
            delta = r - avg_reward + value_function.value(next_state, next_action) - value_function.value(current_state, current_action)
            avg_reward = avg_reward + self.beta * delta
            value_function.update(current_state, current_action, delta)
            # S <-- S'
            current_state = next_state
            # A <-- A'
            current_action = next_action

if __name__ == "__main__":
    server_access = ServerAccess(exp_rate=0.1)
    value_function = ValueFunction()
    server_access.differential_semi_gradient_sarsa(value_function, steps=100000)

    plt.figure(figsize=[10, 6])

    for priority in range(4):
        n_servers = []
        values = []
        for n_server in range(11):
            value = value_function.stateValue((n_server, priority))
            n_servers.append(n_server)
            values.append(value)
        plt.plot(n_servers, values, label="priority {}".format(np.power(2, priority)))
    plt.legend()
    plt.title('Value Function')
    plt.xlabel('Number of free servers')
    plt.ylabel('Differential value of best action')
    plt.show()