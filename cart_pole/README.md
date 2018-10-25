## Deep Q-Learning

### First, what is reinforcement learning?
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)

![reinforcement_learning](https://images.ecosia.org/lisjdPOy_rATnKdUqSj2IFDjD_I=/0x390/smart/https%3A%2F%2Fi.stack.imgur.com%2FeoeSq.png)

### deep reinforcement learning = reinforcement learning + deep learning
In 2013 Google DeepMind published the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), where they demonstrated how a computer learned to play Atari 2600 video games by observing just the screen pixels and receiving a reward when the game score increased. This was done using a new algorithm called **Deep Q Network (DQN)**.

Deep learning plays its part because instead of using the classic **Q function** *Q(s,a)*, which returns the expected value if from the **current state** *s* the agent performs the **action** *a*, in Deep Q-learning there is a **neural network** used to approximate this value.

### The example in this notebook
![cartpole](https://images.ecosia.org/xOJ8LOC_xNbxq2fyI-bXyD1Fnq8=/0x390/smart/https%3A%2F%2Frubenfiszel.github.io%2Fposts%2Frl4j%2Fcartpole.gif)

This is one of the most known example of reinforcement learning. The goal is balancing a pole on top of a moving cart.
Other great examples can be found on the [**OpenAI Gym** website](https://gym.openai.com/envs/#classic_control).

### References and resources
- [Tutorial by keon.io](https://keon.io/deep-q-learning/)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [OpenAI Gym](https://gym.openai.com/envs/#classic_control)
- [Intel article](https://ai.intel.com/demystifying-deep-reinforcement-learning/)
- [Reinforcement Learning wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Q-Learning wikipedia](https://en.wikipedia.org/wiki/Q-learning)