# Deep Q-Learning

> When the state space is small enough, like FrozenLake-v1 or Taxi-v3, then the normal Q-Learning is sufficient to cope with the problem. But if the state space is giganic like atari $256^{210*160*3}$, it is very inefficient to train the value function using normal Q-table. But we are able to use a neural network to train it. It receives a specific state, and output the corresponding value of a specific action in that state. That is **Deep Q-Learning**.

## 1. Deep Q Network (DQN)


