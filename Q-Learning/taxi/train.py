import numpy as np
import gymnasium as gym
import random
import os
import time

from tqdm import tqdm

def initialize_q_table(state_space, action_space):
	Qtable = np.zeros((state_space, action_space))
	return Qtable

def greedy_policy(Qtable, state):
	action = np.argmax(Qtable[state][:])
	return action

def epsilon_greedy_policy(Qtable, state, epsilon):
	randnum = random.uniform(0, 1)
	if randnum > epsilon:
		action = greedy_policy(Qtable, state)
	else:
		action = env.action_space.sample()
	return action

def train(n_training_episodes, min_epsilon, max_epsilon, env, decay_rate, max_steps, Qtable):
	for episode in tqdm(range(n_training_episodes)):
		# Reduce epsilon
		epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

		# Reset the environment
		state, info = env.reset()
		terminated = False
		truncated = False

		# train
		for step in range(max_steps):
			# Choose an action using the epsilon-greedy-policy
			action = epsilon_greedy_policy(Qtable, state, epsilon)
			
			# Take the action and receive the reward and be in a new state
			new_state, reward, terminated, truncated, info = env.step(action)

			# Update Q(s,a) = Q(s,a) + lr[R(s,a) + gamma * maxQ(s',a') - Q(s,a)]
			Qtable[state][action] = Qtable[state][action] + lr * (
				reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
				)
			
			# If terminated or truncated, finish the current episode
			if terminated or truncated:
				break

			state = new_state

	return Qtable

def evaluate(n_eval_episodes, env, max_steps, Qtable, seed):
	episode_rewards = []
	for episode in tqdm(range(n_eval_episodes)):
		if seed:
			state, info = env.reset(seed=seed[episode])
		else:
			state, info = env.reset()

		truncated = False
		terminated = False
		total_reward_ep = 0

		for step in range(max_steps):
			action = greedy_policy(Qtable, state)
			new_state, reward, terminated, truncated, info = env.step(action)
			total_reward_ep += reward

			if terminated or truncated:
				break

			state = new_state

		episode_rewards.append(total_reward_ep)

	mean_reward = np.mean(episode_rewards)
	std_reward = np.std(episode_rewards)

	return mean_reward, std_reward

def watch_agent_play(env_id, Qtable, max_steps):
	env_visual = gym.make(env_id, render_mode="human")

	for episode in range(5):
		state, info = env_visual.reset()
		terminated = False
		truncated = False
		for step in range(max_steps):
			action = greedy_policy(Qtable, state)
			new_state, reward, terminated, truncated, info = env_visual.step(action)
			time.sleep(0.5)
			if terminated or truncated:
				print("Success!")
				break
			state = new_state
	env_visual.close()
		
env = gym.make("Taxi-v3", render_mode="rgb_array")

state_space = env.observation_space.n
action_space = env.action_space.n

print("There are ", state_space, " possible states.")
print("There are ", action_space, " possible actions.")

Qtable = initialize_q_table(state_space, action_space)

# Training Parameters
n_training_episodes = 25000
lr = 0.7

# Evaluating Parameters
n_eval_episodes = 100

# Environment Parameters
env_id = "Taxi-v3"
max_steps = 99
gamma = 0.95
eval_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.005


# Start training
Qtable = train(n_training_episodes, min_epsilon, max_epsilon, env, decay_rate, max_steps, Qtable)

# Evaluating
mean_reward, std_reward = evaluate(n_eval_episodes, env, max_steps, Qtable, eval_seed)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Visualization
watch_agent_play(env_id, Qtable, max_steps)
