import numpy as np
import gym
import env_gym
import algo

if __name__ == '__main__':
    print("----Start----")

    epsilon = 0.05
    gamma = 1
    num_eps = 10000

    env = env_gym.initialise(gym)
    Q, returns, action_space, state_space = env_gym.initialise_values(env, np)
    monte_carlo(Q, returns, action_space, state_space, epsilon, gamma, num_eps)
