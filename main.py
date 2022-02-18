import algo
import numpy as np
import gym
import env_gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("----Start----")

    epsilon = 0.05
    gamma = 1
    num_eps = 10000000
    alpha = 0.01

    env = env_gym.initialise(gym)
    q, returns, action_space, state_space, policy = env_gym.initialise_values(env, np)
    q = algo.q_learning(q, epsilon, gamma, num_eps, alpha, np, env)

    # testing algo
    num_eps_test = 100
    win = 0
    for i in range(0, num_eps_test):
        print("Episode Number: ", i)
        done = False
        state = env.reset()
        while not done:
            if q[(state[0], state[1], int(state[2])), 0] > q[(state[0], state[1], int(state[2])), 1]:
                action = 0
            else:
                action = 1
            print(state, action)
            next_state, reward, done, info = env.step(action)
            state = next_state
        if reward > 0:
            win += 1
            print("WIN")
        else:
            print("LOSS")


    print("Win Pencentage = ",win)
