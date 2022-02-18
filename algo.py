"""
Using Q learning with Epsilon Greedy
"""


def q_learning(q, epsilon, gamma, num_eps, alpha, np, env):
    print("Q Learning")
    for i in range(num_eps):
        if i % 100000 == 0:
            print("Episode Number: ", i)
        done = False
        state = env.reset()
        while not done:
            # implementing epsilon greedy
            var_epsilon = np.random.uniform(0, 1)
            if var_epsilon > epsilon:
                action = np.random.randint(0, 2)
            else:
                """
                if policy[state[0], state[1], int(state[2])][0] > policy[state[0], state[1], int(state[2])][1]:
                    action = 0
                else:
                    action = 1
                """
                # Greedy action
                if q[(state[0], state[1], int(state[2])), 0] > q[(state[0], state[1], int(state[2])), 1]:
                    action = 0
                else:
                    action = 1

            next_state, reward, done, info = env.step(action)
            next_state = (next_state[0], next_state[1], int(next_state[2]))
            if next_state[0] > 22:
                next_state = (22, next_state[1], next_state[2])

            # updating action space
            q[state, int(action)] = q[state, int(action)] + alpha * (
                    reward + gamma * max(q[next_state, 0], q[next_state, 1]) - q[state, int(action)])

            state = next_state
    return q
