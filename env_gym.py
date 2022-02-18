""" Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed dealer.
    Card Values:
    - Face cards (Jack, Queen, King) have point value 10.
    - Aces can either count as 11 or 1, and it's called 'usable ace' at 11.
    - Numerical cards (2-9) have value of their number.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards.
    The player can request additional cards (hit, action=1) until they decide to stop
    (stick, action=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their face down card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.
    The agent take a 1-element vector for actions.
    The action space is `(action)`, where:
    - `action` is used to decide stick/hit for values (0,1).
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace), and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    **Rewards:**
    Reward schedule:
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
        +1.5 (if <a href="#nat">natural</a> is True.)
        +1 (if <a href="#nat">natural</a> is False.)
    ### Arguments
    ```
    gym.make('Blackjack-v1', natural=False)
    ```
    <a id="nat">`natural`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).
    ### Version History
    * v0: Initial versions release (1.0.0)
"""


def initialise(gym):
    env = gym.make('Blackjack-v1', natural=True)
    print("Setting up Env")
    return env


def initialise_values(env, np):
    state_space = np.zeros((400, 3))
    action_space = np.array([0, 1])
    q = {}
    returns = {}
    ctr = 0
    policy = {}

    for total in range(4, 23):
        for dealer in range(1, 11):
            for uace in range(0, 2):
                state_space[ctr] = (np.array([total, dealer, uace]))
                for action in action_space:
                    q[(total, dealer, uace), action] = 0
                    returns[(total, dealer, uace), action] = 0
                ctr += 1

    # we start with a uniform random policy
    for state in state_space:
        policy[(state[0], state[1], state[2])] = (0.5, 0.5)  # both action hit and stick have same probability in start
    return q, returns, action_space, state_space, policy
