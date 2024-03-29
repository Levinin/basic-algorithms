# Common functions for the tabular Q-learning and SARSA algorithms
import numpy as np


def print_world(world):
    for line in world:
        print(line, end="\n")


def print_policy(policy: dict[int, int], length: int):
    action_map = {0: "L", 1: "D", 2: "R", 3: "U"}
    action_renders = list(map(lambda x: action_map[x], policy.values()))
    for i in range(length):
        print("".join(action_renders[i*length:(i+1)*length]))


def get_max_value_action(value_table: dict[tuple[int, int], int],
                         state: int,
                         vt2: dict[tuple[int, int], int] = None) -> int:
    """Get the action with the largest value from the options available in the given state"""
    state_action = [(s, a) for s, a in zip([state] * 4, range(4))]
    vals = [value_table[sa] for sa in state_action]
    if vt2 is not None:
        vals = [vals[i] + vt2[sa] for i, sa in enumerate(state_action)]
    # Where all vals are the same there is no clear policy so choose a random action.
    # If we don't do this, argmax will always choose 0 (left) which is not helpful.
    if len(set(vals)) == 1:
        return np.random.choice(4)
    return np.argmax(vals)


def get_max_next_value(value_table: dict[tuple[int, int], int], state: int) -> float:
    """Get the max value of the next state action pair from the options available"""
    state_action = [(s, a) for s, a in zip([state] * 4, range(4))]
    return max([value_table[sa] for sa in state_action])


def get_best_ucb_action(value_table: dict[tuple[int, int], int], state: int, count_table: dict[tuple[int, int], int],
                        step: int, c: float, vt2: dict[tuple[int, int], int] = None) -> int:
    """Get the best action according to the upper confidence bound"""
    state_action = [(s, a) for s, a in zip([state] * 4, range(4))]
    vals = [value_table[sa] + c * np.sqrt(np.log(step) / count_table[sa]) for sa in state_action]
    if vt2 is not None:
        vals = [vals[i] + (vt2[sa] + c * np.sqrt(np.log(step) / count_table[sa])) for i, sa in enumerate(state_action)]
    return np.argmax(vals)



