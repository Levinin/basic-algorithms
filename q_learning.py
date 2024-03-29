"""Q-learning model to operate on the gymnasium toy-text frozen-lake environment"""

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

from learning_common import *


def q_learning(env, grid_size: int, world: list) -> dict[int:int]:
    """Simple implementation of Q-Learning"""

    learning_rate = 0.2
    gamma = 0.995
    num_episodes = 30_000
    epsilon_start = 0.8
    epsilon_end = 0.01
    eps_factor = (epsilon_start - epsilon_end) / (num_episodes / 2)
    lr_step = (learning_rate - 0.01) / num_episodes

    states = list(range(0, grid_size**2))

    # Set up the value table and the random starting policy
    value_table = {}                                    # Will be a dict of {f"{s},{a}": value}}
    for s in states:
        for a in range(grid_size):
            value_table[(s, a)] = 1
    policy = {}
    for s in states:
        policy[s] = np.random.choice(4)

    episode_count = 0
    while episode_count < num_episodes:
        # Get starting state
        if episode_count % 1000 == 0:
            print(f"Starting episode: {episode_count}")
        if (episode_count - 1) % 10000 == 0 and episode_count > 1000:
            print(f"Rendering policy at episode {episode_count}.")
            test_policy(world, policy)

        current_state = env.reset()[0]

        eps = epsilon_start - eps_factor * episode_count        # Reducing epsilon necessary for convergence
        learning_rate -= lr_step                                # Necessary to reduce learning rate over time

        while True:
            # Get an e-greedy action
            if np.random.rand() < eps:
                action = np.random.choice(4)
            else:
                action = get_max_value_action(value_table, current_state)

            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                value_table[(current_state, action)] = (
                        value_table[(current_state, action)] + learning_rate *
                        (reward - value_table[(current_state, action)])
                )
                break

            else:
                value_table[(current_state, action)] = (
                        value_table[(current_state, action)] + learning_rate * (
                            reward + gamma * get_max_next_value(value_table, new_state) -
                            value_table[(current_state, action)]
                        ))

            current_state = new_state
        episode_count += 1

        # Update the policy for use later
        for s in states:
            best_action = get_max_value_action(value_table, s)
            policy[s] = best_action

    return policy


def test_policy(world, policy):
    test_env = gym.make('FrozenLake-v1', desc=world, render_mode="human", is_slippery=True)
    current_state = test_env.reset()[0]
    while True:
        new_state, reward, end, trunc, _ = test_env.step(policy[current_state])
        current_state = new_state
        test_env.render()
        if end or trunc:
            break

    test_env.close()


def main():
    """Set up the world and try to learn"""
    grid_size = 8
    world = generate_random_map(size=grid_size)
    print(f"Let's learn this new ice world!\n{print_world(world)}\n")

    env = gym.make('FrozenLake-v1', desc=world, is_slippery=True)
    the_policy = q_learning(env, grid_size, world)

    print(f"The world we learnt on is: ")
    print_world(world)

    print(f"\nThe new policy is: ")
    print_policy(the_policy, grid_size)
    env.close()

    # Let's test the new policy
    test_policy(world, the_policy)


def test_get_max_next_value():
    """Unit test for get_max_next_value"""
    value_table = {(0, 0): 1, (0, 1): 2, (0, 2): 3, (0, 3): 4}
    assert get_max_next_value(value_table, 0) == 4


if __name__ == "__main__":
    main()
