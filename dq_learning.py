"""Double Q-learning model to operate on the gymnasium toy-text frozen-lake environment"""

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

from learning_common import *


def q_learning(env, grid_size: int, world: list) -> dict[int:int]:
    """Simple implementation of Q-Learning"""

    num_actions = 4
    learning_rate = 0.2
    gamma = 0.995
    num_episodes = 30_000
    lr_step = (learning_rate - 0.01) / num_episodes

    states = list(range(0, grid_size**2))

    # Set up the value tables and the random starting policies (we need 2 for double Q-learning)
    value_table_a = {}
    value_table_b = {}
    count_table = {}  # To keep track of how many times we've visited a state-action pair
    policy_a = {}
    policy_b = {}
    for s in states:
        for a in range(grid_size):
            value_table_a[(s, a)] = 1       # Optimistic initialisation
            value_table_b[(s, a)] = 1       # Optimistic initialisation
            count_table[(s, a)] = 1
        policy_a[s] = np.random.choice(num_actions)
        policy_b[s] = np.random.choice(num_actions)

    # Training loop
    episode_count = 0
    while episode_count < num_episodes:
        # Get starting state
        if episode_count % 1000 == 0:
            print(f"Starting episode: {episode_count}")
        if (episode_count - 1) % 10000 == 0 and episode_count > 1000:
            print(f"Rendering policy at episode {episode_count}.")
            policy_test(world, policy_a)

        current_state = env.reset()[0]

        learning_rate -= lr_step                                # Necessary to reduce learning rate over time

        step = 1

        while True:
            # Get UCB action
            action = get_best_ucb_action(value_table_a, current_state, count_table, step, 2, value_table_b)

            # Update the count table
            count_table[(current_state, action)] += 1

            new_state, reward, terminated, truncated, _ = env.step(action)

            done = 1 if terminated or truncated else 0
            # Update the value tables with 50% probability
            if np.random.rand() < 0.5:
                value_table_a[(current_state, action)] = (
                        value_table_a[(current_state, action)] + learning_rate * (
                            reward + (1 - done) * gamma * value_table_b[(new_state, get_max_value_action(value_table_a, new_state))] -
                            value_table_a[(current_state, action)]
                        ))
            else:
                value_table_b[(current_state, action)] = (
                        value_table_b[(current_state, action)] + learning_rate * (
                            reward + (1 - done) * gamma * value_table_a[(new_state, get_max_value_action(value_table_b, new_state))] -
                            value_table_b[(current_state, action)]
                        ))
            if done:
                break

            current_state = new_state
            step += 1

        episode_count += 1

        # Update the policy for use later
        for s in states:
            best_action = get_max_value_action(value_table_a, s, value_table_b)
            policy_a[s] = best_action

    return policy_a


def policy_test(world, policy):
    test_env = gym.make('FrozenLake-v1', desc=world, render_mode="human", is_slippery=True)
    current_state = test_env.reset()[0]
    while True:
        new_state, reward, end, trunc, _ = test_env.step(policy[current_state])
        current_state = new_state
        test_env.render()
        if end or trunc:
            break

    test_env.close()


def main(grid_size: int = 8):
    """Set up the world and try to learn"""
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
    policy_test(world, the_policy)


if __name__ == "__main__":
    main(8)
