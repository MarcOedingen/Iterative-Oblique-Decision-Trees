import time
import copy
import numpy as np
import oracle_utils as o_utils
from Results import utils as p_utils
from sklearn.tree import DecisionTreeClassifier
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_environment as le

if __name__ == "__main__":
    max_depth = 10
    numb_runs = 10
    numb_eps = 100
    numb_trees = 10
    numb_samples = int(3e4)
    time_run = np.zeros(numb_runs)
    time_cart = np.zeros((max_depth, numb_runs))
    samples_cart = np.zeros((max_depth, numb_runs))
    rewards_cart = np.zeros((max_depth, numb_runs))
    rewards_oracle = np.zeros((max_depth, numb_runs))
    best_carts = np.empty((max_depth, numb_runs), dtype=object)
    reward_threshold = le.AcrobotEnvironment().get_reward_threshold()
    ab_env = le.AcrobotEnvironment().get_environment()
    dqn_ab = lo.DQN_Oracle().load_model(le.AcrobotEnvironment().get_environmentShort())

    print(
        "----------------------Starting experiment for Acrobot-v1 Episode samples with CART----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        ab_samples, rewards = o_utils.evaluate_oracle_ab_steps(
            model=dqn_ab, env=ab_env, num_samples=numb_samples
        )
        ab_samples = ab_samples.sample(n=numb_samples)
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")
        for i in range(max_depth):
            best_cart, best_reward, best_std = None, -np.inf, -np.inf
            time_start_cart = time.time()
            for j in range(numb_trees):
                ab_cart = DecisionTreeClassifier(max_depth=i + 1)
                ab_cart.fit(ab_samples.iloc[:, :-1].values, ab_samples.iloc[:, -1].values)
                _, cart_rewards = p_utils.eval_cart_class(
                    tree=ab_cart,
                    env=ab_env,
                    columns=ab_samples.columns,
                    num_episodes=numb_eps,
                )
                if np.mean(cart_rewards) > best_reward:
                    best_cart, best_reward, best_std = (
                        ab_cart,
                        np.mean(cart_rewards),
                        np.std(cart_rewards),
                    )
            time_cart[i, k] = time.time() - time_start_cart
            rewards_cart[i, k] = best_reward
            rewards_oracle[i, k] = np.mean(rewards)
            best_carts[i, k] = copy.deepcopy(best_cart)
            samples_cart[i, k] = numb_samples
            print(
                f"CART with a depth of {i+1} with a reward of: {best_reward} +/- {best_std}"
            )
        time_run[k] = time.time() - time_start_run
    p_utils.save_results("Acrobot-v1", rewards_cart, rewards_oracle, method="CART")
    p_utils.save_best_cart(
        best_carts, rewards_cart, reward_threshold, env_name="Acrobot-v1", method="CART"
    )
    p_utils.save_timings_samples(
        time_run, time_cart, samples_cart, env_name="Acrobot-v1", method="CART"
    )
