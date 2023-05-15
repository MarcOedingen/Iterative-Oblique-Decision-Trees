import time
import copy
import numpy as np
import pandas as pd
import oracle_utils as o_utils
from Experiments import utils as p_utils
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_environment as le

if __name__ == "__main__":
    max_depth = 10
    numb_runs = 10
    num_iter = 10
    num_trees = 10
    o_eval_eps = 100
    t_eval_eps = 100
    numb_samples = int(2e4)
    new_samples_iter = int(1e3)
    different = False
    time_run = np.zeros(numb_runs)
    time_cart = np.zeros((max_depth, numb_runs))
    samples_cart = np.zeros((max_depth, numb_runs))
    rewards_cart = np.zeros((max_depth, numb_runs))
    rewards_oracle = np.zeros((max_depth, numb_runs))
    best_carts = np.empty((max_depth, numb_runs), dtype=object)
    reward_threshold = le.Pendulumv1Environment().get_reward_threshold()
    env = le.Pendulumv1Environment().get_environment()
    oracle = lo.TD3_Oracle().load_model(
        le.Pendulumv1Environment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for Pendulum-v1 Iterative samples with CART----------------------"
    )
    for i in range(numb_runs):
        time_start_run = time.time()
        base_samples, rewards = o_utils.evaluate_oracle_pend_steps(
            oracle, env, numb_samples
        )
        base_samples = base_samples.sample(n=numb_samples)
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")
        for j in range(max_depth):
            print(f"Starting depth {j + 1}")
            c_samples_iter = base_samples
            best_cart, best_samples, best_reward, best_std = (
                None,
                None,
                -np.inf,
                -np.inf,
            )
            cart_times = np.zeros(num_iter + 1)
            for k in range(num_iter + 1):
                cart_times[k] = time.time()
                (cart, samples, reward, std) = p_utils.get_best_reg_cart(
                    num_trees=num_trees,
                    c_samples_iter=c_samples_iter,
                    depth=j + 1,
                    env=env,
                    t_eval_eps=t_eval_eps,
                    new_samples_iter=new_samples_iter,
                )
                if reward > best_reward:
                    best_cart = cart
                    best_samples = samples
                    best_reward = reward
                    best_std = std
                cart_times[k] = time.time() - cart_times[k]
                print(
                    f"Best CART in depth {j + 1} of iteration {k + 1} has a reward {reward} +/- {std} with"
                    f" {len(c_samples_iter)} samples."
                )
                new_samples = p_utils.new_samples_continuous(
                    oracle=oracle, best_samples=best_samples, different=different
                )
                if k < num_iter:
                    c_samples_iter = pd.concat([c_samples_iter, new_samples])
            time_cart[j, i] = np.mean(cart_times)
            rewards_cart[j, i] = best_reward
            rewards_oracle[j, i] = np.mean(rewards)
            best_carts[j, i] = copy.deepcopy(best_cart)
            samples_cart[j, i] = len(c_samples_iter)
        time_run[i] = time.time() - time_start_run
    p_utils.save_results("Pendulum-v1", rewards_cart, rewards_oracle, method="ITER_CART")
    p_utils.save_best_cart(
        best_carts,
        rewards_cart,
        reward_threshold,
        env_name="Pendulum-v1",
        method="ITER_CART",
    )
    p_utils.save_timings_samples(
        time_run, time_cart, samples_cart, env_name="Pendulum-v1", method="ITER_CART"
    )


