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
    numb_samples = int(2e4)
    new_samples_iter = int(1e3)
    o_eval_eps = 100
    t_eval_eps = 100
    different = False
    time_run = np.zeros(numb_runs)
    time_opct = np.zeros((max_depth, numb_runs))
    samples_opct = np.zeros((max_depth, numb_runs))
    rewards_opct = np.zeros((max_depth, numb_runs))
    rewards_oracle = np.zeros((max_depth, numb_runs))
    best_opcts = np.empty((max_depth, numb_runs), dtype=object)
    reward_threshold = le.Cartpolev1Environment().get_reward_threshold()
    env = le.Cartpolev1Environment().get_environment()
    oracle = lo.PPO_Oracle().load_model(
        le.Cartpolev1Environment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for CartPole-v1 Iterative samples with OPCT----------------------"
    )
    for i in range(numb_runs):
        time_start_run = time.time()
        base_samples, rewards = o_utils.evaluate_oracle_cp_steps(
            oracle, env, numb_samples
        )
        base_samples = base_samples.sample(n=numb_samples)
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")
        for j in range(max_depth):
            print(f"Starting depth {j + 1}")
            c_samples_iter = base_samples
            best_opct, best_samples, best_reward, best_std = (
                None,
                None,
                -np.inf,
                -np.inf,
            )
            opct_times = np.zeros(num_iter + 1)
            for k in range(num_iter + 1):
                opct_times[k] = time.time()
                (opct, samples, reward, std,) = p_utils.get_best_class_opct(
                    num_trees=num_trees,
                    c_samples_iter=c_samples_iter,
                    depth=j + 1,
                    env=env,
                    t_eval_eps=t_eval_eps,
                    new_samples_iter=new_samples_iter,
                )
                if reward > best_reward:
                    best_opct, best_samples, best_reward, best_std = (
                        opct,
                        samples,
                        reward,
                        std,
                    )
                opct_times[k] = time.time() - opct_times[k]
                print(
                    f"Best OPCT in depth {j + 1} of iteration {k + 1} has a reward {reward} +/- {std} with"
                    f" {len(c_samples_iter)} samples."
                )
                new_samples = p_utils.new_samples_discrete(
                    oracle=oracle, best_samples=best_samples, different=different
                )
                if k < num_iter:
                    c_samples_iter = pd.concat([c_samples_iter, new_samples])
            time_opct[j, i] = np.mean(opct_times)
            rewards_opct[j, i] = best_reward
            rewards_oracle[j, i] = np.mean(rewards)
            best_opcts[j, i] = copy.deepcopy(best_opct)
            samples_opct[j, i] = len(c_samples_iter)
        time_run[i] = time.time() - time_start_run
    p_utils.save_results("CartPole-v1", rewards_opct, rewards_oracle, method="ITER")
    p_utils.save_best_opct(
        best_opcts,
        rewards_opct,
        reward_threshold,
        env_name="CartPole-v1",
        method="ITER",
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, env_name="CartPole-v1", method="ITER"
    )
