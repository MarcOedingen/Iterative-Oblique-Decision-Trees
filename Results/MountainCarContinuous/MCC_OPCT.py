import time
import copy
import numpy as np
import oracle_utils as o_utils
from Results import utils as p_utils
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_environment as le

if __name__ == "__main__":
    max_depth = 10
    numb_runs = 10
    numb_eps = 100
    numb_trees = 10
    numb_samples = int(3e4)
    time_run = np.zeros(numb_runs)
    time_opct = np.zeros((max_depth, numb_runs))
    samples_opct = np.zeros((max_depth, numb_runs))
    rewards_opct = np.zeros((max_depth, numb_runs))
    rewards_oracle = np.zeros((max_depth, numb_runs))
    best_opcts = np.empty((max_depth, numb_runs), dtype=object)
    reward_threshold = le.MountainCarContinuousEnvironment().get_reward_threshold()
    mcc_env = le.MountainCarContinuousEnvironment().get_environment()
    td3_ab = lo.TD3_Oracle().load_model(
        le.MountainCarContinuousEnvironment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for MountainCarContinuous-v0 Episode samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        mcc_samples, rewards = o_utils.evaluate_oracle_mcc_steps(
            td3_ab, mcc_env, num_samples=numb_samples
        )
        mcc_samples = mcc_samples.sample(n=numb_samples)
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")
        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_start_opct = time.time()
            for j in range(numb_trees):
                mcc_opct = p_utils.make_opct_reg(mcc_samples, i + 1)
                _, opct_rewards = p_utils.eval_opct_reg(
                    opct=mcc_opct,
                    env=mcc_env,
                    columns=mcc_samples.columns,
                    num_episodes=numb_eps,
                )
                if np.mean(opct_rewards) > best_reward:
                    best_opct, best_reward, best_std = (
                        mcc_opct,
                        np.mean(opct_rewards),
                        np.std(opct_rewards),
                    )
            time_opct[i, k] = time.time() - time_start_opct
            rewards_opct[i, k] = best_reward
            rewards_oracle[i, k] = np.mean(rewards)
            best_opcts[i, k] = copy.deepcopy(best_opct)
            samples_opct[i, k] = numb_samples
            print(
                f"OPCT with a depth {i + 1} with a reward of: {best_reward} +/- {best_std}"
            )
        time_run[k] = time.time() - time_start_run
    p_utils.save_results(
        "MountainCarContinuous-v0", rewards_opct, rewards_oracle, method="EPS"
    )
    p_utils.save_best_opct(
        best_opcts,
        rewards_opct,
        reward_threshold,
        env_name="MountainCarContinuous-v0",
        method="EPS",
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, "MountainCarContinuous-v0", method="EPS"
    )
