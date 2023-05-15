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
    reward_threshold = le.MountainCarEnvironment().get_reward_threshold()
    mc_env = le.MountainCarEnvironment().get_environment()
    dqn_mc = lo.DQN_Oracle().load_model(
        le.MountainCarEnvironment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for MountainCar-v0 Episode samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        mc_samples, rewards = o_utils.evaluate_oracle_mc_steps(
            dqn_mc, mc_env, num_samples=numb_samples
        )
        mc_samples = mc_samples.sample(n=numb_samples)
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")
        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_start_opct = time.time()
            for j in range(numb_trees):
                mc_opct = p_utils.make_opct_class(mc_samples, i + 1)
                _, opct_rewards = p_utils.eval_opct_class(
                    opct=mc_opct,
                    env=mc_env,
                    columns=mc_samples.columns,
                    num_episodes=numb_eps,
                )
                if np.mean(opct_rewards) > best_reward:
                    best_opct, best_reward, best_std = (
                        mc_opct,
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
    p_utils.save_results("MountainCar-v0", rewards_opct, rewards_oracle, method="EPS")
    p_utils.save_best_opct(
        best_opcts,
        rewards_opct,
        reward_threshold,
        env_name="MountainCar-v0",
        method="EPS",
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, env_name="MountainCar-v0", method="EPS"
    )
