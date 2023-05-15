import time
import numpy as np
import oracle_utils as o_utils
from Results import utils as p_utils
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_environment as le

if __name__ == "__main__":
    max_depth = 10
    numb_runs = 10
    numb_trees = 10
    eval_episodes = 100
    numb_samples = int(3e4)
    time_run = np.zeros(numb_runs)
    time_opct = np.zeros((max_depth, numb_runs))
    samples_opct = np.zeros((max_depth, numb_runs))
    rewards_opct = np.zeros((max_depth, numb_runs))
    rewards_oracle = np.zeros((max_depth, numb_runs))
    best_opcts = np.empty((max_depth, numb_runs), dtype=object)
    reward_threshold = le.MountainCarEnvironment().get_reward_threshold()
    mcc_env = le.MountainCarContinuousEnvironment().get_environment()
    td3_mcc = lo.TD3_Oracle().load_model(
        le.MountainCarContinuousEnvironment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for MountainCarContinuous-v0 Bounding Box samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        mcc_samples, rewards = o_utils.evaluate_oracle_mcc(
            td3_mcc, mcc_env, num_episodes=eval_episodes
        )
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")

        # Get all columns except the action
        boundaries = p_utils.set_boundaries(
            mcc_samples, mcc_samples.columns[mcc_samples.columns != "action"]
        )

        # Clip all boundaries to the environment boundaries
        boundaries.loc["position"] = boundaries.loc["position"].clip(-1.2, 0.6)

        # Create random samples
        random_samples = p_utils.gen_env_samples(boundaries, numb_samples)
        random_samples["action"], _ = td3_mcc.predict(random_samples)

        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_start_opct = time.time()
            for j in range(numb_trees):
                opct = p_utils.make_opct_reg(random_samples, i + 1)
                _, opct_reward = p_utils.eval_opct_reg(
                    opct=opct,
                    env=mcc_env,
                    columns=random_samples.columns,
                    num_episodes=eval_episodes,
                )
                if np.mean(opct_reward) > best_reward:
                    best_reward = np.mean(opct_reward)
                    best_std = np.std(opct_reward)
                    best_opct = opct
            time_opct[i, k] = time.time() - time_start_opct
            rewards_opct[i, k] = best_reward
            rewards_oracle[i, k] = np.mean(rewards)
            best_opcts[i, k] = best_opct
            samples_opct[i, k] = numb_samples
            print(
                f"OPCT with depth {i+1} with a reward of {best_reward} +/- {best_std}"
            )
        time_run[k] = time.time() - time_start_run
    p_utils.save_results(
        "MountainCarContinuous-v0", rewards_opct, rewards_oracle, method="BB"
    )
    p_utils.save_best_opct(
        best_opcts,
        rewards_opct,
        reward_threshold,
        env_name="MountainCarContinuous-v0",
        method="BB",
    )
    p_utils.save_timings_samples(
        time_run,
        time_opct,
        samples_opct,
        env_name="MountainCarContinuous-v0",
        method="BB",
    )
