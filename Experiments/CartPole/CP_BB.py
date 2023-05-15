import time
import numpy as np
import oracle_utils as o_utils
from Experiments import utils as p_utils
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
    reward_threshold = le.Cartpolev1Environment().get_reward_threshold()
    cp_env = le.Cartpolev1Environment().get_environment()
    ppo_cp = lo.PPO_Oracle().load_model(
        le.Cartpolev1Environment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for CartPole-v1 Bounding Box samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        cp_samples, rewards = o_utils.evaluate_oracle_cp(
            ppo_cp, cp_env, num_episodes=eval_episodes
        )
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")

        # Get all columns except the action
        boundaries = p_utils.set_boundaries(
            cp_samples, cp_samples.columns[cp_samples.columns != "action"]
        )

        # Clip all boundaries to the environment boundaries
        boundaries.loc["position"] = boundaries.loc["position"].clip(-4.8, 4.8)
        boundaries.loc["angle"] = boundaries.loc["angle"].clip(-0.418, 0.418)

        # Create random samples
        random_samples = p_utils.gen_env_samples(boundaries, numb_samples)
        random_samples["action"], _ = ppo_cp.predict(random_samples)

        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_start_opct = time.time()
            for j in range(numb_trees):
                opct = p_utils.make_opct_class(random_samples, i + 1)
                _, opct_reward = p_utils.eval_opct_class(
                    opct=opct,
                    env=cp_env,
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
    p_utils.save_results("CartPole-v1", rewards_opct, rewards_oracle, method="BB")
    p_utils.save_best_opct(
        best_opcts, rewards_opct, reward_threshold, env_name="CartPole-v1", method="BB"
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, env_name="CartPole-v1", method="BB"
    )
