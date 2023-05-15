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
    numb_episodes = 100
    numb_samples = int(3e4)
    time_run = np.zeros(numb_runs)
    time_opct = np.zeros((max_depth, numb_runs))
    samples_opct = np.zeros((max_depth, numb_runs))
    rewards_opct = np.zeros((max_depth, numb_runs))
    rewards_oracle = np.zeros((max_depth, numb_runs))
    best_opcts = np.empty((max_depth, numb_runs), dtype=object)
    reward_threshold = le.AcrobotEnvironment().get_reward_threshold()
    ab_env = le.AcrobotEnvironment().get_environment()
    dqn_ab = lo.DQN_Oracle().load_model(le.AcrobotEnvironment().get_environmentShort())

    print(
        "----------------------Starting experiment for Acrobot-v1 Bounding Box samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        ab_samples, rewards = o_utils.evaluate_oracle_ab(
            dqn_ab, ab_env, num_episodes=numb_episodes
        )
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")

        # Manipulate samples, such that cos and sin are replaced by the angle
        ab_samples["theta1"] = np.arctan2(
            ab_samples["theta1_sin"], ab_samples["theta1_cos"]
        )
        ab_samples["theta2"] = np.arctan2(
            ab_samples["theta2_sin"], ab_samples["theta2_cos"]
        )
        ab_samples = ab_samples.drop(
            columns=["theta1_sin", "theta1_cos", "theta2_sin", "theta2_cos"]
        )

        # Get all columns except the action
        boundaries = p_utils.set_boundaries(
            ab_samples, ab_samples.columns[ab_samples.columns != "action"]
        )

        # Clip all boundaries to the environment boundaries
        boundaries.loc["theta1"] = boundaries.loc["theta1"].clip(-np.pi, np.pi)
        boundaries.loc["theta2"] = boundaries.loc["theta2"].clip(-np.pi, np.pi)
        boundaries.loc["theta1_dot"] = boundaries.loc["theta1_dot"].clip(
            -4 * np.pi, 4 * np.pi
        )
        boundaries.loc["theta2_dot"] = boundaries.loc["theta2_dot"].clip(
            -9 * np.pi, 9 * np.pi
        )

        # Create random samples
        random_samples = p_utils.gen_ab_samples(boundaries, numb_samples)
        random_samples["action"], _ = dqn_ab.predict(random_samples)

        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_start_opct = time.time()
            for j in range(numb_trees):
                opct = p_utils.make_opct_class(random_samples, i + 1)
                _, opct_reward = p_utils.eval_opct_class(
                    opct=opct,
                    env=ab_env,
                    columns=random_samples.columns,
                    num_episodes=numb_episodes,
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
                f"OPCT with depth {i+1} with a reward of {best_reward} +/- {best_std} "
            )
        time_run[k] = time.time() - time_start_run
    p_utils.save_results("Acrobot-v1", rewards_opct, rewards_oracle, method="BB")
    p_utils.save_best_opct(
        best_opcts, rewards_opct, reward_threshold, env_name="Acrobot-v1", method="BB"
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, env_name="Acrobot-v1", method="BB"
    )
