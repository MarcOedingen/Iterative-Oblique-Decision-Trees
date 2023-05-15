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
    reward_threshold = le.CartpoleSwingUpEnvironment().get_reward_threshold()
    cpsu_env = le.CartpoleSwingUpEnvironment().get_environment()
    dqn_cpsu = lo.DQN_Oracle().load_model(
        le.CartpoleSwingUpEnvironment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for CartPoleSwingUp-v1 Bounding Box samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        cpsu_samples, rewards = o_utils.evaluate_oracle_cpsu(
            dqn_cpsu, cpsu_env, num_episodes=eval_episodes
        )
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")

        # Manipulate samples, such that cos and sin are replaced by the angle
        cpsu_samples["angle"] = np.arctan2(
            cpsu_samples["angle_sin"], cpsu_samples["angle_cos"]
        )
        cpsu_samples = cpsu_samples.drop(columns=["angle_sin", "angle_cos"])

        # Get all columns except the action
        boundaries = p_utils.set_boundaries(
            cpsu_samples, cpsu_samples.columns[cpsu_samples.columns != "action"]
        )

        # Clip all boundaries to the environment boundaries
        boundaries.loc["position"] = boundaries.loc["position"].clip(-2.4, 2.4)
        boundaries.loc["angle"] = boundaries.loc["angle"].clip(-np.pi, np.pi)

        # Create random samples
        random_samples = p_utils.gen_cpsu_samples(boundaries, numb_samples)
        random_samples["action"], _ = dqn_cpsu.predict(random_samples)

        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_opct_start = time.time()
            for j in range(numb_trees):
                opct = p_utils.make_opct_class(random_samples, i + 1)
                _, opct_reward = p_utils.eval_opct_class(
                    opct=opct,
                    env=cpsu_env,
                    columns=random_samples.columns,
                    num_episodes=eval_episodes,
                )
                if np.mean(opct_reward) > best_reward:
                    best_opct = opct
                    best_reward = np.mean(opct_reward)
                    best_std = np.std(opct_reward)
            time_opct[i, k] = time.time() - time_opct_start
            rewards_opct[i, k] = best_reward
            rewards_oracle[i, k] = np.mean(rewards)
            best_opcts[i, k] = best_opct
            samples_opct[i, k] = numb_samples
            print(
                f"OPCT with depth {i+1} with a reward of {best_reward} +/- {best_std}"
            )
        time_run[k] = time.time() - time_start_run
    p_utils.save_results(
        "CartPoleSwingUp-v1", rewards_opct, rewards_oracle, method="BB"
    )
    p_utils.save_best_opct(
        best_opcts,
        rewards_opct,
        reward_threshold,
        env_name="CartPoleSwingUp-v1",
        method="BB",
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, env_name="CartPoleSwingUp-v1", method="BB"
    )
