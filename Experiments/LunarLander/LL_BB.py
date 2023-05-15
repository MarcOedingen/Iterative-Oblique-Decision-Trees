import time
import numpy as np
import pandas as pd
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
    reward_threshold = le.LunarLanderEnvironment().get_reward_threshold()
    ll_env = le.LunarLanderEnvironment().get_environment()
    ll_init_env = le.LunarLanderHeatmapEnvironment().get_environment()
    ppo_ll = lo.PPO_Oracle().load_model(
        le.LunarLanderEnvironment().get_environmentShort()
    )

    print(
        "----------------------Starting experiment for LunarLander-v2 Bounding Box samples with OPCT----------------------"
    )
    for k in range(numb_runs):
        time_start_run = time.time()
        ll_samples, rewards = o_utils.evaluate_oracle_ll(
            ppo_ll, ll_env, num_episodes=eval_episodes
        )
        print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")

        boundaries = p_utils.set_boundaries(
            ll_samples, set(ll_samples.columns) - {"action", "first_leg", "second_leg"}
        )

        boundaries.loc["x"] = boundaries.loc["x"].clip(-1.5, 1.5)
        boundaries.loc["y"] = boundaries.loc["y"].clip(-1.5, 1.5)
        boundaries.loc["vx"] = boundaries.loc["vx"].clip(-5, 5)
        boundaries.loc["vy"] = boundaries.loc["vy"].clip(-5, 5)
        boundaries.loc["theta"] = boundaries.loc["theta"].clip(-np.pi, np.pi)
        boundaries.loc["theta_dot"] = boundaries.loc["theta_dot"].clip(-5, 5)

        # Create random samples
        random_samples = p_utils.gen_env_samples(boundaries, numb_samples)
        reset_func = lambda x: ll_init_env.reset(start_state=x)[0]
        states = np.apply_along_axis(reset_func, 1, random_samples.to_numpy())
        random_samples = pd.DataFrame(
            states, columns=ll_samples.columns[ll_samples.columns != "action"]
        )
        random_samples["action"], _ = ppo_ll.predict(states)

        for i in range(max_depth):
            best_opct, best_reward, best_std = None, -np.inf, -np.inf
            time_start_opct = time.time()
            for j in range(numb_trees):
                opct = p_utils.make_opct_class(random_samples, i + 1)
                _, opct_reward = p_utils.eval_opct_class(
                    opct=opct,
                    env=ll_env,
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
                f"OPCT with depth {i + 1} with a reward of {best_reward} +/- {best_std}"
            )
        time_run[k] = time.time() - time_start_run
    p_utils.save_results("LunarLander-v2", rewards_opct, rewards_oracle, method="BB")
    p_utils.save_best_opct(
        best_opcts,
        rewards_opct,
        reward_threshold,
        env_name="LunarLander-v2",
        method="BB",
    )
    p_utils.save_timings_samples(
        time_run, time_opct, samples_opct, env_name="LunarLander-v2", method="BB"
    )
