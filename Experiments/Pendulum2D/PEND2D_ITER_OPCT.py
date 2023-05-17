import numpy as np
import pandas as pd
import oracle_utils as o_utils
from Experiments import utils as p_utils
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_environment as le

if __name__ == "__main__":
    max_depth = 6
    numb_runs = 1
    num_iter = 2
    num_trees = 10
    o_eval_eps = 100
    t_eval_eps = 100
    numb_samples = int(2e4)
    new_samples_iter = int(1e3)
    different = False
    env = le.Pendulum2DEnvironment().get_environment()
    oracle = lo.DDPG_Oracle().load_model(
        le.Pendulum2DEnvironment().get_environmentShort()
    )
    print(
        "----------------------Starting experiment for Pendulum2D Iterative samples with OPCT----------------------"
    )
    base_samples, rewards, o_samples_eps = o_utils.evaluate_oracle_pend2d_steps(
        oracle, env, numb_samples
    )
    base_samples = base_samples.sample(n=numb_samples)
    print(f"Oracle performance: {np.mean(rewards)} +/- {np.std(rewards)}")
    for j in range(5, max_depth):
        print(f"Starting depth {j + 1}")
        c_samples_iter = base_samples
        best_opct, best_samples, best_reward, best_std = (
            None,
            None,
            -np.inf,
            -np.inf,
        )
        t_samples_eps = []
        t_rewards = []
        for k in range(num_iter + 1):
            (opct, samples, reward, std, samples_eps) = p_utils.get_best_reg_opct_eps(
                num_trees=num_trees,
                c_samples_iter=c_samples_iter,
                depth=j + 1,
                env=env,
                t_eval_eps=t_eval_eps,
                new_samples_iter=new_samples_iter,
            )
            t_samples_eps.append(samples_eps)
            t_rewards.append(reward)
            if reward > best_reward:
                best_opct, best_samples, best_reward, best_std = (
                    opct,
                    samples,
                    reward,
                    std,
                )
            print(
                f"Best OPCT in depth {j + 1} of iteration {k + 1} has a reward {reward} +/- {std} with"
                f" {len(c_samples_iter)} samples."
            )
            new_samples = p_utils.new_samples_continuous(
                oracle=oracle, best_samples=best_samples, different=different
            )
            if k < num_iter:
                c_samples_iter = pd.concat([c_samples_iter, new_samples])

    oracle_rand_indices = np.random.choice(len(o_samples_eps), size=10, replace=False)
    random_oracle_episodes = [o_samples_eps[i] for i in oracle_rand_indices]


    opct_rand_indices = [np.random.choice(len(t_samples_eps[i]), size=10, replace=False) for i in range(num_iter+1)]
    random_opct_episodes = []
    for i in range(num_iter+1):
        random_opct_episodes.append([t_samples_eps[i][j] for j in opct_rand_indices[i]])
    p_utils.plot_episodes(random_oracle_episodes, random_opct_episodes, num_iter+1)

    # Rewards oracle in rewards and opct rewards in t_rewards
