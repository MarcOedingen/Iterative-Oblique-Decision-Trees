import os
import copy
import spyct
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def save_results(env_name, t_reward, o_reward, method):
    if not os.path.exists("Experiments/Rewards_NPZ"):
        os.makedirs("Experiments/Rewards_NPZ")
    np.savez_compressed(
        f"Experiments/Rewards_NPZ/{env_name}_{method}",
        t_reward=t_reward,
        o_reward=o_reward,
    )


def save_best_opct(best_opcts, best_rewards, reward_threshold, env_name, method):
    if not os.path.exists("Experiments/OPCTs"):
        os.makedirs("Experiments/OPCTs")
    solved_depth = (
        10
        if np.max(np.mean(best_rewards, axis=1)) < reward_threshold
        else np.min(np.where(np.mean(best_rewards, axis=1) >= reward_threshold))
    )
    if solved_depth < 10:
        best_opct = best_opcts[solved_depth, np.argmax(best_rewards[solved_depth])]
    else:
        print(
            f"No OPCT found that exceeds the reward threshold for environment {env_name} with method {method}."
            f" Saving best OPCT with maximum depth (10)."
        )
        best_opct, solved_depth = best_opcts[-1, np.argmax(best_rewards[-1])], 9
    with open(
        f"Experiments/OPCTs/{env_name}_{method}_depth_{solved_depth+1}.pkl", "wb"
    ) as f:
        pickle.dump(best_opct, f)


def save_best_cart(best_carts, best_rewards, reward_threshold, env_name, method):
    if not os.path.exists("Experiments/CARTs"):
        os.makedirs("Experiments/CARTs")
    solved_depth = (
        10
        if np.max(np.mean(best_rewards, axis=1)) < reward_threshold
        else np.min(np.where(np.mean(best_rewards, axis=1) >= reward_threshold))
    )
    if solved_depth < 10:
        best_cart = best_carts[solved_depth, np.argmax(best_rewards[solved_depth])]
    else:
        print(
            f"No CART found that exceeds the reward threshold for environment {env_name} with method {method}."
            f" Saving best CART with maximum depth (10)."
        )
        best_cart, solved_depth = best_carts[-1, np.argmax(best_rewards[-1])], 9
    with open(
        f"Experiments/CARTs/{env_name}_{method}_depth_{solved_depth+1}.pkl", "wb"
    ) as f:
        pickle.dump(best_cart, f)


def save_timings_samples(time_run, time_opct, samples, env_name, method):
    if not os.path.exists("Experiments/Times_Samples"):
        os.makedirs("Experiments/Times_Samples")
    np.savez_compressed(
        f"Experiments/Times_Samples/{env_name}_{method}",
        time_run=time_run,
        time_opct=time_opct,
        samples=samples,
    )


def load_results(env_name, method):
    assert os.path.exists(
        f"Experiments/Rewards_NPZ/{env_name}_{method}.npz"
    ), f"Path does not exist: Experiments/Rewards_NPZ/{env_name}_{method}.npz"
    data = np.load(f"Experiments/Rewards_NPZ/{env_name}_{method}.npz")
    t_reward = data["t_reward"]
    o_reward = data["o_reward"]
    return t_reward, o_reward


def load_timings_samples(env_name, method):
    assert os.path.exists(
        f"Experiments/Times_Samples/{env_name}_{method}.npz"
    ), f"Path does not exist: Experiments/Times_Samples/{env_name}_{method}.npz"
    data = np.load(f"Experiments/Times_Samples/{env_name}_{method}.npz")
    time_run = data["time_run"]
    time_opct = data["time_opct"]
    samples = data["samples"]
    return time_run, time_opct, samples


def set_boundaries(df, rows):
    boundaries = pd.DataFrame(
        np.zeros((len(rows), 2)),
        columns=["lower", "upper"],
        index=rows,
    )
    for row in rows:
        boundaries.loc[row, "lower"] = df[row].min() - 1.5 * iqr(df[row])
        boundaries.loc[row, "upper"] = df[row].max() + 1.5 * iqr(df[row])
    return boundaries


def gen_ab_samples(boundaries, num_samples):
    theta1s = np.random.uniform(
        boundaries.loc["theta1", "lower"],
        boundaries.loc["theta1", "upper"],
        num_samples,
    )
    theta2s = np.random.uniform(
        boundaries.loc["theta2", "lower"],
        boundaries.loc["theta2", "upper"],
        num_samples,
    )
    random_samples = pd.DataFrame(
        {
            "theta1_cos": np.cos(theta1s),
            "theta1_sin": np.sin(theta1s),
            "theta2_cos": np.cos(theta2s),
            "theta2_sin": np.sin(theta2s),
            "theta1_dot": np.random.uniform(
                boundaries.loc["theta1_dot", "lower"],
                boundaries.loc["theta1_dot", "upper"],
                num_samples,
            ),
            "theta2_dot": np.random.uniform(
                boundaries.loc["theta2_dot", "lower"],
                boundaries.loc["theta2_dot", "upper"],
                num_samples,
            ),
        }
    )
    return random_samples


def gen_pend_samples(boundaries, num_samples):
    theta = np.random.uniform(
        boundaries.loc["theta", "lower"], boundaries.loc["theta", "upper"], num_samples
    )
    random_samples = pd.DataFrame(
        {
            "theta_cos": np.cos(theta),
            "theta_sin": np.sin(theta),
            "omega": np.random.uniform(
                boundaries.loc["omega", "lower"],
                boundaries.loc["omega", "upper"],
                num_samples,
            ),
        }
    )
    return random_samples


def gen_cpsu_samples(boundaries, num_samples):
    angle = np.random.uniform(
        boundaries.loc["angle", "lower"], boundaries.loc["angle", "upper"], num_samples
    )
    random_samples = pd.DataFrame(
        {
            "position": np.random.uniform(
                boundaries.loc["position", "lower"],
                boundaries.loc["position", "upper"],
                num_samples,
            ),
            "velocity": np.random.uniform(
                boundaries.loc["velocity", "lower"],
                boundaries.loc["velocity", "upper"],
                num_samples,
            ),
            "angle_cos": np.cos(angle),
            "angle_sin": np.sin(angle),
            "angle_velocity": np.random.uniform(
                boundaries.loc["angle_velocity", "lower"],
                boundaries.loc["angle_velocity", "upper"],
                num_samples,
            ),
        }
    )
    return random_samples


def gen_env_samples(boundaries, num_samples):
    random_samples = pd.DataFrame(columns=boundaries.index)
    for row in boundaries.index:
        random_samples[row] = np.random.uniform(
            boundaries.loc[row, "lower"], boundaries.loc[row, "upper"], num_samples
        )
    return random_samples


def make_opct_class(samples, depth):
    X = samples.loc[:, samples.columns != "action"].to_numpy()
    y = samples["action"].to_numpy()
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    dt = spyct.Model(max_depth=depth, num_trees=1)
    dt.fit(X, y.astype(int))
    return dt


def make_opct_reg(samples, depth):
    X = samples.loc[:, samples.columns != "action"].to_numpy()
    y = samples["action"].to_numpy().reshape(-1, 1)
    dt = spyct.Model(max_depth=depth, num_trees=1)
    dt.fit(X, y)
    return dt


def make_cart_class(samples, depth):
    X = samples.loc[:, samples.columns != "action"].to_numpy()
    y = samples["action"].to_numpy()
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X, y)
    return dt


def make_cart_reg(samples, depth):
    X = samples.loc[:, samples.columns != "action"].to_numpy()
    y = samples["action"].to_numpy().reshape(-1, 1)
    dt = DecisionTreeRegressor(max_depth=depth)
    dt.fit(X, y)
    return dt


def new_samples_continuous(oracle, best_samples, different, epsilon=0.1):
    o_actions = oracle.predict(best_samples.loc[:, best_samples.columns != "action"])[0]
    if different:
        new_samples = best_samples.loc[
            np.abs(o_actions - best_samples["action"]) > epsilon
        ].drop(columns=["action"])
        new_samples["action"] = o_actions[
            np.abs(o_actions - best_samples["action"]) > epsilon
        ]
    else:
        new_samples = best_samples.copy()
        new_samples["action"] = o_actions
    return new_samples


def new_samples_discrete(oracle, best_samples, different):
    o_actions = oracle.predict(best_samples.loc[:, best_samples.columns != "action"])[0]
    if different:
        new_samples = best_samples.loc[o_actions != best_samples["action"]].drop(
            columns=["action"]
        )
        new_samples["action"] = o_actions[o_actions != best_samples["action"]]
    else:
        new_samples = best_samples.copy()
        new_samples["action"] = o_actions
    return new_samples


def get_best_class_cart(
    num_trees, c_samples_iter, depth, env, t_eval_eps, new_samples_iter
):
    best_tree, best_samples, best_reward, best_std = (
        None,
        None,
        -np.inf,
        -np.inf,
    )
    trees = np.empty(num_trees, dtype=object)
    for l in range(num_trees):
        trees[l] = make_cart_class(c_samples_iter, depth)
        tree_samples, tree_rewards = eval_cart_class(
            tree=trees[l],
            env=env,
            columns=c_samples_iter.columns,
            num_episodes=t_eval_eps,
            new_samples=new_samples_iter,
        )
        if best_tree is None or np.mean(tree_rewards) > best_reward:
            best_tree = trees[l]
            best_samples = tree_samples
            best_reward = np.mean(tree_rewards)
            best_std = np.std(tree_rewards)
    return best_tree, best_samples, best_reward, best_std


def get_best_class_opct(
    num_trees, c_samples_iter, depth, env, t_eval_eps, new_samples_iter
):
    best_tree, best_samples, best_reward, best_std = (
        None,
        None,
        -np.inf,
        -np.inf,
    )
    trees = np.empty(num_trees, dtype=object)
    for l in range(num_trees):
        trees[l] = make_opct_class(c_samples_iter, depth)
        tree_samples, tree_rewards = eval_opct_class(
            opct=trees[l],
            env=env,
            columns=c_samples_iter.columns,
            num_episodes=t_eval_eps,
            new_samples=new_samples_iter,
        )
        if best_tree is None or np.mean(tree_rewards) > best_reward:
            best_tree, best_samples, best_reward, best_std = (
                copy.deepcopy(trees[l]),
                tree_samples,
                np.mean(tree_rewards),
                np.std(tree_rewards),
            )
    return best_tree, best_samples, best_reward, best_std


def get_best_reg_cart(
    num_trees, c_samples_iter, depth, env, t_eval_eps, new_samples_iter
):
    best_tree, best_samples, best_reward, best_std = (
        None,
        None,
        -np.inf,
        -np.inf,
    )
    trees = np.empty(num_trees, dtype=object)
    for l in range(num_trees):
        trees[l] = make_cart_reg(c_samples_iter, depth)
        tree_samples, tree_rewards = eval_cart_reg(
            tree=trees[l],
            env=env,
            columns=c_samples_iter.columns,
            num_episodes=t_eval_eps,
            new_samples=new_samples_iter,
        )
        if best_tree is None or np.mean(tree_rewards) > best_reward:
            best_tree, best_samples, best_reward, best_std = (
                copy.deepcopy(trees[l]),
                tree_samples,
                np.mean(tree_rewards),
                np.std(tree_rewards),
            )
    return best_tree, best_samples, best_reward, best_std


def get_best_reg_opct(
    num_trees, c_samples_iter, depth, env, t_eval_eps, new_samples_iter
):
    best_tree, best_samples, best_reward, best_std = (
        None,
        None,
        -np.inf,
        -np.inf,
    )
    trees = np.empty(num_trees, dtype=object)
    for l in range(num_trees):
        trees[l] = make_opct_reg(c_samples_iter, depth)
        tree_samples, tree_rewards = eval_opct_reg(
            opct=trees[l],
            env=env,
            columns=c_samples_iter.columns,
            num_episodes=t_eval_eps,
            new_samples=new_samples_iter,
        )
        if best_tree is None or np.mean(tree_rewards) > best_reward:
            best_tree, best_samples, best_reward, best_std = (
                copy.deepcopy(trees[l]),
                tree_samples,
                np.mean(tree_rewards),
                np.std(tree_rewards),
            )
    return best_tree, best_samples, best_reward, best_std

def get_best_reg_opct_eps(
    num_trees, c_samples_iter, depth, env, t_eval_eps, new_samples_iter
):
    best_tree, best_samples, best_reward, best_std = (
        None,
        None,
        -np.inf,
        -np.inf,
    )
    trees = np.empty(num_trees, dtype=object)
    best_samples_eps = None
    for l in range(num_trees):
        trees[l] = make_opct_reg(c_samples_iter, depth)
        tree_samples, tree_rewards, samples_tot = eval_opct_reg_eps(
            opct=trees[l],
            env=env,
            columns=c_samples_iter.columns,
            num_episodes=t_eval_eps,
            new_samples=new_samples_iter,
        )
        if best_tree is None or np.mean(tree_rewards) > best_reward:
            best_tree, best_samples, best_reward, best_std = (
                copy.deepcopy(trees[l]),
                tree_samples,
                np.mean(tree_rewards),
                np.std(tree_rewards),
            )
            best_samples_eps = samples_tot
    return best_tree, best_samples, best_reward, best_std, best_samples_eps


def eval_opct_class_fixState(env, opct, states, num_episodes=100):
    observations = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset(start_space=states[i])
        done = False
        reward_sum = 0
        while not done:
            action = np.argmax(opct.predict(state.reshape(1, -1)), axis=1)[0]
            observations.append(state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return pd.DataFrame(observations, columns=["position", "velocity"]), rewards


def eval_opct_class(opct, env, columns, num_episodes=100, new_samples=np.inf):
    observations = []
    actions = []
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = np.argmax(opct.predict(state.reshape(1, -1)), axis=1)[0]
            observations.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    samples = pd.DataFrame(
        np.concatenate(
            (np.array(observations), np.array(actions).reshape(-1, 1)), axis=1
        ),
        columns=columns,
    )
    if new_samples < len(samples):
        samples = samples.sample(n=new_samples)
    return samples, rewards


def eval_opct_reg_fixState(env, opct, states, num_episodes=100):
    observations = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset(start_space=states[i])
        done = False
        reward_sum = 0
        while not done:
            action = opct.predict(state.reshape(1, -1))[0]
            observations.append(state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return pd.DataFrame(observations, columns=["position", "velocity"]), rewards


def eval_opct_reg(opct, env, columns, num_episodes=100, new_samples=np.inf):
    observations = []
    actions = []
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = opct.predict(state.reshape(1, -1))[0]
            observations.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)

    samples = pd.DataFrame(
        np.concatenate(
            (np.array(observations), np.array(actions).reshape(-1, 1)), axis=1
        ),
        columns=columns,
    )
    if new_samples < len(samples):
        samples = samples.sample(n=new_samples)
    return samples, rewards

def eval_opct_reg_eps(opct, env, columns, num_episodes=100, new_samples=np.inf):
    observations = []
    actions = []
    rewards = []
    samples_tot = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        samples_eps = []
        while not done:
            action = opct.predict(state.reshape(1, -1))[0]
            observations.append(state)
            samples_eps.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        samples_tot.append(samples_eps)
    samples = pd.DataFrame(
        np.concatenate(
            (np.array(observations), np.array(actions).reshape(-1, 1)), axis=1
        ),
        columns=columns,
    )
    if new_samples < len(samples):
        samples = samples.sample(n=new_samples)
    return samples, rewards, samples_tot


def eval_cart_class(tree, env, columns, num_episodes=100, new_samples=np.inf):
    observations = []
    actions = []
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = tree.predict(state.reshape(1, -1))[0]
            observations.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    samples = pd.DataFrame(
        np.concatenate(
            (np.array(observations), np.array(actions).reshape(-1, 1)), axis=1
        ),
        columns=columns,
    )
    if new_samples < len(samples):
        samples = samples.sample(n=new_samples)
    return samples, rewards


def eval_cart_reg(tree, env, columns, num_episodes=100, new_samples=np.inf):
    observations = []
    actions = []
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = tree.predict(state.reshape(1, -1))
            observations.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    samples = pd.DataFrame(
        np.concatenate(
            (np.array(observations), np.array(actions).reshape(-1, 1)), axis=1
        ),
        columns=columns,
    )
    if new_samples < len(samples):
        samples = samples.sample(n=new_samples)
    return samples, rewards


def plot_episodes(oracle_samples, opct_samples, iterations):
    fig, ax = plt.subplots(1, iterations + 1, figsize=((iterations+1)*3, 5))
    # Plot the episodes



def signum0(val):
    if val < 0:
        sgn = -1  # val=0 gets sgn 1
    else:
        sgn = 1
    return sgn


def shape_df_res(np_res, row, dimname, std):
    np_res = np_res[
        0:row, :
    ]  # remove unused rows (happens if some elements of setzero are true)

    df_res = pd.DataFrame(
        columns=["dim", "xf", "reward_mean", "reward_std", "dim_value"], data=np_res
    )
    dimname.append("thresh")
    df_res["dimname"] = [dimname[i] for i in np.int32(np_res[:, 0])]
    std2 = np.concatenate((std, [0]), axis=0)  # for last 'dimension' (thresh)
    df_res["std"] = [std2[i] for i in np.int32(np_res[:, 0])]  # just diagnostics
    df_res["dim"] = np.int32(df_res["dim"])
    return df_res


class EvalStrategy:
    def __init__(self, reward_solved, n_tree_evaluation_eps):
        self._n_tree_eval_eps = n_tree_evaluation_eps
        self.reward_solved = reward_solved

    @abstractmethod
    def eval_opct(self, env, opct, eps, sig):
        pass


class SensitivityStrategy:
    """
    This class allows each env to bind its own functions

    - ``sensitivity_analysis`` (needed by ``generate_sens_pkls``) and
    - ``multi_node_sensitivity`` (needed by ``generate_multi_sens_pkls``)
    """

    def __init__(self, eval_strategy: EvalStrategy):
        self._evs = eval_strategy

    @abstractmethod
    def sensitivity_analysis(self, inode=0, setzero=None):
        pass

    @abstractmethod
    def multi_node_sensitivity(self, inodes, setzero=None):
        pass

    def sensi_ana_V0(
        self,
        inode,
        setzero,
        dimname,
        lsize,
        env,
        best_tree,
        std,  # params for eval_opct
        ttype="class",
        print_tree=True,
    ):
        """
        Make a sensitivity analysis for node ``inode`` of OPCT in environment ``env``.

        Take each weight of ``inode`` in turn and vary it in interval [-100%, +200%] of its nominal value, while all
        the other weights stay at their nominal values.
        Evaluate the resulting tree by calculating the mean reward from self._evs._n_tree_eval_eps (=30) episodes.

        In addition, if ``setzero`` is not ``None``, set the weights for all dimensions ``j`` with ``setzero[j]=True``
        permanently to zero and exclude them from sweeping. If ``setzero==None`` (the default), no weights are excluded.

        Additionally, take the threshold of ``inode`` and vary it in interval [t - mw, t + mw], where t = nominal
        threshold value and mw = mean of all node weights, while all these node weights stay at their nominal values.

        :param inode:   node number
        :param setzero: a boolean list with length = observation space dim
        :param dimname: list with names for each observation dim
        :param lsize:   linspace size (number of weight steps)
        :param env:     the environment
        :param best_tree:  the oblique tree to investigate
        :param std:
        :param ttype:   tree type, either "class" or "regr"
        :param print_tree
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        assert not (
            best_tree.trees[0][inode].left == -1
        ), f"Error: inode={inode} does not carry a split node!"

        if print_tree:
            print_treemodel(best_tree, 0, ttype=ttype)

        if not os.path.exists("pkl"):
            os.mkdir("pkl")
        ndim = len(dimname)
        if not setzero:
            setzero = [False for _ in range(ndim)]

        np_res = np.zeros(((ndim + 1) * (lsize + 1), 5))
        row = 0
        restore_weights = np.copy(
            best_tree.trees[0][inode].split_weights.data.base[0]
        )  # important: np.copy() !
        for k in range(ndim):
            baseval = best_tree.trees[0][inode].split_weights.data.base[0][k]
            for j in range(ndim):
                if setzero[j]:
                    best_tree.trees[0][inode].split_weights.data.base[0][j] = 0

            if not setzero[k]:
                for xf in np.linspace(-1, 2, lsize):
                    dim_value = baseval * xf
                    print(
                        f"\n Setting node {inode} weight for dimension {k} ({dimname[k]}) to {dim_value:.4f} (xf={xf * 100:.2f}%):"
                    )
                    best_tree.trees[0][inode].split_weights.data.base[0][k] = dim_value
                    (
                        tree_reward_mean,
                        tree_reward_std,
                        tree_samples,
                    ) = self._evs.eval_opct(
                        env, best_tree, self._evs._n_tree_eval_eps, std
                    )
                    np_res[row, :] = [
                        k,
                        xf,
                        tree_reward_mean,
                        tree_reward_std,
                        dim_value,
                    ]
                    row = row + 1

                    print(
                        f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
                    )

            # restore the original weights
            for j in range(ndim):
                best_tree.trees[0][inode].split_weights.data.base[0][
                    j
                ] = restore_weights[j]

        # sweep over threshold t: vary in interval [t-w_bar,t+w_bar] with w_bar = mean weight of node
        w_bar = np.mean(np.abs(restore_weights))
        basethresh = best_tree.trees[0][inode].threshold
        # for xf in np.linspace(-1, 2, lsize):       # option 0: not enough variation, becouse t too small
        #    dim_value = basethresh * xf
        # for xf in np.linspace(-1, 1, lsize + 1):   # option 1: nominal t at xf=0 (different from nominal w_i at 1)
        #    dim_value = basethresh + w_bar * xf
        for xf in np.linspace(0, 2, lsize + 1):  # option 2: nominal t at xf=1
            dim_value = basethresh - w_bar + w_bar * xf
            print(
                f"\n Setting the threshold for node {inode} to {dim_value:.4f} (xf = {xf*100:.2f}%):"
            )
            best_tree.trees[0][inode].threshold = dim_value
            tree_reward_mean, tree_reward_std, tree_samples = self._evs.eval_opct(
                env, best_tree, self._evs._n_tree_eval_eps, std
            )
            np_res[row, :] = [ndim, xf, tree_reward_mean, tree_reward_std, dim_value]
            row = row + 1

            print(
                f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
            )

        # restore the original threshold:
        best_tree.trees[0][inode].threshold = basethresh

        return shape_df_res(np_res, row, dimname, std)

    def sensi_ana_V1(
        self,
        inode,
        setzero,
        dimname,
        lsize,
        env,
        best_tree,
        std,  # params for eval_opct
        ttype="class",
        print_tree=True,
        w_factor=0.04,
    ):
        """
        --- The only differences to sensi_ana_V0 so far:
                a) dim_value in weight loop with "+ signum0(b) * w_factor*w_bar/std[k] * xf"
                b) np.linspace(-1,1,...) also for weights
                c) new parameter w_factor

        Make a sensitivity analysis for node ``inode`` of OPCT in environment ``env``.

        Take each weight of ``inode`` in turn and vary it in interval [w_nom - d, w_nom + d] where w_nom is its nominal
        value and d is given by a formular in notes_pplot_sens.docx, while all the other weights stay at their nominal
        values.
        Evaluate the resulting tree by calculating the mean reward from self._evs._n_tree_eval_eps (=30) episodes.

        In addition, if ``setzero`` is not ``None``, set the weights for all dimensions ``j`` with ``setzero[j]=True``
        permanently to zero and exclude them from sweeping. If ``setzero==None`` (the default), no weights are excluded.

        Additionally, take the threshold of ``inode`` and vary it in interval [t - mw, t + mw], where t = nominal
        threshold value and mw = mean of all node weights, while all these node weights stay at their nominal values.

        :param inode:   node number
        :param setzero: a boolean list with length = observation space dim
        :param dimname: list with names for each observation dim
        :param lsize:   linspace size (number of weight steps)
        :param env:     the environment
        :param best_tree:  the oblique tree to investigate
        :param std:
        :param ttype:   tree type, either "class" or "regr"
        :param print_tree:
        :param w_factor: a common factor applied to all w-ranges
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        assert not (
            best_tree.trees[0][inode].left == -1
        ), f"Error: inode={inode} does not carry a split node!"

        if print_tree:
            print_treemodel(best_tree, 0, ttype=ttype)

        if not os.path.exists("pkl"):
            os.mkdir("pkl")
        ndim = len(dimname)
        if not setzero:
            setzero = [False for _ in range(ndim)]

        np_res = np.zeros(((ndim + 1) * (lsize + 1), 5))
        row = 0
        restore_weights = np.copy(
            best_tree.trees[0][inode].split_weights.data.base[0]
        )  # important: np.copy() !
        w_bar = np.mean(np.abs(restore_weights))
        for k in range(ndim):
            baseval = best_tree.trees[0][inode].split_weights.data.base[0][k]
            for j in range(ndim):
                if setzero[j]:
                    best_tree.trees[0][inode].split_weights.data.base[0][j] = 0

            if not setzero[k]:
                for xf in np.linspace(
                    -1, 1, lsize + 1
                ):  # why 'lsize+1'? - only with uneven number linspace includes 0
                    dim_value = (
                        baseval + signum0(baseval) * w_factor * w_bar / std[k] * xf
                    )
                    print(
                        f"\n Setting node {inode} weight for dimension {k} ({dimname[k]}) to {dim_value:.4f} (xf={xf * 100:.2f}%):"
                    )
                    best_tree.trees[0][inode].split_weights.data.base[0][k] = dim_value
                    (
                        tree_reward_mean,
                        tree_reward_std,
                        tree_samples,
                    ) = self._evs.eval_opct(
                        env, best_tree, self._evs._n_tree_eval_eps, std
                    )
                    np_res[row, :] = [
                        k,
                        xf,
                        tree_reward_mean,
                        tree_reward_std,
                        dim_value,
                    ]
                    row = row + 1

                    print(
                        f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
                    )

            # restore the original weights
            for j in range(ndim):
                best_tree.trees[0][inode].split_weights.data.base[0][
                    j
                ] = restore_weights[j]

        # sweep over threshold t: vary in interval [t-w_bar,t+w_bar] with w_bar = mean of abs weights of node
        basethresh = best_tree.trees[0][inode].threshold
        for xf in np.linspace(
            -1, 1, lsize + 1
        ):  # why 'lsize+1'? - only with uneven number linspace includes 0
            dim_value = basethresh + signum0(basethresh) * w_bar * xf
            print(
                f"\n Setting the threshold for node {inode} to {dim_value:.4f} (xf = {xf*100:.2f}%):"
            )
            best_tree.trees[0][inode].threshold = dim_value
            tree_reward_mean, tree_reward_std, tree_samples = self._evs.eval_opct(
                env, best_tree, self._evs._n_tree_eval_eps, std
            )
            np_res[row, :] = [ndim, xf, tree_reward_mean, tree_reward_std, dim_value]
            row = row + 1

            print(
                f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
            )

        # restore the original threshold:
        best_tree.trees[0][inode].threshold = basethresh

        return shape_df_res(np_res, row, dimname, std)

    def mult_node_sens_V0(
        self,
        inodes,
        setzero,
        dimname,
        lsize,
        env,
        best_tree,
        std,  # params for eval_opct
        ttype="class",
        print_tree=True,
    ):
        """
        Make a sensitivity analysis for node list ``inodes`` of OPCT in environment ``env``. In contrast to
        ``sensi_ana_V0`` which returns a data frame for a specific node, this method operates on a node list
        and returns a data frame for this list ``inodes``.

        Do for all nodes in ``inodes`` simultaneously: Take in turn for each input dim the weight and vary it in
        interval [-100%, +200%] of its nominal value, while all weights for the other dims stay at their nominal values.
        Evaluate the resulting tree by calculating the mean reward from self._evs._n_tree_eval_eps (=30) episodes.

        In addition, if ``setzero`` is not ``None``, set the weights for all dimensions ``j`` with ``setzero[j]=True``
        permanently to zero and exclude them from sweeping. If ``setzero==None`` (the default), no weight is excluded.

        Additionally, take simultaneously the thresholds of all nodes in ``inodes`` and vary them in interval
        [t - mw, t + mw], where t = nominal threshold value of each node and mw = mean of all weights of each node,
        while all node weights stay at their nominal values.

        :param inodes:  a list of node numbers
        :param setzero: a boolean list with length = observation space dim
        :param dimname: list with names for each observation dim
        :param lsize:   linspace size (number of weight steps)
        :param env:     the environment
        :param best_tree:  the oblique tree to investigate
        :param std:
        :param ttype:   tree type, either "class" or "regr"
        :param print_tree
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        for inode in inodes:
            assert not (
                best_tree.trees[0][inode].left == -1
            ), f"Error: inode={inode} does not carry a split node!"

        if print_tree:
            print_treemodel(best_tree, 0, ttype=ttype)

        if not os.path.exists("pkl"):
            os.mkdir("pkl")

        ndim = len(dimname)
        if not setzero:
            setzero = [False for i in range(ndim)]

        np_res = np.zeros(((ndim + 1) * (lsize + 1), 5))
        row = 0
        restore_weights = [
            np.copy(best_tree.trees[0][inode].split_weights.data.base[0])
            for inode in inodes
        ]
        # important: np.copy() !
        for k in range(ndim):
            baseval = [
                best_tree.trees[0][inode].split_weights.data.base[0][k]
                for inode in inodes
            ]
            for j in range(ndim):
                if setzero[j]:
                    for inode in inodes:
                        best_tree.trees[0][inode].split_weights.data.base[0][j] = 0

            if not setzero[k]:
                for xf in np.linspace(-1, 2, lsize):
                    print()
                    dim_value = 0
                    for inode in inodes:
                        dim_value = baseval[inode] * xf
                        print(
                            f" Setting node {inode} weight for dimension {k} ({dimname[k]}) to {dim_value:.4f} (xf={xf * 100:.2f}%):"
                        )
                        best_tree.trees[0][inode].split_weights.data.base[0][
                            k
                        ] = dim_value
                    (
                        tree_reward_mean,
                        tree_reward_std,
                        tree_samples,
                    ) = self._evs.eval_opct(
                        env, best_tree, self._evs._n_tree_eval_eps, std
                    )
                    np_res[row, :] = [
                        k,
                        xf,
                        tree_reward_mean,
                        tree_reward_std,
                        dim_value,
                    ]
                    row = row + 1

                    print(
                        f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
                    )

            # restore the original weights
            for inode in inodes:
                for j in range(ndim):
                    best_tree.trees[0][inode].split_weights.data.base[0][
                        j
                    ] = restore_weights[inode][j]

        # sweep over threshold t: vary in interval [t-w_bar,t+w_bar] with w_bar = mean weight of node
        w_bar = [np.mean(np.abs(restore_weights[inode])) for inode in inodes]
        basethresh = [best_tree.trees[0][inode].threshold for inode in inodes]
        # for xf in np.linspace(-1, 1, lsize + 1):   # option 1: nominal t at xf=0
        for xf in np.linspace(0, 2, lsize + 1):  # option 2: nominal t at xf=1
            print()
            dim_value = 0
            for inode in inodes:
                # dim_value = basethresh[inode] + w_bar[inode] * xf                  # option 1
                dim_value = (
                    basethresh[inode] - w_bar[inode] + w_bar[inode] * xf
                )  # option 2
                print(
                    f" Setting the threshold for node {inode} to {dim_value:.4f} (xf = {xf * 100:.2f}%):"
                )
                best_tree.trees[0][inode].threshold = dim_value
            tree_reward_mean, tree_reward_std, tree_samples = self._evs.eval_opct(
                env, best_tree, self._evs._n_tree_eval_eps, std
            )
            np_res[row, :] = [ndim, xf, tree_reward_mean, tree_reward_std, dim_value]
            row = row + 1

            if print_tree:
                print_treemodel(best_tree, 0, ttype=ttype)
            print(
                f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
            )

        # restore the original threshold:
        for inode in inodes:
            best_tree.trees[0][inode].threshold = basethresh[inode]

        return shape_df_res(np_res, row, dimname, std)

    def mult_node_sens_V1(
        self,
        inodes,
        setzero,
        dimname,
        lsize,
        env,
        best_tree,
        std,  # params for eval_opct
        ttype="class",
        print_tree=True,
        w_factor=0.04,
    ):
        """
        --- The only differences to mult_node_sens_V0 so far:
                a) dim_value in weight loop with "+ signum0(b) *w_factor*w_bar[inode]/std[k] * xf" and
                b) np.linspace(-1,1,...) also for weights
                c) new parameter w_factor

        Make a sensitivity analysis for node list ``inodes`` of OPCT in environment ``env``. In contrast to
        ``sensi_ana_V1`` which returns a data frame for a specific node, this method operates on a node list
        and returns a data frame for this list ``inodes``.

        Do for all nodes in ``inodes`` simultaneously:
        Take each weight of ``inode`` in turn and vary it in interval [w_nom - d, w_nom + d] where w_nom is its nominal
        value and d is given by a formular in notes_pplot_sens.docx, while all the other weights stay at their nominal
        values.
        Evaluate the resulting tree by calculating the mean reward from self._evs._n_tree_eval_eps (=30) episodes.

        In addition, if ``setzero`` is not ``None``, set the weights for all dimensions ``j`` with ``setzero[j]=True``
        permanently to zero and exclude them from sweeping. If ``setzero==None`` (the default), no weight is excluded.

        Additionally, take simultaneously the thresholds of all nodes in ``inodes`` and vary them in interval
        [t - mw, t + mw], where t = nominal threshold value of each node and mw = mean of all weights of each node,
        while all node weights stay at their nominal values.

        :param inodes:  a list of node numbers
        :param setzero: a boolean list with length = observation space dim
        :param dimname: list with names for each observation dim
        :param lsize:   linspace size (number of weight steps)
        :param env:     the environment
        :param best_tree:  the oblique tree to investigate
        :param std:
        :param ttype:   tree type, either "class" or "regr"
        :param print_tree:
        :param w_factor: a common factor applied to all w-ranges
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        for inode in inodes:
            assert not (
                best_tree.trees[0][inode].left == -1
            ), f"Error: inode={inode} does not carry a split node!"

        if print_tree:
            print_treemodel(best_tree, 0, ttype=ttype)

        if not os.path.exists("pkl"):
            os.mkdir("pkl")

        ndim = len(dimname)
        if not setzero:
            setzero = [False for i in range(ndim)]

        np_res = np.zeros(((ndim + 1) * (lsize + 1), 5))
        row = 0
        restore_weights = [
            np.copy(best_tree.trees[0][inode].split_weights.data.base[0])
            for inode in inodes
        ]  # important: np.copy() !
        w_bar = [np.mean(np.abs(restore_weights[inode])) for inode in inodes]
        for k in range(ndim):
            baseval = [
                best_tree.trees[0][inode].split_weights.data.base[0][k]
                for inode in inodes
            ]
            for j in range(ndim):
                if setzero[j]:
                    for inode in inodes:
                        best_tree.trees[0][inode].split_weights.data.base[0][j] = 0

            if not setzero[k]:
                for xf in np.linspace(
                    -1, 1, lsize + 1
                ):  # why 'lsize+1'? - only with uneven number linspace includes 0
                    print()
                    dim_value = 0
                    for inode in inodes:
                        b = baseval[inode]
                        dim_value = (
                            b + signum0(b) * w_factor * w_bar[inode] / std[k] * xf
                        )
                        print(
                            f" Setting node {inode} weight for dimension {k} ({dimname[k]}) to {dim_value:.4f} (xf={xf * 100:.2f}%):"
                        )
                        best_tree.trees[0][inode].split_weights.data.base[0][
                            k
                        ] = dim_value
                    (
                        tree_reward_mean,
                        tree_reward_std,
                        tree_samples,
                    ) = self._evs.eval_opct(
                        env, best_tree, self._evs._n_tree_eval_eps, std
                    )
                    np_res[row, :] = [
                        k,
                        xf,
                        tree_reward_mean,
                        tree_reward_std,
                        dim_value,
                    ]
                    row = row + 1

                    print(
                        f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
                    )

            # restore the original weights
            for inode in inodes:
                for j in range(ndim):
                    best_tree.trees[0][inode].split_weights.data.base[0][
                        j
                    ] = restore_weights[inode][j]

        # sweep over threshold t: vary in interval [t-w_bar,t+w_bar] with w_bar = mean abs weights of node
        basethresh = [best_tree.trees[0][inode].threshold for inode in inodes]
        for xf in np.linspace(
            -1, 1, lsize + 1
        ):  # why 'lsize+1'? - only with uneven number linspace includes 0
            print()
            dim_value = 0
            for inode in inodes:
                b = basethresh[inode]
                dim_value = b + signum0(b) * w_bar[inode] * xf
                print(
                    f" Setting the threshold for node {inode} to {dim_value:.4f} (xf = {xf * 100:.2f}%):"
                )
                best_tree.trees[0][inode].threshold = dim_value
            tree_reward_mean, tree_reward_std, tree_samples = self._evs.eval_opct(
                env, best_tree, self._evs._n_tree_eval_eps, std
            )
            np_res[row, :] = [ndim, xf, tree_reward_mean, tree_reward_std, dim_value]
            row = row + 1

            print(
                f"*** mean tree reward: {tree_reward_mean:.4f} ***   (solved: >= {self._evs.reward_solved})"
            )

        # restore the original threshold:
        for inode in inodes:
            best_tree.trees[0][inode].threshold = basethresh[inode]

        return shape_df_res(np_res, row, dimname, std)


def plot_sensitivity(
    df_res, pngdir, pngname, title=None, ylim=None, first_levels=None, last_levels=None
):
    """
    Plot sensitivity curves from ``df_res`` (as generated by ``sensitivity_analysis`` or ``multi_node_sensitivity``)
    and store the plot in ``pngdir/pngname``.

    :param df_res: data frame
    :param pngdir: png directory
    :param pngname: png file
    :param title: optional title string
    :param ylim: optional tuple (bottom,top) with plot y-limits
    :param first_levels: optional list of strings: level names of 'dimname' that shall appear as first levels in the plot
    :param last_levels: optional list of strings: level names of 'dimname' that shall appear as last levels in the plot
    :return:
    """
    figure, (ax1) = plt.subplots(1, sharex="all", figsize=(7.3, 6), dpi=150)
    # plt.figure(inode, figsize=(7, 4.7))

    # The following is just to ensure that the levels in column 'dimname' that appear in last_levels are really sorted
    # into the last rows of the DataFrame. This ensures that sensitivity plots with and w/o these levels have the same
    # colors in the other levels (including level 'thresh')
    data1 = df_res.copy()
    data1["dimname2"] = data1["dimname"]
    if first_levels != None:
        for level in first_levels:
            data1.loc[data1.dimname == level, "dimname2"] = "aa_" + level
    if last_levels != None:
        for level in last_levels:
            data1.loc[data1.dimname == level, "dimname2"] = "zz_" + level
    data1 = data1.sort_values(
        "dimname2"
    )  # sort such that last_levels are at end, the others alphabetically

    # level 'thresh' gets a dotted line, all other solid lines.
    lev = data1["dimname"].unique()
    nthresh = np.flatnonzero(lev == "thresh")[0]
    dashes = ["" for _ in range(len(lev))]  # solid
    dashes[nthresh] = (1, 3)  # dotted for the level of 'dimname' containing 'thresh'

    sns.lineplot(
        data=data1,
        x="xf",
        y="reward_mean",
        hue="dimname",
        style="dimname",
        dashes=dashes
        # ,palette=sns.color_palette("viridis", n_colors=4)
        ,
        ax=ax1,
    )
    ax1.legend(loc="lower right", shadow=True, fontsize=14)
    # ax1.labelsize
    plt.xlabel("factor ", fontsize=16)  # normal fontsize 12. For smaller paper figures,
    plt.ylabel("mean return", fontsize=16)  # use fontsize 14 or 16
    plt.xticks(fontsize=14)  #
    plt.yticks(fontsize=14)  #
    # plt.locator_params(axis='x', nbins=5)      # for the larger font it is better to have only 5 x-ticks (only V1)
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    if not os.path.exists(pngdir):
        os.mkdir(pngdir)
    pngname = pngdir + "/" + pngname
    print(f"Sensitivity plot saved to {pngname}")
    plt.savefig(fname=pngname, format="png")
    plt.ylim()
    # plt.show()


def generate_sens_pkls(
    ses: SensitivityStrategy, inodes, setzeros, prefix="", ylim=None, last_levels=None
):
    """
    Loop over all node numbers in ``inodes`` and all lists ``setzero`` in ``setzeros`` and call
    ``ses.sensitivity_analysis(inode,setzero)`` for them to generate data frames ``df_res``.
    All ``df_res`` for one node and different lists ``setzero`` are collected in data frame list ``df_res_lst``-

    Return a dictionary ``dd`` of data frame lists. For example, ``dd['n5'][0]`` is the sensitivity
    data frame for ``inode=5`` and ``setzeros[0]``.

    Side effect: Save each ``df_res`` in a pickle file, e.g. ``pkl/df_res_n5_1111.pkl`` and the corresponding
    sensitivity plot in ``png_sens/CP_rew_sens_n5_1111.png``, if ``prefix='CP_'``.

    :param ses: object holding a non-abstract method sensitivity_analysis
    :param inodes: list of node numbers
    :param setzeros: list of setzero lists
    :param prefix: optional prefix for png filenames, e.g. 'CP_'
    :param ylim: optional tuple (bottom,top) with plot y-limits
    :return: dictionary with inode strings as keys and lists df_res_lst as values
    """
    df_res_dict = {}
    for inode in inodes:
        str_inode = f"n{inode}"  # the key for the dict
        df_res_lst = []
        for setzero in setzeros:
            pklfile = f"pkl/df_res_{str_inode}_"
            for s in setzero:
                pklfile = f"{pklfile}{int(not s)}"
            pklfile = f"{pklfile}.pkl"  # e.g. pklfile = 'pkl/df_res_n0_101101.pkl'

            df_res = ses.sensitivity_analysis(inode, setzero)

            if not os.path.exists("pkl"):
                os.mkdir("pkl")
            df_res.to_pickle(pklfile)
            print(f"Saved results 'df_res' to {pklfile}")
            # print(df_res)

            pngfile = pklfile.replace("pkl/df_res", prefix + "rew_sens")
            pngfile = pngfile.replace(".pkl", ".png")

            plot_sensitivity(
                df_res, "png_sens/", pngfile, ylim=ylim, last_levels=last_levels
            )
            df_res_lst.append(df_res)
        inode_dict = {str_inode: df_res_lst}
        df_res_dict.update(inode_dict)

    return df_res_dict  # return dictionary of df_res, one key for each inode


def generate_multi_sens_pkls(
    ses: SensitivityStrategy,
    inodes,
    setzeros,
    prefix="",
    ylim=None,
    first_levels=None,
    last_levels=None,
):
    """
    Loop over all lists ``setzero`` in ``setzeros`` and call
    ``ses.multi_node_sensitivity(inodes, setzero)`` for them to generate a data frame ``df_res``.

    Return results in a dictionary of data frame lists. For example, ``df_res_dict['n5'][0]`` is the sensitivity
    data frame for ``inode=5`` and ``setzeros[0]``.

    Side effect: Save each ``df_res`` in a pickle file ``pkl/df_res__n-0-1-2__1111.pkl`` and the corresponding
    sensitivity plot in ``png_sens/LL_rew_sens__n-0-1-2__1111.png``, if ``inodes=[0,1,2]`` and ``prefix='LL_'``.

    :param ses: object holding a non-abstract method sensitivity_analysis
    :param inodes: list of node numbers
    :param setzeros: list of setzero lists
    :param prefix: optional prefix for png filenames, e.g. 'CP_'
    :param ylim: optional tuple (bottom,top) with plot y-limits
    :param first_levels: optional list of strings: level names of 'dimname' that shall appear as first levels in the plot
    :param last_levels: optional list of strings: level names of 'dimname' that shall appear as last levels in the plot
    :return: dictionary with inode strings as keys and df_res_lst as values
    """
    df_res_dict = {}
    str_inode = "n"
    for inode in inodes:
        str_inode = f"{str_inode}-{inode}"  # the key for the dict

    df_res_lst = []
    for setzero in setzeros:
        pklfile = f"pkl/df_res__{str_inode}__"
        for s in setzero:
            pklfile = f"{pklfile}{int(not s)}"
        pklfile = f"{pklfile}.pkl"  # e.g. pklfile = 'pkl/df_res__n-0-1-2__101101.pkl'

        df_res = ses.multi_node_sensitivity(inodes, setzero)

        if not os.path.exists("pkl"):
            os.mkdir("pkl")
        df_res.to_pickle(pklfile)
        print(f"Saved results 'df_res' to {pklfile}")
        # print(df_res)

        pngfile = pklfile.replace("pkl/df_res", prefix + "rew_sens")
        pngfile = pngfile.replace(".pkl", ".png")

        plot_sensitivity(
            df_res,
            "png_sens/",
            pngfile,
            ylim=ylim,
            first_levels=first_levels,
            last_levels=last_levels,
        )
        df_res_lst.append(df_res)
    inode_dict = {str_inode: df_res_lst}
    df_res_dict.update(inode_dict)

    return df_res_dict  # return dictionary of df_res, one key for each inode


def generate_sens_plots(prefix="", ylim=None, first_levels=None, last_levels=None):
    """
    Loop over all pickle files in ``pkl/``, load their contents ``df_res`` and generate the sensitivity plots in
    directory ``png_sens/``.

    :param prefix: optional prefix for png filenames, e.g. 'CP_'
    :param ylim: optional tuple (bottom,top) with plot y-limits
    :param first_levels: optional list of strings: level names of 'dimname' that shall appear as first levels in the plot
    :param last_levels: optional list of strings: level names of 'dimname' that shall appear as last levels in the plot
    :return: dictionary with pkl filenames as keys and lists [df_res] as values
    """
    df_res_dict = {}
    files = os.listdir("pkl")
    for pklname in files:
        pklfile = "pkl/" + pklname
        df_res = pd.read_pickle(pklfile)

        pngfile = pklfile.replace("pkl/df_res", prefix + "rew_sens")
        pngfile = pngfile.replace(".pkl", ".png")

        plot_sensitivity(
            df_res,
            "png_sens/",
            pngfile,
            ylim=ylim,
            first_levels=first_levels,
            last_levels=last_levels,
        )

        inner_dict = {pklname: [df_res]}
        df_res_dict.update(inner_dict)

    return df_res_dict  # return dictionary of [df_res], one key for each file name


def print_tree(tree, inode=0, ttype="class"):
    """
     Print the subtree starting at node ``inode`` .

    :param tree: oblique tree (OPCT)
    :param inode: node numbe (default 0 = root node)
    :param ttype: tree type, either "class" or "regr"
    :return:
    """
    """

    ttype: 
    """
    node = tree[inode]
    if node.left != -1:
        print_node_rule(inode, node, " < ")
        print_tree(tree, node.left, ttype)
        # print_node_rule(inode,node," >=")
        print(
            "{space}node={inode}, else: ".format(space=node.depth * "|-", inode=inode)
        )
        print_tree(tree, node.right, ttype)
    else:
        if ttype == "class":
            leaftext = "class"
            prototype = np.argmax(np.asarray(node.prototype))
        else:
            leaftext = "regval"
            prototype = np.asarray(node.prototype)[0]
        print(
            "{space}node={inode}, {leaftext}: {proto}".format(
                space=node.depth * "|-", inode=inode, leaftext=leaftext, proto=prototype
            )
        )


def print_treemodel(model, inode=0, ttype="class"):
    return print_tree(model.trees[0], inode, ttype)


def print_node_rule(inode, node, comp):
    print(
        "{space}node={inode}, rule: {weights} * X {comp} {thresh:.4f}".format(
            space=node.depth * "|-",
            inode=inode,
            weights=node.split_weights.to_ndarray()[0],
            comp=comp,
            thresh=node.threshold,
        )
    )


def get_val_from_node(x, opct, inode):
    """
    Return the value(s) that non-leaf node ``inode`` assigns to observation(s) ``x``.

    :param x: np.array(dim_obs), a single observation, or np.array((num_obs,dim_obs)), an array of observations.
    :param opct: an oblique tree (package spyct)
    :param inode: a non-leaf node number
    :returns: the value (array of values) that node inode assigns to x.
    """
    node = opct.trees[0][inode]
    assert node.left != -1, "Error: inode specifies a leaf node"
    weights = node.split_weights.to_ndarray()[0]
    thresh = node.threshold
    val = np.dot(x, weights) - thresh
    return val


def get_dist_from_nodeplane(x, opct, inode):
    """
    Return the distance of all observations in ``x`` to the plane of non-leaf node ``inode`` in
    form of a distance array of shape (num_obs,1).

    :param x: np.array(dim_obs), a single observation, or np.array((num_obs,dim_obs)), an array of observations
    :param opct: an oblique tree (package spyct)
    :param inode: a non-leaf node number
    :return: the distance(s) for observation(s) x, shape (num_obs,1)
    """
    node = opct.trees[0][inode]
    assert node.left != -1, "Error: inode specifies a leaf node"
    weights = node.split_weights.to_ndarray()[0]
    thresh = node.threshold
    dist = (np.dot(x, weights) - thresh) / np.linalg.norm(weights)
    return dist


def get_dist_from_point(x, pnt):
    """
    Return the distance of all observations in ``x`` to the plane of non-leaf node ``inode`` in
    form of a distance array of shape (num_obs,1).

    :param x: np.array(dim_obs), a single observation, or np.array((num_obs,dim_obs)), an array of observations
    :param opct: an oblique tree (package spyct)
    :param inode: a non-leaf node number
    :return: the distance(s) for observation(s) x, shape (num_obs,1)
    """
    delta = x - pnt
    dist = np.sum(delta * delta, axis=1)
    dist = dist**0.5
    return dist


def traverse_tree(obs, inode: int, inode_active, opct, ttype: str, tnode=0):
    """
    Traverse the subtree starting at node ``tnode`` (default: 0 = root node) with observation ``obs``, i.e. route
    observation obs down the tree, ultimately reaching the leaf node ``node``.

    :param inode: a node number to pass through
    :param ttype: "class" or "regr"
    :returns: the leaf's prototype if obs passes through (activates) node inode. The prototype is argmax(node.prototype),
        if ttype=="class" (classification tree) and node.prototype else (regression tree).
        If obs does not activate inode, return "NaN".
    """
    if tnode == inode:
        inode_active = 1
    node = opct.trees[0][tnode]
    if node.left != -1:
        v = get_val_from_node(obs, opct, tnode)
        if v < 0:
            return traverse_tree(obs, inode, inode_active, opct, ttype, node.left)
        else:
            return traverse_tree(obs, inode, inode_active, opct, ttype, node.right)
    else:  # we have reached a leaf
        if inode_active:
            if ttype == "class":
                return np.argmax(np.asarray(node.prototype))
            else:
                return np.asarray(node.prototype)[0]
        else:
            return float("NaN")


def obs_with_inode_active(x, opct, inode, ttype="class"):
    """
    Find the observations that activate ``inode``.

    We get with
        ``1 - np.isnan(obs_with_inode_active)``
    an array which is 1 at every active observation and 0 at every inactive observation.

    :param x: np.array((num_obs,dim_obs)), an array of observations
    :param opct: an oblique tree (package spyct)
    :param inode: a non-leaf node number
    :param ttype: "class" or "regr"
    :return: an array of shape (num_obs) where the i-th entry is the prototype of the leaf where the  i-th observation
            is routed *** if *** it passes through (activates) inode.
            The i-th entry is "NaN" if the i-th observation does not activate inode.
    """
    return np.array(
        [traverse_tree(x[i, :], inode, 0, opct, ttype) for i in range(x.shape[0])]
    )


def get_child_p(child_num, opct, ttype):
    c_node = opct.trees[0][child_num]
    if c_node.left == -1:
        if ttype == "class":
            c_proto = np.argmax(np.asarray(c_node.prototype))
        else:
            c_proto = np.asarray(c_node.prototype)[0]
    else:
        c_proto = float("NaN")

    return c_proto


def get_proto_childs(inode, opct, ttype):
    """
    Return a dictionary {"left", "right"}, where

    - "left" is the prototype of inode's left child (if it is a leaf, else NaN),
    - "right" is the prototype of inode's right child (if it is a leaf, else NaN).
    If ttype=="class", the prototype is the class with the highest probability (in array 'prototype').
    If ttype=="regr", the prototype is the regression value of this leaf ('prototype[0]').
    """
    node = opct.trees[0][inode]
    lproto = get_child_p(node.left, opct, ttype)
    rproto = get_child_p(node.right, opct, ttype)
    return {"left": lproto, "right": rproto}


def get_weighted_protos(inode, dist, leafp, n_act_below, n_act_above):
    """
    Return a dictionary {"left", "right"}, where

    - "left" is the mean prototype of all observations routed through inode's left child,
    - "right" is the mean prototype of all observations routed through inode's right child.
    The prototypes are passed in via leafp = samples[leafpcol] (which is NaN at all observations with inode inactive.
    n_act_below and n_act_above are only needed for assertion checks.
    """
    left_leafp = leafp[np.flatnonzero(dist < 0)]
    left_leafp = left_leafp[
        np.flatnonzero(~np.isnan(left_leafp))
    ]  # strip the NaNs for the following assertion:
    assert left_leafp.shape[0] == n_act_below, f"Error n_act_below at inode {inode}"
    left_wp = np.mean(left_leafp)
    right_leafp = leafp[np.flatnonzero(dist >= 0)]
    right_leafp = right_leafp[np.flatnonzero(~np.isnan(right_leafp))]
    assert right_leafp.shape[0] == n_act_above, f"Error n_act_above at inode {inode}"
    right_wp = np.mean(right_leafp)
    return {"left": left_wp, "right": right_wp}


def get_scaled_dist_fac_for_node(sig, opct, inode):
    """
    return the factor f that relates the distance d' in scaled space to unscaled distance d = f * d'.
    Scaling is done by point-wise division of np.array X with np.array sig. sig[i] is usually the standard
    deviation of dimension i in the unscaled data.
    """
    node = opct.trees[0][inode]
    assert node.left != -1, "Error: inode specifies a leaf node"
    weights = node.split_weights.to_ndarray()[0]
    f = np.linalg.norm(np.multiply(weights, sig)) / np.linalg.norm(weights)
    return f


def append_dist_cols(inode, opct, samples, observations, ttype="class", epsilon=0.0):
    """
    append to ``samples`` the extra columns ``done, node*, delta*, plane*, leafp*, activ*``, where ``* = inode``.
    ``node*`` is the (signed) distance of each observation from the separating hyperplane of ``inode``.
    ``delta*`` is |next_distance| - |distance| for ``inode``. ``delta0`` <0: attract, >0: repell.
    """
    done_arr = samples["done"].values
    dist = np.float64(get_dist_from_nodeplane(observations, opct, inode))
    delta = (np.abs(np.roll(dist, -1)) - np.abs(dist)) * 1000
    delta[np.flatnonzero(np.array(done_arr) == True)] = float("NaN")
    delta = np.float64(delta)
    nodecol = f"node{inode:02d}"
    deltacol = f"delta{inode:02d}"
    planecol = f"plane{inode:02d}"
    leafpcol = f"leafp{inode:02d}"
    activcol = f"activ{inode:02d}"
    samples[nodecol] = dist
    samples[deltacol] = delta
    samples[planecol] = 0  # "attract"
    samples.loc[
        (np.sign(samples[deltacol]) > 0) & (np.abs(dist) >= epsilon), planecol
    ] = 1  # "repell"
    # if epsilon > 0: a sample with absolute distance < epsilon is *always* considered as 0 ('attract')
    # if epsilon = 0: only the deltacol-part is relevant
    samples[leafpcol] = obs_with_inode_active(observations, opct, inode, ttype)
    samples[activcol] = 1 - np.isnan(samples[leafpcol])


def remove_dist_cols(inode, samples):
    nodecol = f"node{inode:02d}"
    deltacol = f"delta{inode:02d}"
    planecol = f"plane{inode:02d}"
    leafpcol = f"leafp{inode:02d}"
    activcol = f"activ{inode:02d}"
    samples = samples.drop([nodecol, deltacol, planecol, leafpcol, activcol], axis=1)
    return samples


def cut_episodes(samples):
    done_rows = np.where(samples["done"].to_numpy() == True)[0]
    epi_start = [0] + [done_rows[k] + 1 for k in range(len(done_rows) - 1)]
    epi_end = done_rows.tolist()
    return epi_start, epi_end


def plot_node_plane(plt, node, inode, xdata, ydata):
    """
    Given a plot context 'plt' with certain 'xdata','ydata', plot on top a red line, the separating plane of
    OPCT node 'node' with index 'inode'.
    """
    assert node.left != -1, f"Error: inode={inode} specifies a leaf node"
    weights = node.split_weights.to_ndarray()[0]
    thresh = node.threshold
    ymin = min(ydata)
    ymax = max(ydata)
    pmin = min(xdata)
    pmax = max(xdata)
    plt.plot([0, 0], [ymin, ymax], c="black", linestyle="dashed", linewidth=0.5)
    plt.plot([pmin, pmax], [0, 0], c="black", linestyle="dashed", linewidth=0.5)
    if weights[1] == 0:
        pmin = pmax = thresh / weights[0]
        plt.plot([pmin, pmax], [ymin, ymax], c="red")
    else:
        pstep = 50  # Why 50 points? - Seems overkill, but for very steep lines we might otherwise miss the part
        # where the line passes through the data
        prange = np.linspace(min(xdata), max(xdata), pstep)
        vrange = (thresh - weights[0] * prange) / weights[1]
        ind = np.flatnonzero((ymin <= vrange) & (vrange <= ymax))
        plt.plot(prange[ind], vrange[ind], c="red")


def check_arctan2():
    th_deg = np.arange(-180, 180)
    th = np.float64(
        th_deg * np.pi / 180
    )  # if the argument were not float64, the conversion is important for numeric
    # accuracy (float32 will not work!)
    sinth = np.sin(th)
    costh = np.cos(th)
    th_arc = np.arctan2(sinth, costh)

    theta = np.arcsin(sinth)
    ind = np.flatnonzero((costh < 0) & (sinth > 0))
    theta[ind] = np.pi - theta[ind]
    ind = np.flatnonzero((costh < 0) & (sinth <= 0))
    theta[ind] = -np.pi - theta[ind]
    samples = pd.DataFrame(  # just to conveniently view the data in debugger
        {
            "costh": costh,
            "sinth": sinth,
            "th_deg": th_deg,
            "th": th,
            "th_arc": th_arc,
            "theta": theta,
        }
    )
    err_arc = max(np.abs(th - th_arc))
    err_the = max(np.abs(th - theta))
    assert err_arc < 1e-8, f"Error th_arc: {err_arc}"
    assert err_the < 1e-8, f"Error theta: {err_the}"
    print("check_arctan2 OK")
