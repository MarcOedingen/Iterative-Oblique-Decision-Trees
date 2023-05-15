import os
import copy
import spyct
import pickle
import numpy as np
import pandas as pd
from scipy.stats import iqr
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

if __name__ == "__main__":
    data = np.load("Paper_Results_v2/Rewards_NPZ/CartPole-v1_CART.npz")
    time_run = data["o_reward"]
    print(time_run)


def save_results(env_name, t_reward, o_reward, method):
    if not os.path.exists("Paper_Results_v2/Rewards_NPZ"):
        os.makedirs("Paper_Results_v2/Rewards_NPZ")
    np.savez_compressed(
        f"Paper_Results_v2/Rewards_NPZ/{env_name}_{method}",
        t_reward=t_reward,
        o_reward=o_reward,
    )


def save_best_opct(best_opcts, best_rewards, reward_threshold, env_name, method):
    if not os.path.exists("Paper_Results_v2/OPCTs"):
        os.makedirs("Paper_Results_v2/OPCTs")
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
        f"Paper_Results_v2/OPCTs/{env_name}_{method}_depth_{solved_depth+1}.pkl", "wb"
    ) as f:
        pickle.dump(best_opct, f)

def save_best_cart(best_carts, best_rewards, reward_threshold, env_name, method):
    if not os.path.exists("Paper_Results_v2/CARTs"):
        os.makedirs("Paper_Results_v2/CARTs")
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
        f"Paper_Results_v2/CARTs/{env_name}_{method}_depth_{solved_depth+1}.pkl", "wb"
    ) as f:
        pickle.dump(best_cart, f)


def save_timings_samples(time_run, time_opct, samples, env_name, method):
    if not os.path.exists("Paper_Results_v2/Times_Samples"):
        os.makedirs("Paper_Results_v2/Times_Samples")
    np.savez_compressed(
        f"Paper_Results_v2/Times_Samples/{env_name}_{method}",
        time_run=time_run,
        time_opct=time_opct,
        samples=samples,
    )


def load_results(env_name, method):
    assert os.path.exists(
        f"Paper_Results_v2/Rewards_NPZ/{env_name}_{method}.npz"
    ), f"Path does not exist: Paper_Results_v2/Rewards_NPZ/{env_name}_{method}.npz"
    data = np.load(f"Paper_Results_v2/Rewards_NPZ/{env_name}_{method}.npz")
    t_reward = data["t_reward"]
    o_reward = data["o_reward"]
    return t_reward, o_reward


def load_timings_samples(env_name, method):
    assert os.path.exists(
        f"Paper_Results_v2/Times_Samples/{env_name}_{method}.npz"
    ), f"Path does not exist: Paper_Results_v2/Times_Samples/{env_name}_{method}.npz"
    data = np.load(f"Paper_Results_v2/Times_Samples/{env_name}_{method}.npz")
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
    return pd.DataFrame(observations, columns=['position', 'velocity']), rewards

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
    return pd.DataFrame(observations, columns=['position', 'velocity']), rewards


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
