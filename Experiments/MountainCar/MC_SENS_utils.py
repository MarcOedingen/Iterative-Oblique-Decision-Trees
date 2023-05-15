import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Experiments.utils as e_utils
from stable_baselines3.common.monitor import Monitor


class MyMonitor(Monitor):
    def __init__(self, env, filename=None):
        super().__init__(env, filename)

    def step(self, action):
        s, r, done, info = super().step(action)
        if done:
            sum_epi = sum(super().get_episode_lengths())
            print(
                f"MyMonitor: sum(epi_lengths)={sum_epi}, total steps = {super().get_total_steps()} "
            )

        return s, r, done, info


class MC_EvalStrategy(e_utils.EvalStrategy):
    def __init__(self, reward_solved, n_tree_evaluation_eps):
        super().__init__(reward_solved, n_tree_evaluation_eps)

    def eval_opct(self, env, opct, eps, sig):
        """
        Evaluate OPCT in environment over a number of episodes

        :param env:  environment
        :param opct: oblique tree
        :param eps: number of evaluation episodes
        :param sig: only needed for diagnostic printout (scaled dist factor)
        :return: avg(return), std(return), samples
        """
        m = np.zeros(eps)
        positions = []
        velocities = []
        done_arr = []
        actions = []
        for e in range(eps):
            s = env.reset()
            done = False
            while not done:
                action = np.argmax(opct.predict(s.reshape(1, -1)), axis=1)[0]
                positions.append(s[0])
                velocities.append(s[1])
                done_arr.append(done)
                actions.append(action)
                s, r, done, _ = env.step(action)
                m[e] += r
            # append the final state (done=True) to separate episodes:
            positions.append(s[0])
            velocities.append(s[1])
            done_arr.append(done)
            actions.append(-1)

        f0 = e_utils.get_scaled_dist_fac_for_node(sig, opct, 0)
        print("global scaled dist factor for node 0: ", f0)

        samples = pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "done": done_arr,
                "action": actions,
            }
        )
        return np.mean(m), np.std(m), samples

    def eval_opct_monitor(self, env, opct, eps, sig):
        """
        like self.eval_opct, but with a Monitor wrapper. Just to test Monitor wrapper
        """
        mon_env = MyMonitor(env, filename="test_")  # logs to "test_monitor.csv"

        mean_m, std_m, samples = self.eval_opct(mon_env, opct, eps, sig)

        assert (
            sum(mon_env.get_episode_lengths()) == mon_env.get_total_steps()
        ), "sum(episode lengths) != total steps"
        mon_env.close()

        return mean_m, std_m, samples


def plot_attract_mc(inode, opct, samples, pngfile, ttype):
    """
    plot for each observation point whether it is attracted to or repelled from node inode's separating plane.
    In addition, draw the separating hyperplane as red line.
    """
    observations = samples.values[:, 0:2]  # columns "position", "velocity" as np.array
    e_utils.append_dist_cols(inode, opct, samples, observations)
    deltacol = f"delta{inode:02d}"
    planecol = f"plane{inode:02d}"
    leafpcol = f"leafp{inode:02d}"
    activcol = f"activ{inode:02d}"
    plt.figure(
        inode, figsize=(7, 4.7)
    )  # why inode? - each node should start a new figure
    sns.scatterplot(
        data=samples,
        x="position",
        y="velocity",
        size=activcol,
        sizes=(40, 10),
        hue=planecol,
        palette=None,
    )  # "Set2")#
    plt.legend(loc="lower right")
    node = opct.trees[0][inode]
    e_utils.plot_node_plane(plt, node, inode, samples.position, samples.velocity)
    #
    # annotate how many samples are routed through inode [in total and with node value (below/above) thresh]
    #
    activ = samples[activcol].values
    plane = samples[planecol].values
    dist = samples[f"node{inode:02d}"].values
    n_act = sum(activ)
    n_act_below = sum(activ[np.flatnonzero(dist < 0)])
    n_act_above = sum(activ[np.flatnonzero(dist >= 0)])
    n_act_attract = sum(activ[np.flatnonzero(plane == 0)])
    n_act_repell = sum(activ[np.flatnonzero(plane == 1)])
    pmin = min(samples.position)
    vmax = max(samples.velocity)
    plt.text(
        pmin,
        0.95 * vmax,
        f"active: {n_act} ({n_act_below}, {n_act_above}) [a:{n_act_attract}, r:{n_act_repell}]",
        horizontalalignment="left",
        fontsize=9,
    )
    #
    # calculate and print the 25%/50%/75%-quantile of all observations routed through inode (disregarding NaN values)
    #
    # avg_activ_delta = np.nanmean(samples[deltacol].values[np.flatnonzero(activ>0)])
    # print(f"Node {inode}: avg_active_delta * 1000 = {avg_activ_delta:.4f} with {n_act} active observations")
    quant_activ_delta = np.nanquantile(
        samples[deltacol].values[np.flatnonzero(activ > 0)], [0.25, 0.50, 0.75]
    )
    np.set_printoptions(precision=4)
    print(
        f"Node {inode}: quant_active_delta * 1000 = {quant_activ_delta} with {n_act} active observations"
    )
    np.set_printoptions(precision=8)  # the default
    #
    # annotate the prototype (below/above) thresh, if inode has leaves as children (NaN, if child is non-leaf):
    #
    protos = e_utils.get_proto_childs(inode, opct, ttype)
    wproto = e_utils.get_weighted_protos(
        inode, dist, samples[leafpcol].values, n_act_below, n_act_above
    )
    plt.text(
        pmin,
        0.85 * vmax,
        f"proto: ({protos['left']:.3f}, {protos['right']:.3f})",
        horizontalalignment="left",
        fontsize=9,
    )
    plt.text(
        pmin,
        0.77 * vmax,
        f"[{wproto['left']:.3f}, {wproto['right']:.3f}]",
        horizontalalignment="left",
        fontsize=9,
    )

    plt.savefig(pngfile, format="png", dpi=150)
    # plt.show()
    samples = e_utils.remove_dist_cols(inode, samples)
    return samples


def plot_all_attract_mc(opct, samples, ttype="class"):
    if os.path.exists("png"):
        shutil.rmtree(
            "png"
        )  # delete all files in png/ (otherwise a prior 'plane14.png' could remain)
    os.mkdir("png")
    for inode in range(opct.trees[0].size):
        node = opct.trees[0][inode]
        if node.left != -1:  # if it is not a leaf node:
            pngfile = f"png/plane{inode:02d}.png"
            samples = plot_attract_mc(inode, opct, samples, pngfile, ttype)


def append_multi_dist_cols(inodes, opct, samples, observations):
    # done_arr = samples["done"].values
    for inode in inodes:
        dist = np.float64(e_utils.get_dist_from_nodeplane(observations, opct, inode))
        distcol = f"dist{inode:02d}"
        samples[distcol] = dist
    colors = ["b", "y", "r", "k"]
    clrs = [colors[k] for k in samples["action"].to_numpy()]
    samples["clrs"] = clrs
    actnames = ["left", "none", "right"]
    acts = [actnames[k] for k in samples["action"].to_numpy()]
    samples["actname"] = acts


def plot_distance_mc(inodes, opct, samples, pngdir, pngbase, prefix="MC_", ttype=None):
    if not os.path.exists(pngdir):
        os.mkdir(pngdir)
    observations = samples.values[:, 0:2]  # columns 'position', 'velocity'  as np.array
    append_multi_dist_cols(inodes, opct, samples, observations)
    epi_start, epi_end = e_utils.cut_episodes(samples)
    for epi in range(len(epi_start)):
        sample1 = pd.DataFrame(
            samples[epi_start[epi] : epi_end[epi]]
        )  # make a copy, not a slice
        nrow = sample1.shape[0]
        sample1["index"] = range(nrow)
        legend = ["auto", False, False]
        fig, ax = plt.subplots(2, 1, sharex="all", figsize=(10, 8))
        hue_order = ["left", "none", "right"]
        data1 = sample1.sort_values("actname", key=np.vectorize(hue_order.index))
        # it is important to sort the data for Seaborn's hue_order, otherwise 'main' may get different colors in
        # different plots
        for i in range(len(inodes)):
            inode = inodes[i]
            ax[i].plot(range(nrow), np.zeros(nrow), c="0.7", lw=0.5)

            # --- matplotlib version: has no easy way to legend ---
            # ax[i].scatter(sample1["index"], sample1[f"dist{inode:02d}"], c=sample1["clrs"].values, s=4)

            # --- seaborn version: legend + nicer points ---
            sns.scatterplot(
                data=data1,
                x="index",
                y=f"dist{inode:02d}",
                hue="actname",
                hue_order=hue_order,
                palette=None,
                alpha=0.5,  # transparency, to see 'points behind'
                sizes=(10, 10),
                legend=legend[i],
                ax=ax[i],
            )
        filename = f"{pngdir}/{prefix}{pngbase}_{epi:02d}.png"
        plt.savefig(filename, format="png", dpi=150)
        print(f"Saved distance plot to {filename}")
        plt.close()
