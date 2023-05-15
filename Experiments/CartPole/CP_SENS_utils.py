import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Experiments.utils as e_utils


class CP_EvalStrategy(e_utils.EvalStrategy):
    def __init__(self, reward_solved, n_tree_evaluation_eps):
        super().__init__(reward_solved, n_tree_evaluation_eps)

    def eval_opct(self, env, opct, eps, sig, samples_only=np.inf):
        """
        Evaluate OPCT in environment over a number of episodes

        :param env:  environment
        :param opct: oblique tree
        :param eps: number of evaluation episodes
        :param sig: only needed for diagnostic printout (scaled dist factor)
        :param samples_only: if =100, then put only the first 100 steps (records) of each episode into ``samples``
        :return: avg(return), std(return), samples
        """
        m = np.zeros(eps)
        positions = []
        velocities = []
        angles = []
        angle_vels = []
        done_arr = []
        actions = []
        for e in range(eps):
            s = env.reset()
            istep = 0
            if type(s) is tuple: s = s[0]       # gym-0.26.2 returns a tuple (was nd.array before)
            done = False
            while not done:
                action = np.argmax(opct.predict(s.reshape(1, -1)), axis=1)[0]
                if istep < samples_only:
                    positions.append(s[0])
                    velocities.append(s[1])
                    angles.append(s[2])
                    angle_vels.append(s[3])
                    done_arr.append(done)
                    actions.append(action)
                s, r, done, _ = env.step(action)
                m[e] += r
                istep += 1
            # append the final state (done=True) to separate episodes:
            positions.append(s[0])
            velocities.append(s[1])
            angles.append(s[2])
            angle_vels.append(s[3])
            done_arr.append(done)
            actions.append(-1)

        f0 = e_utils.get_scaled_dist_fac_for_node(sig, opct, 0)
        print("global scaled dist factor for node 0: ", f0)

        samples = pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "angle": angles,
                "angle_v": angle_vels,
                "done": done_arr,
                "action": actions,
            }
        )
        return np.mean(m), np.std(m), samples

def plot_attract_cp(inode, opct, samples, pngfile, ttype):
    """
    plot for each observation point whether it is attracted to or repelled from node inode's separating plane.
    In addition, draw the separating hyperplane as red line.
    """
    observations = samples.values[:, 0:4]  # columns "position", "velocity", "angle", "angle_v" as np.array
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
        x="angle",
        y="angle_v",
        size=activcol,
        sizes=(10, 10),
        hue=planecol,
        palette=None,
    )  # "Set2")#
    plt.legend(loc="lower right")
    node = opct.trees[0][inode]
    e_utils.plot_node_plane(plt, node, inode, samples.angle, samples.angle_v)
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
    pmin = min(samples.angle)
    vmax = max(samples.angle_v)
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
    plt.close(inode)
    plot_v_delta(inode, samples)
    samples = e_utils.remove_dist_cols(inode, samples)
    return samples

def plot_v_delta(inode, samples):
    offset = 50
    pngfile = f"png/v_delta{inode:02d}.png"
    plt.figure(
        inode+offset, figsize=(7, 4.7)
    )  # why inode? - each node should start a new figure
    sns.scatterplot(
        data=samples,
        x="velocity",
        y=f"delta{inode:02d}",
        # size=activcol,
        sizes=(10, 10),
        # hue=planecol,
        alpha=0.25,
        palette=None,
    )  # "Set2")#
    plt.savefig(pngfile, format="png", dpi=150)
    plt.close(inode+offset)


def plot_all_attract_cp(opct, samples, ttype="class"):
    if os.path.exists("png"):
        shutil.rmtree(
            "png"
        )  # delete all files in png/ (otherwise a prior 'plane14.png' could remain)
    os.mkdir("png")
    for inode in range(opct.trees[0].size):
        node = opct.trees[0][inode]
        if node.left != -1:  # if it is not a leaf node:
            pngfile = f"png/plane{inode:02d}.png"
            samples = plot_attract_cp(inode, opct, samples, pngfile, ttype)

def append_multi_dist_cols(inodes, opct, samples, observations):
    done_arr = samples["done"].values
    for inode in inodes:
        dist = np.float64(e_utils.get_dist_from_nodeplane(observations, opct, inode))
        distcol = f"dist{inode:02d}"
        samples[distcol] = dist
    colors = ['b',  'r', 'k']
    clrs = [ colors[k] for k in samples["action"].to_numpy() ]
    samples["clrs"] = clrs
    actnames = ['left', 'right']
    acts = [ actnames[k] for k in samples["action"].to_numpy() ]
    samples["actname"] = acts
    #samples["index"] = range(len(acts))
    samples["dummy"] = 1

def plot_distance_cp(inodes, opct, samples, pngdir, pngbase, prefix='CP_', ttype=None):
    # if os.path.exists(pngdir):
    #     shutil.rmtree(
    #         pngdir
    #     )  # delete all files in pngdir/ (otherwise a prior 'plane14.png' could remain)
    if not os.path.exists(pngdir):
        os.mkdir(pngdir)
    observations = samples.values[:, 0:4]  # columns 'position', 'velocity', 'angle, 'angle_velocity'
    append_multi_dist_cols(inodes, opct, samples, observations)
    epi_start, epi_end = e_utils.cut_episodes(samples)
    for epi in range(len(epi_start)):
        sample1 = pd.DataFrame(samples[epi_start[epi]:epi_end[epi]])        # make a copy, not a slice
        nrow = sample1.shape[0]
        sample1["index"] = range(nrow)
        legend = ['auto', False, False]
        fig, ax = plt.subplots(3, 1, sharex="all", figsize=(10, 8))
        hue_order = ['left', 'none', 'right']
        data1 = sample1.sort_values('actname', key=np.vectorize(hue_order.index))
        # it is important to sort the data for Seaborn's hue_order, otherwise 'main' may get different colors in
        # different plots
        for i in range(len(inodes)):
            inode = inodes[i]
            ax[i].plot(range(nrow), np.zeros(nrow), c='0.7', lw=0.5)

            #--- matplotlib version: has no easy way to legend ---
            # ax[i].scatter(sample1["index"], sample1[f"dist{inode:02d}"], c=sample1["clrs"].values, s=4)

            #--- seaborn version: legend + nicer points ---
            sns.scatterplot(
                data=data1,
                x="index",
                y=f"dist{inode:02d}",
                hue="actname",
                hue_order=hue_order,
                palette=None,
                alpha=0.5,              # transparency, to see 'points behind'
                #size="dummy",
                sizes=(10, 10),
                legend=legend[i],
                ax=ax[i]
            )
        filename = f"{pngdir}/{prefix}{pngbase}_{epi:02d}.png"
        plt.savefig(filename, format="png", dpi=150)
        print(f"Saved distance plot to {filename}")
        plt.close()
    #plt.show()
