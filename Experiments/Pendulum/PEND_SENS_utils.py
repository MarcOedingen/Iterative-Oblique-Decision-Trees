import os
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Experiments.utils as e_utils


class Pend_EvalStrategy(e_utils.EvalStrategy):
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
        theta_cos = []
        theta_sin = []
        omegas = []
        done_arr = []
        actions = []
        for e in range(eps):
            s = env.reset()
            done = False
            ret = 0
            while not done:
                obs = np.array(s).reshape(1, -1)
                action = opct.predict(obs)[0][0]
                theta_cos.append(s[0])
                theta_sin.append(s[1])
                omegas.append(s[2])
                done_arr.append(done)
                actions.append(action)
                s, r, done, _ = env.step([action])
                ret += r
            m[e] = ret
            # append the final state (done=True) to separate episodes:
            theta_cos.append(s[0])
            theta_sin.append(s[1])
            omegas.append(s[2])
            done_arr.append(done)
            actions.append(-1)

        f0 = e_utils.get_scaled_dist_fac_for_node(sig, opct, 0)
        f1 = e_utils.get_scaled_dist_fac_for_node(sig, opct, 1)
        print("global scaled dist factor for node 0: ", f0)
        print("global scaled dist factor for node 1: ", f1)

        samples = pd.DataFrame(
            {
                "theta_cos": theta_cos,
                "theta_sin": theta_sin,
                "omega": omegas,
                "done": done_arr,
                "action": actions,
            }
        )
        return np.mean(m), np.std(m), samples


def test_model_predict(opct, samples):
    """
    If samples contains n_record observations that an OPCT has to predict,
    it is 4x faster to do the prediction in one array call (see "pred_action2") than to do
    the prediction in an implicit for-loop (see "pred_action1")
    """
    n_samples = samples.shape[0]
    print(f"OPCT prediction of {n_samples} records in for-loop ... ")
    start = time.process_time()
    samples["pred_action1"] = [
        opct.predict(
            np.array(
                (
                    samples["theta_cos"][i],
                    samples["theta_sin"][i],
                    samples["omega"][i],
                )
            ).reshape(1, -1)
        )[0][0]
        for i in range(n_samples)
    ]
    print(f"Completed in {time.process_time() - start} sec")

    print(f"OPCT prediction of {n_samples} records in one call ... ")
    start = time.process_time()
    samples["pred_action2"] = opct.predict(
        np.array(
            (
                samples["theta_cos"],
                samples["theta_sin"],
                samples["omega"],
            )
        ).T
    )
    print(f"Completed in {time.process_time() - start} sec")

    assert all(
        samples["pred_action1"].values == samples["pred_action2"].values
    ), "Error in test_model_predict"


def plot3D_attract_pend(inode, opct, samples, pngfile, ttype):
    """
    plot for each observation point whether it is attracted to or repelled from node inode's separating plane.
    In addition, draw the separating hyperplane as red line.
    """
    observations = samples.values[
        :, 0:3
    ]  # columns "theta_cos", "theta_sin", "omega" as np.array
    e_utils.append_dist_cols(
        inode, opct, samples, observations, ttype="regr", epsilon=0.15
    )
    deltacol = f"delta{inode:02d}"
    planecol = f"plane{inode:02d}"
    activcol = f"activ{inode:02d}"
    plt.figure(
        inode, figsize=(7, 4.7)
    )  # why inode? - each node should start a new figure
    axes = plt.axes(projection="3d")
    print(type(axes))
    fg = axes.scatter3D(
        samples["theta_cos"].values,
        samples["theta_sin"].values,
        samples["omega"].values,
        c=samples[planecol].values,
        s=20 * samples[activcol].values + 2,
    )
    axes.set_xlabel("cos(t)")
    axes.set_ylabel("sin(t)")
    axes.set_zlabel("omega")

    #
    # annotate how many samples are routed through inode [in total and with node value (below/above) thresh]
    #
    activ = samples[activcol].values
    dist = samples[f"node{inode:02d}"].values
    n_act = sum(activ)
    n_act_below = sum(activ[np.flatnonzero(dist < 0)])
    n_act_above = sum(activ[np.flatnonzero(dist >= 0)])
    vmax = max(samples.omega)
    zdir = (1, 1, 0)  # 'x' #
    # axes.text(0.0,0.0,0.0,f"active: {n_act} ({n_act_below}, {n_act_above})", zdir)
    axes.text2D(
        0.0,
        0.0,
        f"active: {n_act} ({n_act_below}, {n_act_above})",
        transform=axes.transAxes,
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
    # annotate the prototype (below/above) thresh, if inode has leaves as children
    #
    protos = e_utils.get_proto_childs(inode, opct, ttype)
    axes.text2D(
        1.0,
        1.0,
        f"proto: ({protos['left']:.3f}, {protos['right']:.3f})",
        transform=axes.transAxes,
    )

    plt.savefig(pngfile, format="png", dpi=150)
    plt.show()
    samples = e_utils.remove_dist_cols(inode, samples)
    return samples


def project2d_node_plane(plt, node, inode, samples):
    """
    Given a plot context ``plt`` with certain 'xdata','ydata', plot on top a red line, the separating plane of
    OPCT node ``node`` with index ``inode``.
    """
    assert node.left != -1, f"Error: inode={inode} specifies a leaf node"
    weights = node.split_weights.to_ndarray()[0]
    thresh = node.threshold
    xdata = samples.theta
    ydata = samples.omega
    ymin = min(ydata)
    ymax = max(ydata)
    pmin = min(xdata)
    pmax = max(xdata)
    plt.plot([0, 0], [ymin, ymax], c="black", linestyle="dashed", linewidth=0.5)
    plt.plot([pmin, pmax], [0, 0], c="black", linestyle="dashed", linewidth=0.5)
    if weights[2] == 0:
        if weights[1] == 0:
            pmin = pmax = np.arccos(thresh / weights[0])
        if weights[0] == 0:
            pmin = pmax = np.arcsin(thresh / weights[1])
        if weights[0] != 0 and weights[1] != 0:
            tstep = 50
            trange = np.linspace(-np.pi, np.pi, tstep)
            delta = np.abs(
                weights[0] * np.cos(trange) + weights[1] * np.sin(trange) - thresh
            )
            pmin = pmax = trange[np.argmin(delta)]
        plt.plot([pmin, pmax], [ymin, ymax], c="red")
    else:
        pstep = 50
        prange = np.linspace(min(xdata), max(xdata), pstep)
        pcos = np.cos(prange)
        psin = np.sin(prange)
        vrange = (thresh - weights[0] * pcos - weights[1] * psin) / weights[2]
        ind = np.flatnonzero((ymin <= vrange) & (vrange <= ymax))
        plt.plot(prange[ind], vrange[ind], c="red")


def plot_attract_pend(inode, opct, samples, pngfile, ttype, xcol="theta", ycol="omega"):
    """
    Plot for each observation point whether it is attracted to or repelled from node ``inode``'s separating plane.
    Make a scatter plot of only dimension ``xcol`` and ``ycol`` of ``samples``.
    In addition, draw the separating hyperplane as red line.
    """

    # append column "theta":
    samples["theta"] = np.arctan2(
        np.float64(samples.theta_sin),
        np.float64(samples.theta_cos)
        # np.float64 important for numeric accuracy (!!)
    )

    observations = samples.values[
        :, 0:3
    ]  # columns "theta_cos", "theta_sin", "omega" as np.array
    e_utils.append_dist_cols(
        inode, opct, samples, observations, ttype="regr", epsilon=0.15
    )
    deltacol = f"delta{inode:02d}"
    planecol = f"plane{inode:02d}"
    leafpcol = f"leafp{inode:02d}"
    activcol = f"activ{inode:02d}"
    plt.figure(
        inode, figsize=(7, 4.7)
    )  # why inode? - each node should start a new figure
    sns.scatterplot(
        data=samples,
        x=xcol,
        y=ycol,
        size=activcol,
        sizes=(40, 10),
        hue=planecol,
        palette=None,
    )  # "Set2")
    plt.legend(loc="upper center")
    node = opct.trees[0][inode]
    project2d_node_plane(plt, node, inode, samples)
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
    vmin = min(samples.omega)
    vmax = max(samples.omega)
    plt.text(
        0,
        0.70 * vmin,
        f"active: {n_act} ({n_act_below}, {n_act_above}) [a:{n_act_attract}, r:{n_act_repell}]",
        horizontalalignment="center",
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
    # annotate the prototype (below/above) thresh, if inode has leaves as children
    #
    protos = e_utils.get_proto_childs(inode, opct, ttype)
    wproto = e_utils.get_weighted_protos(
        inode, dist, samples[leafpcol].values, n_act_below, n_act_above
    )
    plt.text(
        0,
        0.84 * vmin,
        f"proto: ({protos['left']:.3f}, {protos['right']:.3f})",
        horizontalalignment="center",
        fontsize=9,
    )
    plt.text(
        0,
        0.94 * vmin,
        f"[{wproto['left']:.3f}, {wproto['right']:.3f}]",
        horizontalalignment="center",
        fontsize=9,
    )

    plt.savefig(pngfile, format="png", dpi=150)
    # plt.show()
    samples = e_utils.remove_dist_cols(inode, samples)
    return samples


def plot_all_project2d_pend(opct, samples, ttype="regr", xcol="theta", ycol="omega"):
    if os.path.exists("png"):
        shutil.rmtree(
            "png"
        )  # delete all files in png/ (otherwise a prior 'plane14.png' could remain)
    os.mkdir("png")
    for inode in range(opct.trees[0].size):
        node = opct.trees[0][inode]
        if node.left != -1:  # if it is not a leaf node:
            pngfile = f"png/plane{inode:02d}.png"
            samples = plot_attract_pend(
                inode, opct, samples, pngfile, ttype, xcol=xcol, ycol=ycol
            )


def heatmap_pend(model, fignum, pngfile):
    # print ("Starting heatmap plot ...")
    N_POINTS = [100, 100]
    thetas = np.linspace(-np.pi, np.pi, N_POINTS[0])
    omegas = np.linspace(-8, 8, N_POINTS[1])
    thetas = np.round(thetas, 3)  # limit the digits --> nicer axis in plot
    omegas = np.round(
        omegas, 3
    )  # (but not too few digits --> pivot may fail due to duplicate indices (!))
    mesh = np.meshgrid(thetas, omegas)
    combinations = np.array(mesh).T.reshape(-1, 2)
    assert len(combinations) == np.prod(np.array(N_POINTS))

    obs = np.array(
        [np.cos(combinations[:, 0]), np.sin(combinations[:, 0]), combinations[:, 1]]
    ).T
    z = model.predict(obs)
    if type(z) == tuple:  # if called with TD3 model, it returns a tuple
        z = z[0]

    plt.figure(
        fignum, figsize=(7, 4.7)
    )  # why inode? - each node should start a new figure
    # reformat the meshgrid data to a pivot and show log(zc) as a heatmap:
    data = np.hstack((combinations, z.reshape(-1, 1)))
    df = pd.DataFrame(columns=["theta", "omega", "height"], data=data)
    pivot = df.pivot(index="omega", columns="theta", values="height")
    pivot = pivot.sort_index(ascending=False)  # row with highest omega at top
    ax = sns.heatmap(pivot, xticklabels=3, yticklabels=5, annot=False)
    # show every 3rd column name as tick-label
    ax.plot(
        [0, 0],
        [np.min(omegas), np.max(omegas)],
        c="black",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax.plot(
        [np.min(thetas), np.max(thetas)],
        [0, 0],
        c="black",
        linestyle="dashed",
        linewidth=0.5,
    )
    plt.savefig(pngfile, format="png", dpi=150)
    # plt.show()
    print(f"Heatmap saved to {pngfile} ")


def append_multi_dist_cols(inodes, opct, samples, observations):
    for inode in inodes:
        dist = np.float64(e_utils.get_dist_from_nodeplane(observations, opct, inode))
        distcol = f"dist{inode:02d}"
        leafpcol = f"leafp{inode:02d}"
        activcol = f"activ{inode:02d}"
        samples[distcol] = dist
        samples[leafpcol] = e_utils.obs_with_inode_active(
            observations, opct, inode, "class"
        )
        samples[activcol] = 1 - np.isnan(samples[leafpcol])  # 0: inactive, 1: active
    # colors = ['b', 'y', 'r', 'k']
    # clrs = [ colors[k] for k in samples["action"].to_numpy() ]
    # samples["clrs"] = clrs
    # actionnames = ['none','left','main','right']
    # acts = [ actionnames[k] for k in samples["action"].to_numpy() ]
    # samples["actname"] = acts
    # samples["index"] = range(len(acts))


def append_equilib_col(samples, observations):
    pnt = np.array([1, 0, 0]).reshape(
        (1, 3)
    )  # the unstable equilibrium point theta=0 (cos(th)=1,sin(th)=0), omega=0
    dist = np.float64(e_utils.get_dist_from_point(observations, pnt))
    samples["equilib"] = dist
    samples["activequi"] = 0 * dist + 1


def plot_distance_pend(
    inodes, opct, samples, pngdir, pngbase, show_eq, prefix="PE_", ttype=None
):
    """
    Generate distance plots for each node in ``inodes`` and each episode in ``samples``, colouring the points according
    to the actions taken.

    ``samples`` may contain multiple episodes separated by a ``done=False``-row.

    :param inodes:  a set of split nodes (usually not larger than 3, would make plots too small)
    :param opct:    an oblique tree
    :param samples: episode samples
    :param pngdir:  where to save the plots
    :param pngbase: e.g. 'distp', then the 5th episode is saved in 'PE_distp_05.png'
    :param show_eq: switch: if True, show equilib, else, show dist01
    :param prefix:  prefix for PNG filename
    :param ttype:   not used
    :return: nothing
    """
    # if os.path.exists(pngdir):
    #     shutil.rmtree(
    #         pngdir
    #     )  # delete all files in pngdir/ (otherwise a prior 'distp_14.png' could remain)
    if not os.path.exists(pngdir):
        os.mkdir(pngdir)
    observations = samples.values[
        :, 0:3
    ]  # columns "theta_cos", "theta_sin", "omega" as np.array
    append_multi_dist_cols(inodes, opct, samples, observations)
    append_equilib_col(samples, observations)
    epi_start, epi_end = e_utils.cut_episodes(samples)
    mean_last_dist = pd.DataFrame()
    for epi in range(len(epi_start)):
        sample1 = pd.DataFrame(
            samples[epi_start[epi] : epi_end[epi]]
        )  # make a copy, not a slice
        nrow = sample1.shape[0]
        sample1["t"] = range(nrow)
        sample1["action"] = np.round(sample1["action"], 2)  # to get a nicer legend
        distcols = [f"dist{inode:02d}" for inode in inodes] + ["equilib", "theta_sin"]
        last50 = sample1.loc[epi_end[epi] - 50 : epi_end[epi] - 1, distcols]
        mean_last_dist = pd.concat(
            [mean_last_dist, pd.DataFrame(last50.mean()).transpose()], axis=0
        )
        legend = ["auto", False, False, False]

        # select the columns to plot
        ycol = [f"dist{inode:02d}" for inode in inodes]
        ynam = [f"distance for node {inode:01d}" for inode in inodes]
        activcol = [f"activ{inode:02d}" for inode in inodes]
        if show_eq:
            # special case (only for len(inodes)==3): plot in lower left subplot the distance of each observation to the
            # equilibrium point --> we can see that node 0 gets earlier a small plane-distance than the distances
            # to the equilibrium point (1,0,0)
            ycol += ["equilib"]
            ynam += ["distance to equilibrium"]
            activcol += ["activequi"]
            fig, ax = plt.subplots(2, 2, sharex="all", figsize=(10, 8))
            numplots = 4
            ind_sub = [(0, 0), (0, 1), (1, 1), (1, 0)]
        else:
            # the normal case: distance plot for each inode (len(inodes) subplots in one column):
            fig, ax = plt.subplots(len(inodes), 1, sharex="all", figsize=(10, 8))
            if len(inodes) == 1:
                ax = [ax]
            numplots = len(inodes)
            ind_sub = [k for k in range(len(inodes))]

        for k in range(numplots):
            # inode = inodes[k]
            ax[ind_sub[k]].plot(
                range(nrow), np.zeros(nrow), c="0.7", lw=0.5
            )  # plot the zero line
            ax[ind_sub[k]].plot(
                [64, 64 + 1e-6],  # plot a vertical line at t=64
                [min(sample1[ycol[k]]), max(sample1[ycol[k]])],
                c="0.7",
                lw=0.5,
            )

            ax[ind_sub[k]].set_ylabel(ynam[k])

            # --- seaborn version: legend + nicer points ---
            sns.scatterplot(
                data=sample1,
                x="t",
                y=ycol[k],
                hue="action",
                palette=None,
                alpha=0.7,  # transparency, to see 'points behind'
                size=activcol[k],
                sizes=(40, 10),
                legend=legend[k],
                ax=ax[ind_sub[k]],
            )
        filename = f"{pngdir}/{prefix}{pngbase}_{epi:02d}.png"
        # red_patch = mlines.Line2D([], [], color='red', marker='o', label='right')
        # blue_patch = mlines.Line2D([], [], color='blue', marker='o', label='none')
        # plt.legend(handles=[red_patch,blue_patch])
        plt.savefig(filename, format="png", dpi=150)
        print(f"Saved distance plot to {filename}")
        plt.close()

    # mean_last_dist: what is the average distance to hyperplanes during the last 50 steps of an episode?
    mean_last_dist.index = range(len(epi_start))
    print(mean_last_dist)
    print()
    print(mean_last_dist.describe())
    # plt.show()
