import os
import numpy as np
import matplotlib.pyplot as plt
from Iterative import utils
import oracle_utils as o_utils
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_env as le
import pickle
import Pend_WK_utils as p_utils
from WK_utils import print_treemodel
import PlanePlot.sensi_helpers as seh
from Paper_Results.utils import eval_opct_reg

RELOAD = True  # if True, reload a previously calculated pklname, e.g. 'pkl/df_res.pkl' (sensitivity)
PRINT_TREE = True
SINGLE_NODE = True  # if True, generate single-node sensitivity plots, if False multi-node sensitivity plots
pklname = "pkl/df_res.pkl"
pngdir = "png_sens"
# pngname = "reward_sensitivity.png"
n_oracle_evaluation_eps = 100
n_tree_evaluation_eps = (
    90  # how many reward-evaluation episodes for tree in sensitivity
)
n_tree_plane_eps = 10  # how many reward-evaluation episodes for tree in plane plots
lsize = 16  # how many points in sensitivity interval [-100%, 200%]
reward_solved = -178
ylim = (-500, -100)


class Pend_Sensitivity(seh.SensitivityStrategy):
    def __init__(self, eval_strategy: seh.EvalStrategy):
        super().__init__(eval_strategy)

    def sensitivity_analysis(self, inode=0, setzero=None):
        """
        Starter for ``self.sensi_ana``, see its docu in ``sensi_helpers.py``

        :param inode: node number
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.Pendulumv1Environment().get_environment()
        # model = lo.TD3_Oracle().load_model(le.Pendulumv1Environment().get_environmentShort())
        # base_samples, rewards = o_utils.evaluate_oracle_pend(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = np.array(base_samples.std())[0:3]
        # print("std obs space: ",std[0],std[1],std[2])   # 0.4618, 0.3059
        std = np.array([0.4504522979259491, 0.29696765542030334, 1.5146982669830322])
        # reload a tree generated and saved by PendulumIterative.py
        with open(
            "../../Paper_Results/OPCTs/Pendulum-v1_ITER_depth_3.pkl", "rb"
        ) as input:  # best_tree_MO_d02
            best_tree = pickle.load(input)

        dimname = ["theta_cos", "theta_sin", "omega"]

        # tree_samples, tree_rewards = eval_opct_reg(
        #     opct=best_tree,
        #     env=env,
        #     columns=dimname + ['actions'],
        #     num_episodes=n_tree_evaluation_eps
        # )

        df_res = self.sensi_ana_V0(
            inode,
            setzero,
            dimname,
            lsize,
            env,
            best_tree,
            std,
            ttype="regr",
            print_tree=PRINT_TREE,
        )
        return df_res

    def multi_node_sensitivity(self, inodes, setzero=None):
        """
        Starter for ``self.mult_node_sens``, see its docu in ``sensi_helpers.py``

        :param inodes: a list of node numbers
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.Pendulumv1Environment().get_environment()
        # model = lo.TD3_Oracle().load_model(le.Pendulumv1Environment().get_environmentShort())
        # base_samples, rewards = o_utils.evaluate_oracle_pend(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = np.array(base_samples.std())[0:3]
        # print("std obs space: ",std[0],std[1],std[2])   # 0.4618, 0.3059
        std = np.array([0.4504522979259491, 0.29696765542030334, 1.5146982669830322])
        # reload a tree generated and saved by PendulumIterative.py
        with open(
            "../../Paper_Results/OPCTs/Pendulum-v1_ITER_depth_3.pkl", "rb"
        ) as input:  # best_tree_MO_d02
            best_tree = pickle.load(input)

        dimname = ["theta_cos", "theta_sin", "omega"]

        df_res = self.mult_node_sens_V0(
            inodes, setzero, dimname, lsize, env, best_tree, std, ttype="regr"
        )
        return df_res


def main_part_sensitivity():
    if RELOAD:
        # df_res_lst = []
        # df_res_lst.append(pd.read_pickle(pklname))
        # df_res_dict = { 'n0': df_res_lst}
        # print(f"Reloaded results 'df_res' from {pklname}")
        df_res_dict = seh.generate_sens_plots(prefix="PE_", ylim=ylim)
    else:
        pend_evs = p_utils.Pend_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        pend_ses = Pend_Sensitivity(pend_evs)
        inodes = [0, 1, 2]
        setzeros = [
            # th_cos th_sin omega
            [False, False, False]
            # ,[False, True, False]
        ]
        if SINGLE_NODE:
            df_res_dict = seh.generate_sens_pkls(
                pend_ses, inodes, setzeros, prefix="PE_", ylim=ylim
            )
        else:
            df_res_dict = seh.generate_multi_sens_pkls(
                pend_ses, inodes, setzeros, prefix="PE_", ylim=ylim
            )
        # --- methods seh.generate_** include plotting ---

    # --- obsolete now, already done by plots above  ---
    # keys = list(df_res_dict.keys())
    # df_res_lst = df_res_dict[keys[0]]
    # seh.plot_sensitivity(df_res_lst[0], pngdir, pngname)
    plt.close()


def main_part_attract_plot():
    pend_evs = p_utils.Pend_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.Pendulumv1Environment().get_environment()
    model = lo.TD3_Oracle().load_model(
        le.Pendulumv1Environment().get_environmentShort()
    )
    # base_samples, rewards = o_utils.evaluate_oracle_pend(
    #     model, env, n_oracle_evaluation_eps
    # )
    # std = np.array(base_samples.std())[0:3]
    # print("std obs space: ",std[0],std[1],std[2])   # 0.4618, 0.3059
    std = np.array([0.4504522979259491, 0.29696765542030334, 1.5146982669830322])

    # reload a tree generated and saved by PendulumIterative.py
    with open(
        "../../Paper_Results/OPCTs/Pendulum-v1_ITER_depth_3.pkl", "rb"
    ) as input:  # best_tree_MO_d02
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = pend_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    print("rewards: ", tree_reward_mean)
    (
        best_tree_reward_mean,
        best_tree_reward_std,
        best_tree_samples,
    ) = utils.evaluate_opct_pend(env, best_tree, 100)
    print("best tree reward mean: ", best_tree_reward_mean)

    inode = 2
    if PRINT_TREE:
        print_treemodel(best_tree, 0, ttype="regr")
    print(
        f"*** mean tree reward: ",
        tree_reward_mean,
        f"***   (solved: >= {reward_solved})",
    )
    # plot_attract_pend(inode, best_tree, tree_samples,f"plane{inode}.png", ttype="regr")
    p_utils.plot_all_project2d_pend(
        best_tree, tree_samples, ttype="regr", xcol="theta", ycol="omega"
    )

    png_tree_hm = "png/tree_heatmap.png"
    png_oracle_hm = "png/sb3_heatmap.png"
    p_utils.heatmap_pend(best_tree, 100, png_tree_hm)
    p_utils.heatmap_pend(model, 101, png_oracle_hm)


def main_part_distance_plot():
    SHOW_EQUILIB = True
    pend_evs = p_utils.Pend_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.Pendulumv1Environment().get_environment()
    env.seed(seed=20)
    # model = lo.TD3_Oracle().load_model(le.Pendulumv1Environment().get_environmentShort())
    # base_samples, base_rewards = o_utils.evaluate_oracle_pend(
    #     model, env, n_oracle_evaluation_eps
    # )
    # std = np.array(base_samples.std())[0:3]
    std = [0.4679, 0.2999, 1.500]
    print("std obs space: ", std[0], std[1], std[2])  # 0.4618, 0.3059
    std = np.array([0.4504522979259491, 0.29696765542030334, 1.5146982669830322])

    # reload a tree generated and saved by PendulumIterative.py
    with open("../../Paper_Results/OPCTs/Pendulum-v1_ITER_depth_3.pkl", "rb") as input:
        # with open("../../Paper_Results/OPCTs/Pendulum-v0_ITER_depth_4.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = pend_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    print("best tree reward mean: ", tree_reward_mean)
    if PRINT_TREE:
        print_treemodel(best_tree, 0, ttype="regr")
    inodes = [0, 1, 2]

    if SHOW_EQUILIB:
        pngdir = "png_dist/with_equilib"
    else:
        pngdir = "png_dist/w_o_equilib"
    if not os.path.exists(pngdir):
        os.mkdir(pngdir)

    pngbase = "distp"
    p_utils.plot_distance_pend(
        inodes, best_tree, tree_samples, pngdir, pngbase, SHOW_EQUILIB
    )


if __name__ == "__main__":
    # main_part_sensitivity()
    # main_part_attract_plot()
    main_part_distance_plot()
