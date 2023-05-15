import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import oracle_utils as o_utils
from Initializing.Oracles import load_oracle as lo
from Initializing.Environments import load_environment as le
from Experiments.CartPoleSwingUp.CPSU_SENS_utils import (
    CPSU_EvalStrategy,
    plot_all_attract_cpsu,
    plot_distance_cpsu,
)
import Experiments.utils as e_utils


RELOAD = False  # if True, reload a previously calculated pklname, e.g. 'pkl/df_res.pkl'
pklname = "pkl/df_res.pkl"
pngdir = "png_sens"
pngname = "reward_sensitivity.png"
n_oracle_evaluation_eps = 40
n_tree_evaluation_eps = (
    30  # how many reward-evaluation episodes for tree in sensitivity
)
n_tree_plane_eps = 10  # how many reward-evaluation episodes for tree in plane plots
lsize = 16  # how many points in sensitivity interval [-100%, 200%]
reward_solved = 830
ylim = None  # (-200,300)


class CPSU_Sensitivity(e_utils.SensitivityStrategy):
    def __init__(self, eval_strategy: e_utils.EvalStrategy):
        super().__init__(eval_strategy)

    def sensitivity_analysis(self, inode=0, setzero=None):
        """
        Starter for ``self.sensi_ana``, see its docu in ``sensi_helpers.py``

        :param inode: node number
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.CartpoleSwingUpEnvironment().get_environment()
        # model = lo.DQN_Oracle().load_model(
        #     le.CartpoleSwingUpEnvironment().get_environmentShort()
        # )
        # base_samples, rewards = o_utils.evaluate_oracle_cpsu(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = base_samples.std()
        # print("std obs space: ",std)   # 0.020306, 0.142200, 0.008996, 0.204207
        std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306])
        # reload a tree generated and saved by MountainCarIterative.py
        with open(
            "../../Paper_Results/OPCTs/CartPoleSwingUp-v0_ITER_depth_3.pkl", "rb"
        ) as input:
            best_tree = pickle.load(input)

        dimname = ["position", "velocity", "angle_cos", "angle_sin", "angle_velocity"]

        df_res = self.sensi_ana_V0(
            inode, setzero, dimname, lsize, env, best_tree, std, print_tree=False
        )
        return df_res

    def multi_node_sensitivity(self, inodes, setzero=None):
        """
        Starter for ``self.mult_node_sens``, see its docu in ``sensi_helpers.py``

        :param inodes: a list of node numbers
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.CartpoleSwingUpEnvironment().get_environment()
        model = lo.DQN_Oracle().load_model(
            le.CartpoleSwingUpEnvironment().get_environmentShort()
        )
        base_samples, rewards = o_utils.evaluate_oracle_cpsu(
            model, env, n_oracle_evaluation_eps
        )
        std = base_samples.std()
        print("std obs space: ", std)  # 0.020306, 0.142200, 0.008996, 0.204207
        # std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306, 0.142200, 0.008996, 0.204207])
        # reload a tree generated and saved by CPSU_ITER
        with open(
            "../../Paper_Results/OPCTs/CartPoleSwingUp-v0_ITER_depth_3.pkl", "rb"
        ) as input:
            best_tree = pickle.load(input)

        dimname = ["position", "velocity", "angle_cos", "angle_sin", "angle_velocity"]

        df_res = self.mult_node_sens_V0(
            inodes, setzero, dimname, lsize, env, best_tree, std, print_tree=False
        )
        return df_res


def main_part_sensitivity():
    if RELOAD:
        # df_res_lst = []
        # df_res_lst.append(pd.read_pickle(pklname))
        # df_res_dict = { 'n0': df_res_lst}
        # print(f"Reloaded results 'df_res' from {pklname}")
        df_res_dict = e_utils.generate_sens_plots(prefix="CPSU_", ylim=ylim)
    else:
        cpsu_evs = CPSU_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        cpsu_ses = CPSU_Sensitivity(cpsu_evs)
        inodes = [0, 1, 2]
        setzeros = [
            # position velocity angle_cos angle_sin angle_velocity
            [False, False, False, False, False]
            # ,[False, False, False, False,False]
        ]
        df_res_dict = e_utils.generate_sens_pkls(
            cpsu_ses, inodes, setzeros, prefix="CPSU_", ylim=ylim
        )
        # df_res_dict = seh.generate_multi_sens_pkls(cpsu_ses, inodes, setzeros, prefix='CPSU_', ylim=ylim)
        # --- seh.generate_... includes plotting ---

    # --- obsolete now, already done by plots above  ---
    # keys = list(df_res_dict.keys())
    # df_res_lst = df_res_dict[keys[0]]
    # seh.plot_sensitivity(df_res_lst[0], pngdir, pngname, ylim=ylim)
    plt.close()


def main_part_attract_plot():
    ll_evs = CPSU_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.CartpoleSwingUpEnvironment().get_environment()
    model = lo.DQN_Oracle().load_model(
        le.CartpoleSwingUpEnvironment().get_environmentShort()
    )
    base_samples, rewards = o_utils.evaluate_oracle_cpsu(
        model, env, n_oracle_evaluation_eps
    )
    std = base_samples.std()
    print("std obs space: ", std)  # 0.020306, 0.142200, 0.008996, 0.204207
    # std = np.array([0.020306, 0.142200, 0.008996, 0.204207,0.020306, 0.142200, 0.008996, 0.204207])
    with open(
        "../../Paper_Results/OPCTs/CartPoleSwingUp-v0_ITER_depth_3.pkl", "rb"
    ) as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = ll_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    plot_all_attract_cpsu(best_tree, tree_samples, ttype="class")
    inode = 0
    e_utils.print_treemodel(best_tree, inode, ttype="class")


def main_part_distance_plot():
    SHOW_EQUILIB = True
    NODE_SET = "0-1-2"  # "0-1-2" or "2-3-4" or "4-9-10"
    cpsu_evs = CPSU_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.CartpoleSwingUpEnvironment().get_environment()
    # model = lo.DQN_Oracle().load_model(
    #     le.CartpoleSwingUpEnvironment().get_environmentShort()
    # )
    # base_samples, rewards = o_utils.evaluate_oracle_cpsu(
    #     model, env, n_oracle_evaluation_eps
    # )
    # std = base_samples.std()
    # print("std obs space: ", std)  # 0.020306, 0.142200, 0.008996, 0.204207
    std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306])
    with open(
        "../../Paper_Results/OPCTs/CartPoleSwingUp-v0_ITER_depth_3.pkl", "rb"
    ) as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = cpsu_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )

    if SHOW_EQUILIB:
        pngdir_base = "png_dist/with_equilib"
    else:
        pngdir_base = "png_dist/w_o_equilib"
    if not os.path.exists(pngdir_base):
        os.mkdir(pngdir_base)

    if NODE_SET == "0-1-2":
        inodes = [0, 1, 2]
        pngdir = f"{pngdir_base}/nodes0-1-2"
    else:
        if NODE_SET == "2-3-4":
            inodes = [2, 3, 4]
            pngdir = f"{pngdir_base}/nodes2-3-4"
        else:
            inodes = [4, 9, 10]
            pngdir = f"{pngdir_base}/nodes4-9-10"
    pngbase = "distp"
    plot_distance_cpsu(inodes, best_tree, tree_samples, pngdir, pngbase, SHOW_EQUILIB)


if __name__ == "__main__":
    # main_part_sensitivity()
    # main_part_attract_plot()
    main_part_distance_plot()
