import numpy as np
import pandas as pd
from Initializing.Environments import load_environment as le
import pickle
from Experiments.MountainCar.MC_SENS_utils import (
    MC_EvalStrategy,
    plot_attract_mc,
    plot_all_attract_mc,
    plot_distance_mc
)
import Experiments.utils as e_utils

RELOAD = False      # if True, reload a previously calculated df_res from file pklname (sensitivity)
pklname = 'pkl/df_res.pkl'
pngdir = "png_sens"
pngname = "reward_sensitivity.png"
n_oracle_evaluation_eps = 100
n_tree_evaluation_eps = 30      # how many reward-evaluation episodes for tree in sensitivity
n_tree_plane_eps = 10           # how many reward-evaluation episodes for tree in plane plots
lsize = 16                      # how many points in sensitivity interval [-100%, 200%]
reward_solved = -110

class MC_Sensitivity(e_utils.SensitivityStrategy):
    def sensitivity_analysis(self,inode=0, setzero=None):
        """
        Make a sensitivity analysis for node ``inode`` of OPCT in environment MountainCar.

        Take each weight of ``inode`` in turn and vary it in interval [-100%, +200%] of its nominal value, while all
        the other weights stay at their nominal values.
        Evaluate the resulting tree by calculating the mean reward from n_tree_evaluation_eps (=30) episodes.

        In addition, if ``setzero`` is not ``None``, set the weights for all dimensions ``j`` with ``setzero[j]=True``
        permanently to zero and exclude them from sweeping. If ``setzero==None`` (the default), no weight is excluded.

        Additionally, take the threshold of ``inode`` and vary it in interval [t - mw, t + mw], where t = nominal
        threshold value and mw = mean of all node weights, while all these node weights stay at their nominal values.

        :param inode: node number
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        # mc_evs = MC_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        env = le.MountainCarEnvironment().get_environment()
        # model = lo.DQN_Oracle().load_model(
        #     le.MountainCarEnvironment().get_environmentShort()
        # )
        # base_samples, rewards = o_utils.evaluate_oracle_mc(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = base_samples.std()
        # print("std obs space: ",std[0],std[1])   # 0.4239, 0.0245
        std = np.array([0.4239, 0.0245])
        # reload a tree generated and saved by MC_ITER.py
        with open("../../Paper_Results/OPCTs/MountainCar-v0_ITER_depth_1.pkl", "rb") as input:
            best_tree = pickle.load(input)

        dimname = ['position', 'velocity']

        df_res = self.sensi_ana_V0(inode, setzero, dimname, lsize,
                                   env, best_tree, std
                                   )
        return df_res

def main_part_sensitivity():
    if RELOAD:
        df_res_lst = []
        df_res_lst.append(pd.read_pickle(pklname))
        df_res_dict = { 'n0': df_res_lst}
        print(f"Reloaded results 'df_res' from {pklname}")
    else:
        mc_evs = MC_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        mc_ses = MC_Sensitivity(mc_evs)
        inodes = [0]
        setzeros = [
            #  position velocity
              [False,   False]
            , [True,    False]
        ]
        df_res_dict = e_utils.generate_sens_pkls(mc_ses,inodes,setzeros, prefix='MC_')

    keys = list(df_res_dict.keys())
    df_res_lst = df_res_dict[keys[0]]
    e_utils.plot_sensitivity(df_res_lst[0],pngdir,pngname)

def main_part_attract_plot():
    mc_evs = MC_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.MountainCarEnvironment().get_environment()
    # model = lo.DQN_Oracle().load_model(
    #     le.MountainCarEnvironment().get_environmentShort()
    # )
    # base_samples, rewards = o_utils.evaluate_oracle_mc(
    #     model, env, n_oracle_evaluation_eps
    # )
    # std = base_samples.std()
    # print("std obs space: ",std[0],std[1])   # 0.4239, 0.0245
    std = np.array([0.4239, 0.0245])

    # reload a tree generated and saved by MountainCarIterative.py
    with open("../../Paper_Results/OPCTs/MountainCar-v0_ITER_depth_1.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = mc_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )

    inode = 0
    e_utils.print_treemodel(best_tree, inode, ttype="class")
    print(f"*** mean tree reward: ", tree_reward_mean, f"***   (solved: >= {reward_solved})")
    # plot_attract_mc(inode, best_tree, tree_samples,f"plane{inode}.png")
    plot_all_attract_mc(best_tree, tree_samples, ttype="class")

def main_part_distance_plot():
    mc_evs = MC_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.MountainCarEnvironment().get_environment()
    std = np.array([0.4239, 0.0245])
    with open("../../Paper_Results/OPCTs/MountainCar-v0_ITER_depth_1.pkl", "rb") as input:
        best_tree = pickle.load(input)

    e_utils.print_treemodel(best_tree)
    tree_reward_mean, tree_reward_std, tree_samples = mc_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    # tree_reward_mean, tree_reward_std, tree_samples = mc_evs.eval_opct_monitor(
    #     env, best_tree, n_tree_plane_eps, std
    # )

    inodes = [0]
    pngdir = "png_dist"
    pngbase = "distp"
    plot_distance_mc(inodes, best_tree, tree_samples, pngdir, pngbase)

if __name__ == "__main__":
    #main_part_sensitivity()
    #main_part_attract_plot()
    main_part_distance_plot()

