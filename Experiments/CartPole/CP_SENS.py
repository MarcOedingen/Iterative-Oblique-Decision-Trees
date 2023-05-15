import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Initializing.Environments import load_environment as le
from Experiments.CartPole.CP_SENS_utils import (
    CP_EvalStrategy,
    plot_all_attract_cp,
    plot_distance_cp
)
import Experiments.utils as e_utils

RELOAD = True      # if True, reload a previously calculated pklname, e.g. 'pkl/df_res.pkl' (sensitivity)
PRINT_TREE = False
SENS_VERS = "V0"    # "V0" or "V1", old or new version of sensitivity, see notes_pplot_sens.docx
pklname = 'pkl/df_res.pkl'
pngdir = "png_sens" # PNG directory for sensitivity part
#pngname = "reward_sensitivity.png"
n_oracle_evaluation_eps = 40
n_tree_evaluation_eps = 100     # how many reward-evaluation episodes for tree in sensitivity
n_tree_plane_eps = 10           # how many reward-evaluation episodes for tree in plane plots
lsize = 16                      # how many points in sensitivity interval [-100%, 200%]
reward_solved = 475
ylim = None
w_factor = 0.20  # 0.04      # only relevant for V1

class CP_Sensitivity(e_utils.SensitivityStrategy):
    def __init__(self, eval_strategy: e_utils.EvalStrategy):
        super().__init__(eval_strategy)

    def sensitivity_analysis(self,inode=0, setzero=None):
        """
        Starter for ``self.sensi_ana``, see its docu in ``sensi_helpers.py``

        :param inode: node number
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.Cartpolev1Environment().get_environment()
        cp_evs = CP_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        # model = lo.PPO_Oracle().load_model(
        #      le.Cartpolev1Environment().get_environmentShort()
        # )
        # total_count, counter = count_params_SB3(model) # 27477 [9155 18322]
        # base_samples, rewards = o_utils.evaluate_oracle_cp(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = base_samples.std()
        # print("std obs space: ",std[0],std[1])   # 0.020306, 0.142200, 0.008996, 0.204207
        std = np.array([0.020306, 0.142200, 0.008996, 0.204207])
        dimname = ['position', 'velocity', 'angle', 'angle_v']

        # reload a tree generated and saved by ITER for CartPole:
        with open("../../Paper_Results/OPCTs/CartPole-v1_ITER_depth_1.pkl", "rb") as input:
            best_tree = pickle.load(input)
        tree_reward_mean, tree_reward_std, tree_samples = cp_evs.eval_opct(
            env, best_tree, n_tree_evaluation_eps, std, samples_only=np.inf
        )
        std2 = tree_samples.std().to_numpy()[0:len(dimname)]

        if (SENS_VERS=="V0"):
            df_res = self.sensi_ana_V0(inode, setzero, dimname, lsize,
                                       env, best_tree, std, print_tree=PRINT_TREE
                                       )
        else:
            df_res = self.sensi_ana_V1(inode, setzero, dimname, lsize,
                                       env, best_tree, std2, print_tree=PRINT_TREE, w_factor=w_factor
                                       )
        return df_res

def analyze_pickles():
    """
        Analyze a certain sensitivity pickle file ``pklname`` in all directories listed in ``pkldir``:

        What is the min-max range for each dimension's weight that is swept in sensitivity analysis?

        :return: three data frames:
            1) [min,max] range for each variable in each pickle file,
            2) the standard deviation for each variable and each pickle file,
            3) the nominal weight value for each variable and each pickle file.
    """
    #pkldir = ["pkl_V0","pkl_V1_020_std050","pkl_V1_020_std500"]
    pkldir = ["pkl_V0"] #,"pkl_V1_004_std050","pkl_V1_004_std500"]
    pklname = 'df_res_n0_1111.pkl'
    res = pd.DataFrame()
    dstd= pd.DataFrame()
    dnom= pd.DataFrame()
    for dir in pkldir:
        pklfile = dir + '/' + pklname
        df_res = pd.read_pickle(pklfile)
        df_res.describe()
        res[dir+'_min'] = df_res.groupby('dimname').min()['dim_value']
        res[dir+'_max'] = df_res.groupby('dimname').max()['dim_value']
        dstd[dir+'_std'] = df_res.groupby('dimname').min()['std']
#        if dir == pkldir[2]:
#            df_nom = df_res[df_res['xf'] == 0]
#            dnom[dir+'_nom'] = df_nom.groupby('dimname').max()['dim_value']    # the nominal weight values

    res = res.T
    dstd= dstd.T
    dnom= dnom.T
    print(res)
    print(dstd)
    print(dnom)
    return res, dstd, dnom


def main_part_sensitivity():
    if RELOAD:
        # df_res_lst = []
        # df_res_lst.append(pd.read_pickle(pklname))
        # df_res_dict = { 'n0': df_res_lst}
        # print(f"Reloaded results 'df_res' from {pklname}")
        analyze_pickles()
        df_res_dict = e_utils.generate_sens_plots(prefix='CP_', ylim=ylim, last_levels=['velocity'])
    else:
        cp_evs = CP_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        cp_ses = CP_Sensitivity(cp_evs)
        inodes = [0]
        setzeros = [
            #pos    vel    angle  angle_v
            [False, False, False, False]
            ,[False, True, False, False]
        ]
        df_res_dict = e_utils.generate_sens_pkls(cp_ses, inodes, setzeros, prefix='CP_', ylim=ylim, last_levels=['velocity'])
        #--- method seh.generate_** includes plotting ---

    # --- obsolete now, already done by plots above  ---
    # keys = list(df_res_dict.keys())
    # df_res_lst = df_res_dict[keys[0]]
    # seh.plot_sensitivity(df_res_lst[0],pngdir,pngname)
    plt.close()

def main_part_attract_plot():
    cp_evs = CP_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.Cartpolev1Environment().get_environment()
    std = np.array([0.020306, 0.142200, 0.008996, 0.204207])
    with open("../../Iterative/CartPole/pkl/best_tree_d01.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = cp_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    plot_all_attract_cp(best_tree, tree_samples, ttype="class")
    inode = 0
    if PRINT_TREE: e_utils.print_treemodel(best_tree, inode, ttype="class")

def main_part_distance_plot():
    cp_evs = CP_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.Cartpolev1Environment().get_environment()
    std = np.array([0.020306, 0.142200, 0.008996, 0.204207])
    with open("../../Paper_Results/OPCTs/CartPole-v1_ITER_depth_1.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = cp_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    inodes = [0]
    pngdir = "png_dist"
    pngbase = "distp"
    plot_distance_cp(inodes, best_tree, tree_samples, pngdir, pngbase)

if __name__ == "__main__":
    main_part_sensitivity()
    #main_part_attract_plot()
    #main_part_distance_plot()


