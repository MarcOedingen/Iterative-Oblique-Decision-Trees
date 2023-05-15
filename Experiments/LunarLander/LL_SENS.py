import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Initializing.Environments import load_environment as le
from Experiments.LunarLander.LL_SENS_utils import (
    LL_EvalStrategy,
    plot_all_attract_ll,
    plot_distance_ll
)
import Experiments.utils as e_utils


RELOAD = True       # if True, reload a previously calculated pklname, e.g. 'pkl/df_res.pkl' (sensitivity)
PRINT_TREE = False
SENS_VERS = "V0"    # "V0" or "V1", old or new version of sensitivity, see notes_pplot_sens.docx
SINGLE_NODE = True  # if True, generate single-node sensitivity plots, if False multi-node sensitivity plots
pklname = 'pkl/df_res.pkl'
pngdir = "png_sens"
# pngname = "reward_sensitivity.png"
n_oracle_evaluation_eps = 40
n_tree_evaluation_eps = 100     # how many reward-evaluation episodes for tree in sensitivity
n_tree_plane_eps = 10           # how many reward-evaluation episodes for tree in plane plots
lsize = 16                      # how many points in sensitivity interval [-100%, 200%]
reward_solved = 200
ylim = (-200,300)
w_factor = 0.20     # only relevant for V1


class LL_Sensitivity(e_utils.SensitivityStrategy):
    def __init__(self, eval_strategy: e_utils.EvalStrategy):
        super().__init__(eval_strategy)

    def sensitivity_analysis(self, inode=0, setzero=None):
        """
        Starter for ``self.sensi_ana``, see its docu in ``sensi_helpers.py``

        :param inode: node number
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.LunarLanderEnvironment().get_environment()
        ll_evs = LL_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        # model = lo.PPO_Oracle().load_model(
        #      le.LunarLanderEnvironment().get_environmentShort()
        # )
        # total_count, counter = count_params_SB3(model) # 27477 [9155 18322]
        # base_samples, rewards = o_utils.evaluate_oracle_cp(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = base_samples.std()
        # print("std obs space: ",std[0],std[1])   # 0.020306, 0.142200, 0.008996, 0.204207
        std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306, 0.142200, 0.008996, 0.204207])
        dimname = ['x', 'y', 'vx', 'vy', 'theta', 'omega', 'leg1', 'leg2']

        # reload a tree generated and saved by ITER for LunarLander:
        with open("../../Paper_Results/OPCTs/LunarLander-v2_ITER_depth_2.pkl", "rb") as input:
            best_tree = pickle.load(input)
        tree_reward_mean, tree_reward_std, tree_samples = ll_evs.eval_opct(
            env, best_tree, n_tree_plane_eps, std
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

    def multi_node_sensitivity(self, inodes, setzero=None):
        """
        Starter for ``self.mult_node_sens``, see its docu in ``sensi_helpers.py``

        :param inodes: a list of node numbers
        :param setzero: a boolean list with length = observation space dim
        :returns: data frame with columns 'dim', 'xf', 'reward_mean', 'reward_std', 'dim_value', 'dimname'
        """
        env = le.LunarLanderEnvironment().get_environment()
        ll_evs = LL_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        # model = lo.PPO_Oracle().load_model(
        #      le.LunarLanderEnvironment().get_environmentShort()
        # )
        # total_count, counter = count_params_SB3(model) # 27477 [9155 18322]
        # base_samples, rewards = o_utils.evaluate_oracle_cp(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = base_samples.std()
        # print("std obs space: ",std[0],std[1])
        std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306, 0.142200, 0.008996, 0.204207])
        dimname = ['x', 'y', 'vx', 'vy', 'theta', 'omega', 'leg1', 'leg2']

        # reload a tree generated and saved by LunarLanderIterative.py
        with open("../../Paper_Results/OPCTs/LunarLander-v2_ITER_depth_2.pkl", "rb") as input:
            best_tree = pickle.load(input)
        tree_reward_mean, tree_reward_std, tree_samples = ll_evs.eval_opct(
            env, best_tree, n_tree_plane_eps, std
        )
        std2 = tree_samples.std().to_numpy()[0:len(dimname)]


        if (SENS_VERS=="V0"):
            df_res = self.mult_node_sens_V0(inodes, setzero, dimname, lsize,
                                            env, best_tree, std, print_tree=PRINT_TREE
                                            )
        else:
            df_res = self.mult_node_sens_V1(inodes, setzero, dimname, lsize,
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
    pkldir = ["pkl_V0"] #,"pkl_V1_004","pkl_V1_020"]
    pklname = 'df_res__n-0-1-2__11111111.pkl'
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
        #if dir == pkldir[2]:
        #    df_nom = df_res[df_res['xf'] == 0]
        #    dnom[dir+'_nom'] = df_nom.groupby('dimname').max()['dim_value']    # the nominal weight values

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
        df_res_dict = e_utils.generate_sens_plots(prefix='LL_', ylim=ylim,
                                              first_levels=['x','y','vx','vy'],
                                              last_levels=['leg1','leg2'])
    else:
        ll_evs = LL_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        ll_ses = LL_Sensitivity(ll_evs)
        inodes = [0, 1, 2]
        setzeros = [
            #  x     y     vx     vy     theta  omega  leg1  leg2
            [False, False, False, False, False, False, False, False]
            , [False, False, False, False, False, False, True, True]
        ]
        if SINGLE_NODE:
            df_res_dict = e_utils.generate_sens_pkls(ll_ses, inodes, setzeros, prefix='LL_', ylim=ylim,
                                                 first_levels=['x', 'y', 'vx', 'vy'],
                                                 last_levels=['leg1','leg2'])
        else:
            df_res_dict = e_utils.generate_multi_sens_pkls(ll_ses, inodes, setzeros, prefix='LL_', ylim=ylim,
                                                       first_levels=['x', 'y', 'vx', 'vy'],
                                                       last_levels=['leg1','leg2'])
        # --- methods seh.generate_** include plotting ---

    # --- obsolete now, already done by plots above  ---
    # keys = list(df_res_dict.keys())
    # df_res_lst = df_res_dict[keys[0]]
    # seh.plot_sensitivity(df_res_lst[0], pngdir, pngname, ylim=ylim)
    plt.close()


def main_part_attract_plot():
    ll_evs = LL_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.LunarLanderEnvironment().get_environment()
    std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306, 0.142200, 0.008996, 0.204207])
    with open("../../Paper_Results/OPCTs/LunarLander-v2_ITER_depth_2.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = ll_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    plot_all_attract_ll(best_tree, tree_samples, ttype="class")
    inode = 0
    if PRINT_TREE: e_utils.print_treemodel(best_tree, inode, ttype="class")


def main_part_distance_plot():
    ll_evs = LL_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.LunarLanderEnvironment().get_environment()
    env.seed(seed=17)
    std = np.array([0.020306, 0.142200, 0.008996, 0.204207, 0.020306, 0.142200, 0.008996, 0.204207])
    with open("../../Paper_Results_v2/OPCTs/LunarLander-v2_ITER_depth_2.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = ll_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    inodes = [0, 1, 2]
    pngdir = "png_dist"
    pngbase = "distp"
    plot_distance_ll(inodes, best_tree, tree_samples, pngdir, pngbase)


if __name__ == "__main__":
    # main_part_sensitivity()
    # main_part_attract_plot()
    main_part_distance_plot()

