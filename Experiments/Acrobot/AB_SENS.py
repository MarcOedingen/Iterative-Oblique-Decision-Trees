import pickle
import numpy as np
import pandas as pd
from Initializing.Environments import load_environment as le
from Experiments.Acrobot.AB_SENS_utils import (
    AB_EvalStrategy,
    plot_all_attract_ab,
    plot_distance_ab
)
from Experiments.utils import SensitivityStrategy, plot_sensitivity, generate_sens_pkls

RELOAD = False      # if True, reload a previously calculated df_res from file pklname
pklname = 'pkl/df_res.pkl'
pngdir = "png_sens"
pngname = "reward_sensitivity.png"
n_oracle_evaluation_eps = 40
n_tree_evaluation_eps = 30      # how many reward-evaluation episodes for tree in sensitivity
n_tree_plane_eps = 10           # how many reward-evaluation episodes for tree in plane plots
lsize = 16                      # how many points in sensitivity interval [-100%, 200%]
reward_solved = -86

class AB_Sensitivity(SensitivityStrategy):
    def sensitivity_analysis(self,inode=0, setzero=None):
        """
        Make a sensitivity analysis for node ``inode`` of OPCT in environment Acrobat.

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
        env = le.AcrobotEnvironment().get_environment()
        # model = lo.DQN_Oracle().load_model(le.AcrobotEnvironment().get_environmentShort())
        # total_count, counter = count_params_SB3(model) # 27477 [9155 18322]
        # base_samples, rewards = o_utils.evaluate_oracle_ab(
        #     model, env, n_oracle_evaluation_eps
        # )
        # std = base_samples.std()
        # print("std obs space: ",std)   # 0.509362  0.662336  0.677396 0.697312  2.184065  3.238214
        std = np.array([0.509362, 0.662336, 0.677396, 0.697312, 2.184065, 3.238214])

        # reload a tree generated and saved by AB_ITER.py
        with open("../../Experiments/OPCTs/Acrobot-v1_ITER_depth_1.pkl", "rb") as input:
            best_tree = pickle.load(input)

        dimname = ['theta1_cos', 'theta1_sin', 'theta2_cos', 'theta2_sin', 'theta1_dot', 'theta2_dot']

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
        ab_evs = AB_EvalStrategy(reward_solved, n_tree_evaluation_eps)
        ab_ses = AB_Sensitivity(ab_evs)
        inodes = [0]
        setzeros = [
            #  theta1        theta2      theta1 theta2
            #cos    sin    cos    sin    dot    dot
            [False, False, False, False, False, False]
            , [False, True, False, False, False, False]
            , [False, True, False, False, True, False]
        ]
        df_res_dict = generate_sens_pkls(ab_ses,inodes,setzeros, prefix='AB_')

    keys = list(df_res_dict.keys())
    df_res_lst = df_res_dict[keys[0]]
    plot_sensitivity(df_res_lst[0],pngdir,pngname)

def main_part_distance_plot():
    ab_evs = AB_EvalStrategy(reward_solved, n_tree_evaluation_eps)
    env = le.AcrobotEnvironment().get_environment()
    env.seed(seed=17)
    std = np.array([0.509362, 0.662336, 0.677396, 0.697312, 2.184065, 3.238214])
    with open("../../Experiments/OPCTs/Acrobot-v1_ITER_depth_1.pkl", "rb") as input:
        best_tree = pickle.load(input)
    tree_reward_mean, tree_reward_std, tree_samples = ab_evs.eval_opct(
        env, best_tree, n_tree_plane_eps, std
    )
    inodes = [0]
    pngdir = "png_dist"
    pngbase = "distp"
    plot_distance_ab(inodes, best_tree, tree_samples, pngdir, pngbase)

if __name__ == "__main__":
    main_part_sensitivity()


