"""Reproduction of results published in
Verma, A., Murali, V., Singh, R., Kohli, P. & Chaudhuri, S. (2018).
Programmatically Interpretable Reinforcement Learning.
Proceedings of the 35th International Conference on Machine Learning,
in Proceedings of Machine Learning Research 80:5045-5054
Available from https://proceedings.mlr.press/v80/verma18a.html.
"""

import gym
import numpy as np

EPS = 100
SEED = 42

res = np.zeros(EPS)
env = gym.make("Acrobot-v1")

for i in range(EPS):
    env.seed(i+SEED)
    s = env.reset()
    done = 0
    ret = 0
    while not done:
        # Policy stated in Figure 8 of supplementary material
        # http://proceedings.mlr.press/v80/verma18a/verma18a-supp.pdf
        if 0.1357 + s[4] < 0:
            a = 2
        else:
            a = 0

        s, r, done, _ = env.step(a)
        ret += r

    res[i] = ret

print(f"Average return in {EPS} episodes: <R> = {np.mean(res)} +/- {np.std(res)}")
