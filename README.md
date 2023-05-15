# Iterative-Oblique-Decision-Trees

This repository contains all experiments from the paper [Engelhardt2023] and allows the reproduction of the results.

[Engelhardt2023]: Engelhardt, R.C.; Oedingen, M.; Lange, M.; Wiskott, L.; Konen, W. Iterative Oblique Decision Trees Deliver Explainable RL Models. Preprints.org 2023, 2023041162. https://doi.org/10.20944/preprints202304.1162.v1.

## Structure
All experiments presented are located in the 'Experiments' folder. Further, the 'Initializing' folder contains the code for the initialization of the oracles and environments.

## Experiments
In total, there are six different experiments X:

* X_1: Bounding Box (BB)
* X_2: Classification and Regression Trees (CART)
* X_3: Iterative Classification and Regression Trees (ITER_CART)
* X_4: Oblique Predictive Clustering Trees (OPCT)
* X_5: Iterative Oblique Predictive Clustering Trees (ITER_OPCT)
* X_6: Sensitivity Analysis (SENS)

Each experiment can be executed in one of seven environments Y:

* Y_1: Acrobot (AB)
* Y_2: CartPole (CP)
* Y_3: CartPoleSwingUp (CPSU)
* Y_4: LunarLander (LL)
* Y_5: MountainCar (MC)
* Y_6: MountainCarContinuous (MCC)
* Y_7: Pendulum (PEND)

To execute a specific experiment in an environment, run the following command in the terminal from the root directory of this repository:
``` 
python3 -m Experiments.Y.Y_X
``` 
For example, to run Iterative Oblique Predictive Clustering Trees in the CartPole environment, run the following command:
```
python3 -m Experiments.CartPole.CP_ITER_OPCT
```

## Results
After running an experiment, the results are stored in different subfolders in the parent folder 'Experiments'. The results are stored in the following subfolders:

* CARTs: Contains the Classification and Regression Trees solving an environment for experiments X_2 and X_3.
* OPCTs: Contains the Oblique Predictive Clustering Trees solving an environment for experiments X_1, X_4, and X_5.
* Rewards_NPZ: Contains the rewards of trees and oracles solving an environment for experiments X_1, ..., X_5.
* Times_Samples: Contains the times and samples of trees solving an environment for experiments X_1, ..., X_5.