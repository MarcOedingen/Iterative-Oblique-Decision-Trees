import numpy as np
from gym.core import ObservationWrapper
from gym.spaces import Box


class PendObs2DWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        max_speed = 8
        high = np.array([np.pi, max_speed], dtype=np.float32)
        self.observation_space = Box(shape=(2,), low=-high, high=high)

    def observation(self, obs):
        """Returns the modified observation [theta,thetadot]"""
        obs = np.float64(obs)  # important for numeric accuracy
        theta = np.arctan2(obs[1], obs[0])
        # theta = np.arcsin(obs[1])
        # if obs[0]<0:
        #     if obs[1]>0: theta = np.pi - theta
        #     else: theta = -np.pi - theta
        assert np.abs(obs[0] - np.cos(theta)) < 1e-7, "Error cos(theta) {}".format(
            np.abs(obs[0] - np.cos(theta))
        )
        assert np.abs(obs[1] - np.sin(theta)) < 1e-7, "Error sin(theta) {}".format(
            np.abs(obs[1] - np.sin(theta))
        )
        assert -np.pi <= theta, "Error theta={} is smaller than -pi".format(theta)
        assert theta <= np.pi, "Error theta={} is larger than +pi".format(theta)
        return np.array([theta, obs[2]])