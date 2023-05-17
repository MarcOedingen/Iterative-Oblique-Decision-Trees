import gym
from abc import ABC, abstractmethod
from Initializing.Environments.LL_Init import LunarLanderBB
from Initializing.Environments.Pend2D_Wrapper import PendObs2DWrapper
from Initializing.Environments.CPSU_discrete import CartPoleSwingUpV1


class EnvironmentStrategy(ABC):
    @abstractmethod
    def get_environmentShort(self):
        pass

    @abstractmethod
    def get_environment(self):
        pass

    @abstractmethod
    def get_reward_threshold(self):
        pass


class AcrobotEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "ab"

    def get_environment(self):
        return gym.make("Acrobot-v1")

    def get_reward_threshold(self):
        return -100


class Cartpolev1Environment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "cp"

    def get_environment(self):
        return gym.make("CartPole-v1")

    def get_reward_threshold(self):
        return 475


class CartpoleSwingUpEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "cpsu"

    def get_environment(self):
        return gym.wrappers.TimeLimit(
            CartPoleSwingUpV1(),
            max_episode_steps=1000,
        )

    def get_reward_threshold(self):
        return 840


class MountainCarEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "mc"

    def get_environment(self):
        return gym.make("MountainCar-v0")

    def get_reward_threshold(self):
        return -110


class MountainCarContinuousEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "mcc"

    def get_environment(self):
        return gym.make("MountainCarContinuous-v0")

    def get_reward_threshold(self):
        return 90

class LunarLanderEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "ll"

    def get_environment(self):
        return gym.wrappers.TimeLimit(
            gym.make("LunarLander-v2"), max_episode_steps=1000
        )

    def get_reward_threshold(self):
        return 200


class LunarLandeBoundingBoxEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "llbb"

    def get_environment(self):
        return gym.wrappers.TimeLimit(LunarLanderBB(), max_episode_steps=1000)

    def get_reward_threshold(self):
        return 200

class Pendulumv1Environment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "pend"

    def get_environment(self):
        return gym.make("Pendulum-v1")

    def get_reward_threshold(self):
        return -170

class Pendulum2DEnvironment(EnvironmentStrategy):
    def get_environmentShort(self):
        return "pend2d"

    def get_environment(self):
        return PendObs2DWrapper(gym.make("Pendulum-v1"))

    def get_reward_threshold(self):
        return -200
