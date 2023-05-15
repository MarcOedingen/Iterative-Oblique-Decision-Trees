import os
from abc import ABC, abstractmethod
from stable_baselines3 import DQN, PPO, TD3, DDPG


def get_current_path():
    return os.path.dirname(os.path.realpath(__file__))


class OracleStrategy(ABC):
    @abstractmethod
    def load_model(self, EnvironmentStrategy):
        pass


class DQN_Oracle(OracleStrategy):
    def load_model(self, EnvironmentStrategy):
        return DQN.load(f"{get_current_path()}/{EnvironmentStrategy}_dqn")


class PPO_Oracle(OracleStrategy):
    def load_model(self, EnvironmentStrategy):
        return PPO.load(f"{get_current_path()}/{EnvironmentStrategy}_ppo")


class TD3_Oracle(OracleStrategy):
    def load_model(self, EnvironmentStrategy):
        return TD3.load(f"{get_current_path()}/{EnvironmentStrategy}_td3")


class DDPG_Oracle(OracleStrategy):
    def load_model(self, EnvironmentStrategy):
        return DDPG.load(f"{get_current_path()}/{EnvironmentStrategy}_ddpg")
