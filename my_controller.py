import gym
import numpy as np
import time
import real_robots
from real_robots.policy import BasePolicy
from baseline.baseline import Baseline

class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space

    def step(self, observation, reward, done):
        return self.action_space.sample()


SubmittedPolicy=Baseline
