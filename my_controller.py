import gym
import numpy as np
import time
import real_robots
from real_robots.policy import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def step(self, observation, reward, done):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action


SubmittedPolicy=RandomPolicy