import gym
import numpy as np
import real_robots
from my_controller import SubmittedPolicy
import os

DATASET_PATH=os.getenv("AICROWD_DATASET_PATH", "./data/goals-REAL2020-s2020-50-1.npy.npz")

result, detailed_scores = real_robots.evaluate(
                SubmittedPolicy,
                intrinsic_timesteps=100000,
                extrinsic_timesteps=10000,
                extrinsic_trials=50,
                visualize=False,
                goals_dataset_path=DATASET_PATH
            )
#  NOTE : You can find a sample goals.npy.npz file at
#
#  https://aicrowd-production.s3.eu-central-1.amazonaws.com/misc/REAL-Robots/goals.npy.npz
print(result)
print(detailed_scores)
