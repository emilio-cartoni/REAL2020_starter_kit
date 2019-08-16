import gym
import numpy as np
import real_robots
from my_controller import SubmittedPolicy
import os

DATASET_PATH=os.getenv("AICROWD_DATASET_PATH", "./goals.npy.npz")

result, detailed_scores = real_robots.evaluate(
                SubmittedPolicy,
                intrinsic_timesteps=40,
                extrinsic_timesteps=40,
                extrinsic_trials=5,
                visualize=False,
                goals_dataset_path=DATASET_PATH
            )
#  NOTE : You can find a sample goals.npy.npz file at
#
#  https://aicrowd-production.s3.eu-central-1.amazonaws.com/misc/REAL-Robots/goals.npy.npz
print(result)
print(detailed_scores)
