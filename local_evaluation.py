import gym
import numpy as np
import real_robots
from my_controller import SubmittedPolicy
import os

DATASET_PATH=os.getenv("AICROWD_DATASET_PATH", "./data/goals-REAL2020-s2020-50-1.npy.npz")

result, detailed_scores = real_robots.evaluate(
                SubmittedPolicy,
                environment='R1',
                action_type='macro_action',
                n_objects=1,
                intrinsic_timesteps=15e6,
                extrinsic_timesteps=10e3,
                extrinsic_trials=50,
                visualize=False,
                goals_dataset_path=DATASET_PATH
            )
#  NOTE : You can find a sample goals.npy.npz file at
#
#  https://aicrowd-production.s3.eu-central-1.amazonaws.com/misc/REAL-Robots/goals.npy.npz
print(result)
print(detailed_scores)
