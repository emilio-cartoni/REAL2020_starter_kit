import numpy as np
import real_robots
from my_controller import SubmittedPolicy
import os

#########################################################
### Please specify the action_type and n_objects here ###
###     These will be used during your evaluation     ###
#########################################################
EVALUATION_ACTION_TYPE = 'macro_action'
EVALUATION_N_OBJECTS = 1
DATASET_PATH = "./data/allGoalsPyRepRestricted.npy.npz"

result, detailed_scores = real_robots.evaluate(
                SubmittedPolicy,
                environment='PyRep',
                action_type=EVALUATION_ACTION_TYPE,
                n_objects=EVALUATION_N_OBJECTS,
                intrinsic_timesteps=15000,
                extrinsic_timesteps=5,
                extrinsic_trials=50,
                visualize=False,
                goals_dataset_path=DATASET_PATH
         #       , selected_goals=[1, 4, 6, 7, 8, 10, 15, 18]
            )

print(result)
print(detailed_scores)

result_data = {'result': result,
               'detailed_scores': detailed_scores}

run_id = np.random.randint(0, 1000000)

np.save("./R{}-{:.3f}".format(run_id, result['score_total']), result_data)
