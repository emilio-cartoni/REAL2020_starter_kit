# Baseline 

## Context description
Before describing the system, let's describe the context in which it operates:

There is an Agent-Environment context in which the agent has control of a robotic arm and the environment includes a table and a cube, in addition to the robotic arm. The agent must learn to use the robotic arm in such a way as to be able to move the cube at any point on the table without any prior knowledge. The agent will therefore have to collect different skills and, on request, be able to use them to move the object from one position to another.

![ciaone](https://github.com/emilio-cartoni/REAL2020_starter_kit/blob/master/baseline/media/extrinsic_phase_little_video.gif)

## Simplifications problem
In this version of the baseline we have made several simplifications to the problem:

 1. At each step it is possible to have the current position (x, y) of the object.
 2. A special action is available which, given two points in two-dimensional space (x1, y1) and (x2, y2), generates a trajectory on the table that starts from (x1, y1) and ends in (x2, y2). This actions are showed in previous video.



## Vocabulary
Action = a triple (precondition, trajectory, postcondition)

World state = position (x, y) of the cube

## Approach to the problem
The problem was addressed with several components: Policy, Explorer, Abstractor and Planner. We will expose the system with a top-down approach, then we will first see the Policy component and lastly the Plan.

![Architecture](https://github.com/emilio-cartoni/REAL2020_starter_kit/blob/master/baseline/media/architecture.svg)

### Policy component
The Policy component represents the agent and defines its behavior depending on the phase in which it is located. In the intrinsic phase it will deal with exploring (using explorer) the space of actions in order to collect as many actions as possible. In the extrinsic phase he will deal with using the collected actions to generate sequences (using planner and abstractor) of actions that lead to the desired state.

![Policy flow](https://github.com/emilio-cartoni/REAL2020_starter_kit/blob/master/baseline/media/class_policy.svg)

### Explorer component
The Explorer component deals with the exploration of the two-dimensional space above the table. To do this it generates two points in the two-dimensional space [(x1, y1), (x2, y2)] which then the Policy component uses to create a trajectory on the table. Before making the movement corresponding to the trajectory, the agent saves the current state of the world so that the triple (precondition, trajectory, postcondition) can be saved later, where postcondition is the state of the world after performing the action . This exploration phase corresponds to the intrinsic phase, where the agent limits himself to collecting actions to perform at his best in the extrinsic phase.

### Abstractor component
The Abstractor component takes the actions collected in the intrinsic phase and for each state variable extrapolates the most significant distances. In other words, for each variable we have a set of points and these points have a distance between them. Each distance will have a certain frequency and therefore a relevance, we are interested in the latter. To derive the most relevant distances, the abstractor calculates all the distances between precondition and postcondition. These will represent all the distances experienced as an effect of actions, after which it sorts these distances and samples them equally distributed according to the number of abstractions required by the config file. These differences will allow the agent to discretize the state space and to distinguish on the basis of the current abstraction between actions that modify the state of the world and actions that do not.

### Planner component
The Planner component has an instance of the abstraction tool available which will allow it to access the distance associated with each level of abstraction. The planner uses the abstractor to make ever more abstract plans until he finds a plane, where "more and more abstract" means more collapse between states. Before moving on to planning, you prepare the actions in the following way: calculate the distance between each precondition and postcondition and if this is smaller than the distance associated with the current abstraction, then the action in question is ignored. The abstraction is done incrementally from the bottom up until a plane is found. This will ensure that you prefer plans with lower abstraction.

## Config file
The agent's behavior also depends on hyperparameters specified in the config.yaml file. These parameters are divided into 3 different categories: abstraction, planner and simulation. Below we explain their semantics:
 - Abstraction:
   - precision: allows to truncate the numbers contained in the vector representing the state of the world to the i-th digit after the comma.
   - type: with the possible values ​​[pos, pos_noq] allows you to specify if you want to use the position of the cube with or without quaternion
   - total_abstraction: allows you to specify how many equally distributed samples you want to take in the set of differences experienced between precondition and postcondition
   - percentage_of_actions_ignored_at_the_extremes: allows you to specify the percentage that you want to ignore at the extremes of the actions (sorted). This variable was made in such a way as to be able to eliminate the most unlikely actions to be performed.
 - Planner:
   - replan: allows you to specify if you want the agent to replan after each action performed
   - try_random: allows you to specify if you want the agent to do random actions if he does not find a plan
   - type: with the possible values ​​[mega, noplan] it allows to specify if you want to plan, or not
   - action_size: allows you to specify the length of the actions that the planner takes into consideration in order to give an upper limit to the sequence of actions to be planned
 - Simulation:
   - experience_data: allows you to specify the file to be used to face the extrinsic phase
   - extrinsic_steps: allows you to specify the number of steps dedicated to each goal with the same “action_size” aim 
   - save_images: allows you to specify if you want the triple of the actions to contain in preconditions and postconditions the image of the state of the world rather than just the coordinates of the cube.
   - save_masks: same meaning as the previous one but in this case you specify if you want a matrix as large as that of the images but with the object contained in the cell in question specified.
   - use_experience_data: allows you to specify if you want to use the file specified in "experience_data", or (if false) if you want to use the actions collected in the intrinsic phase just carried out

