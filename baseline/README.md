# Baseline 

## Context description
Before describing the system, let's describe the context in which it operates:
There is an Agent-Environment context in which the agent has control of a robotic arm and the environment includes a table and a cube, in addition to the robotic arm. The agent must learn to use the robotic arm in such a way as to be able to move the cube at any point on the table without any prior knowledge. The agent will therefore have to collect different skills and, on request, be able to use them to move the object from one position to another.

[Extrinsic phase](https://github.com/emilio-cartoni/REAL2020_starter_kit/blob/master/baseline/media/extrinsic_phase_very_little_video_compressed.gif)

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
