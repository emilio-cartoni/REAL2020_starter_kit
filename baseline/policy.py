import numpy as np
from real_robots.policy import BasePolicy
from baseline.planner import Planner
from baseline.abstractor import currentAbstraction
import baseline.config as config
import baseline.explorer as exp


class State:
    '''
    This class represents a generic state.

    Args:

    Attributes:
        caller (class instance): where the caller instance are saved
        actionsData (list): where the actions are saved
    '''
    caller = None
    actionData = []


class DoAction(State):
    '''
    This class represents the state in which the agent performs an action

    Args:
        raw_actions (list): list of robot arm positions (one for each timestep)

    Attributes:
        n_timesteps (int): duration of action
        actionTimer (int): steps taken of the action
        raw_actions (list): where the positions of the robotic arm to be reached at each step are saved
    '''
    def __init__(self, action):
        self.n_timesteps = 1000
        self.actionTimer = -1
        self.action = action

    def step(self, observation, reward, done):
        '''
        It takes from the list of positions of the robotic arm the position to go to in the current step. In case the action is over, instantiate the goBackHome state

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''
        self.actionTimer += 1
        if self.actionTimer < self.n_timesteps:
            return self, self.action, self.actionTimer == (self.n_timesteps - 1)
        else:
            nextState = EndAction()
            return nextState.step(observation, reward, done)


class EndAction(State):
    '''
    This class represents the state in which the agent moves the robotic arm to the starting position

    Args:

    Attributes:
    '''

    def step(self, observation, reward, done):
        '''
        It save the current action and it passes into the ActionStart state.

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''
        keep_image = config.sim['save_images'] or config.abst['type'] == 'image'
        keep_mask = config.sim['save_masks'] or config.abst['type'] == 'mask'

        post_image = observation['retina'] if keep_image else None
        post_pos = observation['object_positions']
        post_mask = observation['mask'] if keep_mask else None
        post = (post_image, post_pos, post_mask)
        self.actionData += [post]
        self.caller.storeAction(self.actionData)

        State.actionData = []

        nextState = ActionStart()
        return nextState.step(observation, reward, done)


class ProposeNewAction(State):
    '''
    This class represents the state in which the agent randomly generates the action that will be performed

    Args:

    Attributes:
        point_1 (numpy.ndarray): two-dimensional point from which the trajectory starts
        point_2 (numpy.ndarray): two-dimensional point where the trajectory ends
    '''
    def __init__(self):

        action, debug = self.caller.explorer.selectNextAction()

        self.point_1 = action[0]
        self.point_2 = action[1]

    def step(self, observation, reward, done):
        '''
        Generates the trajectory based on the two pre-generated points and returns a DoAction instance instantiated with the trajectory

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''
        keep_image = config.sim['save_images'] or config.abst['type'] == 'image'
        keep_mask = config.sim['save_masks'] or config.abst['type'] == 'mask'

        pre_image = observation['retina'] if keep_image else None
        pre_pos = observation['object_positions']
        pre_mask = observation['mask'] if keep_mask else None
        pre = (pre_image, pre_pos, pre_mask)

        self.action = np.array([self.point_1, self.point_2])

        self.actionData += [pre, (self.point_1, self.point_2)]
        nextState = DoAction(self.action)
        return nextState.step(observation, reward, done)


class ActionStart(State):
    '''
    This class represents the state in which the agent checks whether he has been asked to achieve a goal through observations

    Args:

    Attributes:

    '''
    def step(self, observation, reward, done):
        '''
        Check if the goal is contained in the observations, if there was then it transit in the PlanAction status, otherwise it passes in the ProposeNewAction status

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''
        goal = observation['goal']
        if np.any(goal):
            nextState = PlanAction()
            return nextState.step(observation, reward, done)
        else:
            nextState = ProposeNewAction()
            return nextState.step(observation, reward, done)


class PlanAction(State):
    '''
    This class represents the state in which the agent plans the sequence of actions to do

    Args:

    Attributes:

    '''
    def step(self, observation, reward, done):
        '''
        The Planner class is used to obtain an action sequence to be performed to achieve the required goal. If the sequence is not empty then you go into the DoAction state
        with the first action of the sequence, otherwise you wait for a new goal, or it try a random action depending on the config file

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''
        keep_image = config.sim['save_images'] or config.abst['type'] == 'image'
        keep_mask = config.sim['save_masks'] or config.abst['type'] == 'mask'

        pre_image = observation['retina'] if keep_image else None
        pre_pos = observation['object_positions']
        pre_mask = observation['mask'] if keep_mask else None
        pre = (pre_image, pre_pos, pre_mask)

        goal = (observation['goal'], observation['goal_positions'], observation['goal_mask'])

        pre_abs = currentAbstraction(pre)
        goal_abs = currentAbstraction(goal)
        plan = self.caller.plan(goal_abs, pre_abs)

        if len(plan) > 0:
            action = plan[0][1]
            self.actionData += [pre, (action[0], action[1])]

            action = np.array([action[0], action[1]])
            nextState = DoAction(action)
            return nextState.step(observation, reward, done)
        else:
            if config.plan['try_random']:
                nextState = ProposeNewAction()
                return nextState.step(observation, reward, done)
            else:
                nextState = WaitForNewGoal(goal_abs)
                return nextState.step(observation, reward, done)


class WaitForNewGoal():
    '''
    This class represent the state where the agent wait to have a new goal.

    Args:
        observation (dict): dictionary of all the observations given by the environment

    Attributes:
        current_goal (numpy.ndarray): where the current goal is saved
        current_state (numpy.ndarray): where the current state of the world is saved

    '''
    def __init__(self, goal_abs):
        self.current_goal = goal_abs

    def step(self, observation, reward, done):
        '''
        It is checked whether the goal contained in the observations is different from that contained in the instantiation time.
        If it is, then switches to the ActionStart state

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''

        pre_abs, goal_abs = self.getCurrentStateAndGoal(observation)

        sameGoal = np.all(goal_abs == self.current_goal)

        if sameGoal:
            return self, None, False

        nextState = ActionStart()
        return nextState.step(observation, reward, done)

    def getCurrentStateAndGoal(self, observation):
        '''
        Extrapolate the current state and the desired state of the world

        Args:
            observation (dict): dictionary of all the observations given by the environment

        Returns:
            current state, desidered state (numpy.ndarray,numpy.ndarray)
        '''
        pre = (observation['retina'], observation['object_positions'], observation['mask'])
        goal = (observation['goal'], observation['goal_positions'], observation['goal_mask'])
        pre_abs = currentAbstraction(pre)
        goal_abs = currentAbstraction(goal)
        return pre_abs, goal_abs


class Baseline(BasePolicy):
    '''
    This class allows you to gradually release the description of the states contained in the actions

    Args:

    Attributes:
        allActions (list): where the actions of the intrinsic phase are saved
        state (ActionStart state instance): where the current state of the agent is saved
        planner (None): where the instance of the planner class will be saved in the extrinsic phase
        goal (None): where the desired state of the world will be saved
        plan_sequence (list): where the planning result is saved
        n_plans (int): where the plans counter is saved
        action_space (Box class instance): where the generator of the points used to create the actions trajectories is saved
        explorer (Explorer class instance): where the Explorer class instance is saved
    '''
    def __init__(self, action_space):
        self.allActions = []
        self.state = ActionStart()
        State.caller = self
        self.planner = None
        self.goal = None
        self.plan_sequence = []
        self.n_plans = 0
        self.action_space = action_space['macro_action']
        self.explorer = exp.RandomExploreAgent(self.action_space)

    def storeAction(self, actionData):
        '''
        Adds the action just performed to the actions saved so far

        Args:
            actionData (tuple(numpy.ndarray,numpy.ndarray,numpy.ndarray)): the vectors represent precondition, action points and postcondition respectively

        Returns:

        '''
        self.allActions += [actionData]

    def save(self, fileName):
        '''
        Saving the list of actions in a file which will be called fileName

        Args:
            fileName (string)

        Returns:

        '''
        if not config.sim['save_images']:
            for action in self.allActions:
                pre = action[0]
                action[0] = (None, pre[1], pre[2])
                post = action[2]
                action[2] = (None, post[1], post[2])

        if not config.sim['save_masks']:
            for action in self.allActions:
                pre = action[0]
                action[0] = (pre[0], pre[1], None)
                post = action[2]
                action[2] = (post[0], post[1], None)

        np.save(fileName, self.allActions)

    def step(self, observation, reward, done):
        '''
        Calculate the maximum distance that can occur with probability p

        Args:
            observation (dict): dictionary of all the observations given by the environment
            reward (float)
            done (boolean)

        Returns:
            (State instance, joints position, bool): where bool is True only when the robotic arm is in the home position
        '''
        self.state, action, render = self.state.step(observation, reward, done)

        return {'macro_action': action, 'render': render}

    def plan(self, goal_abs, pre_abs):
        '''
        Use the Planner class to plan the list of actions to be performed to reach the goal

        Args:
            goal_abs (numpy.ndarray): desidered state of the world
            pre_abs (numpy.ndarray): current state of the world

        Returns:
            plan_sequence (list): list of actions planned by the planner
        '''
        goalChanged = not(np.all(self.goal == goal_abs))
        self.goal = goal_abs
        no_sequence = len(self.plan_sequence) == 0

        if config.plan['replan'] or goalChanged or no_sequence:
            self.plan_sequence = self.planner.plan(goal_abs, pre_abs, alg=config.plan['type'])
        else:
            self.plan_sequence = self.plan_sequence[1:]

        return self.plan_sequence

    def end_intrinsic_phase(self, observation, reward, done):
        self.step(observation, reward, done)

    def start_extrinsic_phase(self):
        '''
        Instantiate the Planner class with the actions collected in the intrinsic phase, or with the actions saved in the file written in the config file

        Args:

        Returns:

        '''

        self.save("./{}".format(np.random.randint(0, 10000)))

        allActions = self.allActions

        if config.sim['use_experience_data']:
            allActions = np.load(config.sim['experience_data'], allow_pickle=True)

        allAbstractedActions = [[currentAbstraction(a[0]), a[1], currentAbstraction(a[2])] for a in allActions]

        self.planner = Planner(allAbstractedActions)
        del allActions
        del allAbstractedActions

    def start_extrinsic_trial(self):
        self.state = ActionStart()

    def end_extrinsic_trial(self, observation, reward, done):
        self.step(observation, reward, done)
