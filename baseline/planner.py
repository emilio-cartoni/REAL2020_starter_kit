import baseline.node as nodeClass
import baseline.priorityQueue as pqClass
import numpy as np
import baseline.abstractor as abstr
import baseline.config as config


class Planner():
    '''
    This class implements a planning algorithm which is used by passing it the abstract actions, the level of abstraction and the maximum length of the plan. 
	The abstraction is done from the smallest to the largest in order to prefer minimal abstractions. The length of the plan is given incrementally so as to prefer short plans.

    Args:
        actions (list): a list of actions with format (precondition, action, postcondition)

    Attributes:
        actions (list): where the actions are saved
        last_goal (numpy.ndarray): where the current goal is saved to intercept the moment when it changes
        abstractor (DynamicAbstractor instance): this instance allows you to have the abstract actions to pass to the planner
        stop_plan (bool): a flag that allows you not to replan if a plan has not been found in the previous step and the goal has not changed
        actions_dicts (dict): where for each pair (i-th action, k-th level of abstraction) are saved all the actions with abstract precondition at the k-th 
                                level of abstraction compatible with the postcondition of the i-th abstract action at the k-th level of abstraction
    '''
    def __init__(self, actions = None):
        self.actions = actions
        self.abstractor = abstr.DynamicAbstractor(self.actions)
        self.last_goal = np.array([])
        self.stop_plan = False
        self.actions_dicts = {}

    def plan(self, goal, start, actions=None, alg='mega'):
        '''
        Search a sequence of actions that bring the current state to the goal state by giving priority to the smaller ones. Based on the number of steps dedicated to 
        the extrinsic phase for each goal, it calculates the maximum possible length L of the plan and requires at the planning algorithm to have a plan of length 1,2,...,L. 
        Before increasing the length of the plane, try to use all the available abstractions giving priority to the smallest ones.

        Args:
            goal (numpy.ndarray): vector representing the desired state
            start (numpy.ndarray): vector representing the current state
            actions (float): a list of actions with format (precondition, action, postcondition)

        Returns:
            sequence (numpy.ndarray): a list of actions with format [(precondition, action, postcondition),...,(precondition, action, postcondition)] where the first action 												represents the current state and the last action the desired state    
        '''
        if alg == 'mega':
            if not np.all(self.last_goal == goal):
                self.last_goal = goal
                self.stop_plan = False
                self.plan_size = int(np.floor(config.sim['extrinsic_steps']/1200))
            else:
                self.plan_size -= 1

            if self.stop_plan:
                return []

            seq = []
            for lev_depth in range(1,self.plan_size+1):
                print("Depth level: {}".format(lev_depth))

                levelAbstraction = 0
                seq = []
                res = []
                while levelAbstraction < self.abstractor.max_abstr:
                    n = self.abstractor.max_abstr - levelAbstraction - 1   
                    n = levelAbstraction           
                    print("Abstraction level: {}".format(n))
                    
                    actions = self.abstractor.get_actions_abstr(n)
                    abstr_goal = self.abstractor.get_cond_abstr(goal, n)
                    abstr_curr = self.abstractor.get_cond_abstr(start, 0)
                    res = self.forward_planning_with_prior_abstraction(abstr_goal, abstr_curr, 
                                                                            actions, n, depth=lev_depth)
                    if res:
                        return res
                    if not res and seq:
                        return seq
                    elif res:
                        print("Solution")
                        seq = res

                    levelAbstraction += 1

                if seq:
                    return seq

            if seq is None or not seq:
                self.stop_plan = True
                return []

            return seq
        if alg == 'noplan':
            return []
        raise NotImplementedError

    def forward_planning_with_prior_abstraction(self, goal_image, current, actions, lev_abstr, depth=None):
        '''
        Forward planning algorithm divided into three steps: (1) It find all the actions that have a precondition equal to the current state of the world and 
        it put them in the frontier set, (2) it add for each action in the frontier set the actions with precondition compatible with the postcondition of the 
        action in question, (3) if it find a sequence of actions that reach to the desired state of the world, then it returns that sequence

        Args:
            goal_image (numpy.ndarray): vector representing the desired abstract state at level lev_abstr 
            current (numpy.ndarray): vector representing the current abstract state at level 0 (to be precise)
            actions (float): a list of actions with format (abstract precondition, action, abstract postcondition)
            lev_abstr (int): abstraction level
            depth (int): maximum sequence length

        Returns:
            sequence (numpy.ndarray): a list of actions with format [(precondition, action, postcondition),...,(precondition, action, postcondition)] where the first action 												represents the current state and the last action the desired state          
        '''
        if np.all(goal_image == current):
            return None

        q = pqClass.PriorityQueue()

        frontier = set()
        for i in range(len(actions)):
            pre, act, post = actions[i] 
            if np.all(pre == post):
                continue
            if np.all(current == pre):
                node = nodeClass.Node(i, self.abstractor.get_dist(post, goal_image), self.abstractor.get_dist(pre, post), None)
                q.enqueue(node, node.get_value_plus_cost())
                frontier.add(i)
        print("Add {} initial states".format(len(frontier)))
        

        visited = set()
        while not q.is_empty():
            if len(visited) % 100 == 0 or len(visited) == 0:
                print("Visited actions: {} Ready actions in queue: {}".format(len(visited),len(frontier)))
            node, value = q.dequeue()

            if depth is not None:
                if node.get_depth() == depth:
                    frontier.remove(node.get_attribute())
                    continue 

            if node.get_attribute() in visited:
                continue            

            
            if node is None:
                break

            if np.all(actions[node.get_attribute()][2] == goal_image) and node.get_value() < self.abstractor.get_dist(current, goal_image):
                frontier.remove(node.get_attribute())
                visited.add(node.get_attribute())
                break

            frontier.remove(node.get_attribute())
            visited.add(node.get_attribute())

            post = actions[node.get_attribute()][2]
            
            if (node.get_attribute(),lev_abstr) in self.actions_dicts: 
                l1 = self.actions_dicts[(node.get_attribute(),lev_abstr)]
            else:
                self.actions_dicts[(node.get_attribute(),lev_abstr)] = []
                l1 = []
                for j in range(len(actions)):
                    pre1 = actions[j][0]
                    post1 = actions[j][2]
                    if np.all(pre1 == post1):
                        continue 
                    if j == node.get_attribute():
                        continue
                    if np.all(pre1 == post):
                        self.actions_dicts[(node.get_attribute(),lev_abstr)] += [j]
                l1 = self.actions_dicts[(node.get_attribute(),lev_abstr)]

            l2 = []
            for i in l1:
                if not i in visited:
                    l2 += [(i,actions[i])]
         
            for a, action in l2:
                pre1 = action[0]
                post1 = action[2]

                if not a in visited and not a in frontier:
                    node1 = nodeClass.Node(a, self.abstractor.get_dist(post1, goal_image), self.abstractor.get_dist(pre1, post1), node)
                    q.enqueue(node1,node1.get_value_plus_cost())
                    frontier.add(a)
            
                elif a in frontier:
                    node1 = nodeClass.Node(a, self.abstractor.get_dist(post1, goal_image), self.abstractor.get_dist(pre1, post1), node)
                    q.replace_if_better(node1,node1.get_value_plus_cost())


        sequence = []
        flag = 0
        if not visited:
            return sequence

        if node is not None and not q.is_empty():
            while node.get_father() is not None:
                sequence += [[self.abstractor.actions[node.get_attribute()][0], self.abstractor.actions[node.get_attribute()][1], self.abstractor.actions[node.get_attribute()][2]]] + sequence
                node = node.get_father()
            sequence = [[self.abstractor.actions[node.get_attribute()][0], self.abstractor.actions[node.get_attribute()][1], self.abstractor.actions[node.get_attribute()][2]]] + sequence 
        
        if not sequence:
            return []
        return sequence

