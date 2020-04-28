import numpy as np
import baseline.config as config
import baseline.priorityQueue as pq

def currentAbstraction(obs):
    '''
    Extract the coordinates from the input (observation) taking only x, y, z depending on the config file

    Args:
        obs (list): list of observations in different form (image, coordinates, mask)

    Returns:
        coordinates list (list)          
    '''
    if config.abst['type'] == 'pos':
        return abstractionFromPos(obs[1])
    if config.abst['type'] == 'pos_noq':
        return abstractionFromPosNoQ(obs[1])

def abstractionFromPos(pos_dict):
    '''
    Extract the coordinates from the input 

    Args:
        pos_dict (dictionary): dictionary = {object0:coords0,...,objectN:coordsN}

    Returns:
        coordinates list (list): [coords0,...,coordsN]         
    '''
    abst = np.hstack([pos_dict[obj] for obj in ['cube','tomato','mustard'] if obj in pos_dict])
    abst = np.round(abst, config.abst['precision'])
    return abst

def abstractionFromPosNoQ(pos_dict):
    '''
    Extract the coordinates from the input taking only x, y, z 

    Args:
        pos_dict (dictionary): dictionary = {object0:coords0,...,objectN:coordsN}

    Returns:
        coordinates list (list): [coords0[:2],...,coordsN[:2]]         
    '''
    abst = np.hstack([pos_dict[obj][:2] for obj in ['cube','tomato','mustard'] if obj in pos_dict])
    abst = np.round(abst, config.abst['precision'])
    return abst


class DynamicAbstractor():
    '''
    This class allows you to gradually release the description of the states contained in the actions.

    Args:
        actions (list): a list of (precondition, action, postcondition), where conditions are float lists 

    Attributes:
        actions (list): where the input actions are saved
        dictionary_abstract_actions (dict): where abstract actions are saved. The keys will be the levels of abstraction and the values ​​will be the lists of abstract actions
        lists_significative_differences (list): where significant distances are saved for each condition variable. These will be used to relax conditions.
    '''
    def __init__(self, actions):
        if len(actions[0]) != 3:
            print("Legal actions consist of (precondition,action,postcondition)")
            return None
        
        if not isinstance(actions[0][0], type(np.array([]))) or not isinstance(actions[0][2], type(np.array([]))):
            print("Each conditions have to be numpy.ndarray")
            return None         
        
        self.actions = actions
        self.dictionary_abstract_actions = {}

        #For each variable in actions condition it add a list to put the significative differences 
        condition_dimension = len(actions[0][0])
        self.lists_significative_differences = [[] for i in range(condition_dimension)]

        ordered_differences_queues = [pq.PriorityQueue() for i in range(condition_dimension)]   

        differences = abs(np.take(self.actions,0,axis=1)-np.take(self.actions,2,axis=1))
        for i in range(condition_dimension): 
            for j in range(len(actions)): 
                ordered_differences_queues[i].enqueue(None, differences[j][i])         

        actions_to_remove = int(np.floor(len(actions)*config.abst['percentage_of_actions_ignored_at_the_extremes']))
        
        for i in range(condition_dimension): 
            sup = ordered_differences_queues[i].get_queue_values()
            for j in np.linspace(actions_to_remove,len(actions)-actions_to_remove, config.abst['total_abstraction']).round(0):
                self.lists_significative_differences[i] += [sup[int(j)]]
            

    def get_abstraction(self, abstraction_level):
        '''
        Calculate the vector representing the abstraction required as input in each variable

        Args:
            abstraction_level (int)

        Returns:
            distances (numpy.ndarray): where the i-th cell of the vector represents the input abstraction on the i-th variable         
        '''    
        return np.array([self.lists_significative_differences[i][abstraction_level] for i in range(len(self.lists_significative_differences))])

    def get_dist(self, cond1, cond2):
        '''
        Calculate the amount of abstractions that distances the two input conditions

        Args:
            cond1 (list of float)
            cond2 (list of float)

        Returns:
            distance (int)         
        '''      
        return np.sum(abs(cond1-cond2))   

