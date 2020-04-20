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
        gaussian_approximation (boolean): if true, then gaussine is used to estimate the significant distances experienced for each variable of conditions

    Attributes:
        actions (list): where the input actions are saved
        dictionary_abstract_actions (dict): where abstract actions are saved. The keys will be the levels of abstraction and the values ​​will be the lists of abstract actions
        dictionary_distance_actions (dict): where distances between actions are saved. The keys will be the levels of abstraction and the values ​​will be the lists of abstract actions
        lists_significative_differences (list): where significant distances are saved for each condition variable. These will be used to relax conditions.
        max_abstr (int): where the number of abstractions is saved.
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
        self.dictionary_distance_actions = {}

        #For each variable in actions condition it add a list to put the significative differences 
        precondition_dimension = len(actions[0][0])
        self.lists_significative_differences = [[] for i in range(precondition_dimension)]

        ordered_differences_queues = [pq.PriorityQueue() for i in range(precondition_dimension)]   

        for i in range(precondition_dimension): 
            for j in range(len(actions)): 
                ordered_differences_queues[i].enqueue(None, abs(actions[j][0][i]-actions[j][2][i]))         

        points = [[] for i in range(precondition_dimension)]
        for i in range(len(actions)):   
            for j in range(len(points)): 
                points[j] += [actions[i][0][j],actions[i][2][j]] 
        
        max_len = 0        
        for i in range(len(points)):
            if max_len < np.max(points[i])-np.min(points[i]):
                max_len = np.max(points[i])-np.min(points[i])

        proporsions = [max_len/(np.max(points[i])-np.min(points[i])) for i in range(precondition_dimension)]

        n_abstractions = 200
        jump_size = int(np.floor(len(actions)/n_abstractions))
        five_percent = int(np.floor(len(actions)*0.05))
        for i in range(precondition_dimension): 
            sup = ordered_differences_queues[i].get_queue_values()
            for j in np.linspace(five_percent,len(actions)-five_percent,n_abstractions).round(0):
                
                self.lists_significative_differences[i] += [sup[int(j)]]

        self.max_abstr = n_abstractions
            

    def get_cond_abstr(self, condition, abstraction_level):
        '''
        Transforms the input float vector into a vector of intervals representing the abstract condition at k level

        Args:
            condition (list): float list
            abstraction_level (int)

        Returns:
            intervals vector (numpy.ndarray of Interval istances)         
        '''
        intervals_vector = []
        for j in range(len(condition)):     
            if abstraction_level < len(self.lists_significative_differences[j]):
                k = self.lists_significative_differences[j][abstraction_level]/2 
                intervals_vector += [Interval(condition[j]-k, condition[j]+k)]
            else:
                k = self.lists_significative_differences[j][-1]/2
                intervals_vector += [Interval(condition[j]-k, condition[j]+k)]
        return np.array(intervals_vector)

    def get_actions_abstr(self, abstraction_level):
        '''
        Transform all conditions contained in actions with get_cond_abstr(abstraction_level)

        Args:
            abstraction_level (int)

        Returns:
            actions list (list)         
        '''
        if abstraction_level in self.dictionary_abstract_actions.keys():
            return self.dictionary_abstract_actions[abstraction_level]
        else:
            newActions = []
            for i in range(len(self.actions)):
                action = self.actions[i]
                pre,act,post = action
                intervalsPre = self.get_cond_abstr(pre, abstraction_level)
                intervalsPost = self.get_cond_abstr(post, abstraction_level)
                newActions += [[np.array(intervalsPre),act,np.array(intervalsPost)]]
            self.dictionary_abstract_actions[abstraction_level] = newActions
            return newActions

    def get_action_abstr(self, action, abstraction_level):
        '''
        Transform the conditions contained in action with get_cond_abstr(abstraction_level)

        Args:
            abstraction_level (int)
            action (list)

        Returns:
            actions list (list)         
        '''
        pre,act,post = action
        intervalsPre = self.get_cond_abstr(pre, abstraction_level)
        intervalsPost = self.get_cond_abstr(post, abstraction_level)
        return [np.array(intervalsPre),act,np.array(intervalsPost)]

    def get_cond_deabstr(self, cond):
        '''
        Transforms a list of Interval istances into a list of corresponding floats

        Args:
            cond (list of Intervals)

        Returns:
            cond (list of floats)         
        '''
        cond1 = cond.copy()
        for i in range(len(cond)):     
            cond1[i] = cond1[i].a + (cond1[i].b - cond1[i].a)/2
        return cond1

    def get_dist(self, cond1, cond2):
        '''
        Calculate the amount of abstractions that distances the two input conditions

        Args:
            cond1 (list of Intervals)
            cond2 (list of Intervals)

        Returns:
            distance (int)         
        '''
        cond1 = self.get_cond_deabstr(cond1)
        cond2 = self.get_cond_deabstr(cond2)
        
        if str(cond1)+str(cond2) in self.dictionary_distance_actions:
            return self.dictionary_distance_actions[str(cond1)+str(cond2)]

        if str(cond2)+str(cond1) in self.dictionary_distance_actions:
            return self.dictionary_distance_actions[str(cond2)+str(cond1)]        

        dist = 0
        for i in range(len(self.actions[0][0])):
            for k in range(len(self.lists_significative_differences[i])):
                arr0 = self.get_cond_abstr(cond1,k)
                arr1 = self.get_cond_abstr(cond2,0)
                if arr0[i] == arr1[i]:
                    dist += k
                    break
                elif k == len(self.lists_significative_differences[i])-1:
                    dist += len(self.lists_significative_differences[i])

        self.dictionary_distance_actions[str(cond1)+str(cond2)] = dist
        return self.dictionary_distance_actions[str(cond1)+str(cond2)]    

class Interval():
    '''
    This class represents a numeric range

    Args:
        a (float): minimum value
        b (float): maximum value

    Attributes:
        a (float): where minimum value is saved
        b (float): where maximum value is saved
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, interval):
        if not isinstance(interval,Interval):
            print("Abstractor: input is not Interval instance")
            return None
        
        if (self.a <= interval.a and interval.a <= self.b) or (self.a <= interval.b and interval.b <= self.b) or (interval.a <= self.a and self.a <= interval.b) or (interval.a <= self.b and self.b <= interval.b):
            return True
        return False

    def __ne__(self, interval): 
        return not interval == self 
  
    def __str__(self): 
        return "[{},{}]".format(self.a,self.b)  


