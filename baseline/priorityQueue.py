class PriorityQueue(object):
    def __init__(self):
        self.queue = []
        self.empty = True

    def is_empty(self):
        return self.empty

    def enqueue(self, data, value):
        if self.empty:
            self.queue = [(data,value)]
        else:
            for i in range(len(self.queue)):
                if self.queue[i][1] > value:
                    self.queue =  self.queue[:i] + [(data,value)] + self.queue[i:]
                    break
                if i == len(self.queue)-1:
                    self.queue = self.queue + [(data,value)]
        self.empty = False

 
    def dequeue(self):
        if self.empty:
            return None, None
     
        e = self.queue[0]
        self.queue = self.queue[1:]
        
        if not self.queue:
            self.empty = True
        return e

    def get_queue(self):
        return self.queue.copy()

    def get_queue_values(self):
        lst = []
        for i in range(len(self.queue)):
            lst += [self.queue[i][1]]
        return lst

    def get_queue_data(self):
        lst = []
        for i in range(len(self.queue)):
            lst += [self.queue[i][0]]
        return lst

    def replace_if_better(self, data, value):
        if self.empty:
            return False
        else:
            for i in range(len(self.queue)):
                if self.queue[i][0] == data and self.queue[i][1] > value:
                    self.queue =  self.queue[:i] + self.queue[i+1:]
                    self.enqueue(data, value)
                    return True
            return False


