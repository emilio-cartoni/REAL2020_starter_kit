import numpy as np


class PriorityQueue(object):
    def __init__(self):
        self.queue = []
        self.empty = True

    def is_empty(self):
        return self.empty

    def enqueue(self, data, value):
        if self.empty:
            self.queue = [(data, value)]
        else:
            i = self.binary_search(value)
            self.queue = self.queue[:i] + [(data, value)] + self.queue[i:]

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
        return np.take(self.queue, 1, axis=1)

    def get_queue_data(self):
        return np.take(self.queue, 0, axis=1)

    def replace_if_better(self, data, value):
        if self.empty:
            return False
        else:
            for i in range(len(self.queue)):
                if self.queue[i][0] == data and self.queue[i][1] > value:
                    self.queue = self.queue[:i] + [(data, value)] + self.queue[i + 1:]
                    return True
            return False

    def binary_search(self, value):
        if self.queue[-1][1] <= value:
            return len(self.queue)
        if self.queue[0][1] >= value:
            return 0
        i = 0
        j = len(self.queue) - 1
        while i != j - 1:
            idx = i + int(np.floor((j - i) / 2))
            if self.queue[idx][1] == value:
                return idx
            elif self.queue[idx][1] < value:
                i = idx
            else:
                j = idx
        return j
