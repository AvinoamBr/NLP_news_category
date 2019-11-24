import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as torch_data
from scipy import sparse
import scipy
import pandas as pd
from collections import deque


def sparse_to_matrix(A):
    if type(A) == scipy.sparse.csr.csr_matrix:
        return np.array(A.todense())
    return A


class CircularBufferloss(deque):
    def __init__(self, size=0, min_loss = 10**6):
        super(CircularBufferloss, self).__init__(maxlen=size)
        self.min_loss = min_loss
        self.last_lr_update_index = 0

    @property
    def step(self, val):
        self.append(val)
        self.min_loss = min(min(self),val)
        return sum(self)/len(self)
    @property
    def average(self):
        return sum(self)/len(self)

    @property
    def is_full(self):
        return len(self) == self.maxlen
#######################
class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data


##################
if __name__ == "__main__":
    cb = CircularBufferloss(size=4)
    for i in range(30):
        val = np.polyval([-0.05,10],i) + np.random.rand()

        if val > cb.min_loss:
            print ("val is gt cb.min_loos" , cb)
            # consider change learning rate...
        else:
            print("val is lt cb.min_loos", cb)

        cb.append(val)
        print (i, cb, cb)

