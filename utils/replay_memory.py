from collections import namedtuple
import random
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    
class MultiAgentMemory(object):
    def __init__(self):
        self.state = []
        self.action = []
        self.mask = []
        self.next_state = []
        self.reward = []
        
    def push(self, state, action, mask, next_state, reward):
        """Saves a transition."""
        self.state += state
        self.action += action
        self.mask += mask
        self.next_state += next_state
        self.reward += reward
    
    def convert_ndarray(self):
        self.state = np.array(self.state)
        self.action = np.array(self.action)
        self.mask = np.array(self.mask)
        self.next_state = np.array(self.next_state)
        self.reward = np.array(self.reward)
        
    def __len__(self):
        return len(self.mask)