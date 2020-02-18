# replay memory class from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'done', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_sample(self,l):
        return len(self.memory) > l

    def __len__(self):
        return len(self.memory)