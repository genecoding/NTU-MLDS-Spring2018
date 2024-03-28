import random
import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # turn shape from (84, 84, 4) to (1, 4, 84, 84)
        state = np.transpose(state, (2, 0, 1))
        state = np.expand_dims(state, 0)
        next_state = np.transpose(next_state, (2, 0, 1))
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.capacity