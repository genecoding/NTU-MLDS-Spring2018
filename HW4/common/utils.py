import gym
import numpy as np
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0., std=0.01)
        

class PyTorchEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # modify the definition of observation space
        # self.observation_space = gym.spaces.Box(
        #     low=0.0,
        #     high=1.0,
        #     shape=(
        #         env.observation_space.shape[2],
        #         env.observation_space.shape[0],
        #         env.observation_space.shape[1],
        #     ),
        #     dtype=np.float32,
        # )

    def observation(self, observation):
        # turn shape from (84, 84, 4) to (4, 84, 84)
        # return np.transpose(observation, (2, 0, 1))
        
        return np.array(observation)