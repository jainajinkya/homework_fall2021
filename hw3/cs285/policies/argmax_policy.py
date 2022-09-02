import numpy as np
from cs285.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)

        ## TODO return the action that maxinmizes the Q-value
        # at the current observation as the output
        _, idx = self.critic.q_net(observation).max(dim=1)
        action = ptu.to_numpy(idx)
        return action.squeeze()
