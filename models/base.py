
import os
import numpy as np

from pickle import load, dump
from os.path import join
from core.replay_experience import ReplayMemory

from pdb import set_trace as debug

class base:

    def __init__(
        self, env, lr, b1, b2, tau, bs, g, ms, e, em, 
        ed, ef, up_t, lg_t, ls, ar, ren, name, path,
        ve, vt
        ):
        self.model_name = name
        self.save_path = path
        self.__parameter_dict = None
        self.env = env
        self.learning_rate = lr
        self.beta_1 = b1
        self.beta_2 = b2
        self.tau = tau
        self.batch_size = bs
        self.gamma = g
        self.memory = ReplayMemory(capacity=ms)
        self.epsilon = e
        self.epsilon_min = em
        self.epsilon_decay = ed
        self.exploration_fraction = ef
        self.update_timesteps = up_t
        self.logging_step = lg_t
        self.learning_starts = ls
        self.action_replay = ar
        self.render = ren

        # validation parameters
        self.num_validation_episode = ve
        self.num_validation_timesteps = vt

        self._num_episodes = 0
        self._eps_rew = 0
        self._eps_rew_list = []
        self._mean_eps_rew = 0
        self._current_timesteps = 0

        self._parameter_dict = {}
        self.save_path = join(self.save_path, self.model_name)
        os.makedirs(self.save_path, exist_ok=True)

    def update_dictionary(self):
        if(len(self._parameter_dict)  == 0):
            self._parameter_dict = {
                key:value for key, value in self.__dict__.items() if 
                not key.startswith('__') and not callable(key) and key is not 'shap_predict'
            }
        else:
            for k,v in self.__dict__.items():
                if not k.startswith("__") and not callable(k) and k is not 'shap_predict':
                    self._parameter_dict[k] = v

    def act_once(self, at, st):
        snext, rt, done, _ = self.env.step(at)
        if done:
            self._num_episodes += 1
            self._eps_rew_list.append(self._eps_rew)
            self._mean_eps_rew = sum(self._eps_rew_list)/len(self._eps_rew_list)
            self._eps_rew = 0
            snext = self.env.reset()
        self._eps_rew += rt
        self.memory.push(st, at, snext, done, rt)
        return snext

    def target_predict(self, state, single_obs=False):
        raise NotImplementedError

    def predict(self, state):
        if(len(self.env.observation_space.shape) <= 2):
            #tabular environment
            if(len(self.env.observation_space.shape) < 1):
                # single observation
                return self.target_predict(state,single_obs=True)
            else:
                return self.target_predict(state,single_obs=False)
        else:
            if(len(self.env.observation_space.shape) == 3):
                # width, height, channel
                return self.target_predict(state,single_obs=True)
            else:
                return self.target_predict(state, single_obs=False)
        
