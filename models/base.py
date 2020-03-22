
import os
import numpy as np

from pickle import load, dump
from os.path import join
from core.replay_experience import ReplayMemory

from pdb import set_trace as debug

class base:

    def __init__(
        self, env, lr, b1, b2, tau, bs, g, ms, e, em, 
        ed, ef, dt, up_t, lg_t, ls, ar, ren, name, path,
        ve, vt, make_gif, gif, gl, ss
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
        self.decay_timestep = dt
        self.update_timesteps = up_t
        self.logging_step = lg_t
        self.learning_starts = ls
        self.action_replay = ar
        self.render = ren
        self.gif = make_gif
        self.gif_logger_step = gif
        self.gif_frames = gl
        self.save_log = ss

        self.state = {}



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

        self.exploration_noise = None

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
        self._eps_rew_list.append(rt)
        if done:
            self._num_episodes += 1
            self.state['training/reward'] = sum(self._eps_rew_list)/len(self._eps_rew_list)
            self._eps_rew_list = []
            snext = self.env.reset()
            if(self.exploration_noise is not None):
                self.exploration_noise.reset()
        
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
        
