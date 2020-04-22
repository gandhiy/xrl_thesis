
import os
import numpy as np

from copy import deepcopy
from pickle import load, dump
from os.path import join
from core.replay_experience import ReplayMemory

from pdb import set_trace as debug

class base:

    def __init__(
        self, env, name, path, lr, b1, b2, ep, tau, bs, g, 
        ms, vlg, wu, ren, ve, gif,
        gls, gf, ss, vb
        ):
        self.model_name = name
        self.save_path = path
        self.__parameter_dict = None
        self.env = env
        self.v_env = deepcopy(env)
        self.learning_rate = lr
        self.beta_1 = b1
        self.beta_2 = b2
        self.epochs = ep
        self.tau = tau
        self.batch_size = bs
        self.gamma = g
        self.memory = ReplayMemory(capacity=ms)
        self.validation_logging = vlg
        self.warmup = wu
        self.render = ren
        self.save_gif = gif
        self.gif_logging = gls
        self.gif_frames = gf
        self.save_log = ss
        self.num_validation_episode = ve
        self.state = {}
        self.verbose = vb


        self._parameter_dict = {}
        self.save_path = join(self.save_path, self.model_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.generate_random_baseline()

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


        
    def generate_random_baseline(self):
        rewards = []
        per_step_rewards = []
        env = deepcopy(self.env)
        for _ in range(50):
            done = False
            obs = env.reset()
            r = 0
            while not done:
                a = env.action_space.sample()
                obs, reward, done, _ = env.step(a)
                per_step_rewards.append(reward)
                r += reward
            rewards.append(r)

        self._exp_episode_reward = np.mean(rewards)
        self._std_episode_reward = max(1e-5, np.std(rewards)) # too small of a stddev causes an error
        self._exp_step_reward = np.mean(per_step_rewards)
        self._std_step_reward = max(1e-5, np.std(per_step_rewards))
        print(" ------- Random Episode Stats ------- ")
        print(f"Avg_Eps: {self._exp_episode_reward} \t Std_Eps: {self._std_episode_reward}")
        print(f"Avg_Step: {self._exp_step_reward} \t Std_Step: {self._std_step_reward}")
