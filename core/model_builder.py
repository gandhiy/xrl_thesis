"""
Take a set of parameters and build a model
"""
import gym
import numpy as np


from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv






class model_builder:

    def __init__(self, parameters, log_dir):


        self.model = None
        self.parameters = parameters
        self.model_parameters = parameters['model_parameters']
        self.switcher = {
            'A2C': self.build_A2C,
            'DQN': self.build_DQN,
            'TD3': self.build_TD3,
            'SAC': self.build_SAC,
            'TRPO': self.build_TRPO,
            'PPO1': self.build_PPO1,
            'DDPG': self.build_DDPG,
            'ACER': self.build_ACER
        }
        self.allowed_action_spaces = []

        self.env = gym.make(self.parameters['environment'])
        

        if(self.parameters['logging']): 
            self.env = Monitor(self.env, log_dir)
            self.tf_board = 'models/{}/tensorboard_logs/'.format(self.parameters['run_name'])
        else:
            self.tf_board = None

        self.env = DummyVecEnv([lambda: self.env])

        

    def build_DQN(self):
        from stable_baselines import DQN
        from stable_baselines.deepq.policies import MlpPolicy

        self.allowed_action_spaces = [gym.spaces.Discrete]


        if(type(self.env.action_space) in self.allowed_action_spaces):
            self.model = DQN(MlpPolicy, self.env, tensorboard_log=self.tf_board, **self.model_parameters)
        else:
            raise AttributeError(
                "DQN does not support given environment.\n" + 
                "The current environment action space is: " + 
                str(self.env.action_space))
        
        return self.model

    def build_A2C(self):
        from stable_baselines import A2C
        from stable_baselines.common.policies import MlpPolicy

        self.model = A2C(MlpPolicy, self.env, tensorboard_log=self.tf_board, **self.model_parameters)
        return self.model

    def build_ACER(self):
        from stable_baselines import ACER
        from stable_baselines.common.policies import MlpPolicy

        self.allowed_action_spaces = [gym.spaces.Discrete]

        if(type(self.env.action_space) in self.allowed_action_spaces):
            self.model = ACER(MlpPolicy, self.env, tensorboard_log=self.tf_board, **self.model_parameters)
        
        else:
            raise AttributeError(
                "ACER does not support given environment.\n" + 
                "The current environment action space is: " + 
                str(self.env.action_space))
        
        return self.model

    def build_DDPG(self):
        from stable_baselines import DDPG
        from stable_baselines.ddpg.policies import MlpPolicy
        from stable_baselines.common.noise import NormalActionNoise


        self.allowed_action_spaces = [gym.spaces.Box]

        if(type(self.env.action_space) in self.allowed_action_spaces):
            n_actions = self.env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
            self.model = DDPG(MlpPolicy, self.env, action_noise=action_noise, tensorboard_log=self.tf_board, **self.model_parameters)

        else:
            raise AttributeError(
                "DDPG does not support given environment.\n" + 
                "The current environment action space is: " + 
                str(self.env.action_space))


        return self.model

    def build_PPO1(self):
        from stable_baselines import PPO1
        from stable_baselines.common.policies import MlpPolicy

        # All action and observation spaces are allowed
        self.model = PPO1(MlpPolicy, self.env, tensorboard_log=self.tf_board, **self.model_parameters)
        return self.model

    def build_TRPO(self):
        from stable_baselines import TRPO
        from stable_baselines.common.policies import MlpPolicy

        # All action and observation spaces are allowed
        self.model = TRPO(MlpPolicy, self.env, tensorboard_log=self.tf_board, **self.model_parameters)
        return self.model

    def build_SAC(self):
        from stable_baselines import SAC
        from stable_baselines.sac.policies import MlpPolicy

        self.allowed_action_spaces = [gym.spaces.Box]

        if(type(self.env.action_space) in self.allowed_action_spaces):
            self.model = SAC(MlpPolicy, self.env, tensorboard_log=self.tf_board, **self.model_parameters)
        else:
            raise AttributeError(
                "TD3 does not support given environment.\n" + 
                "The current environment action space is: " + 
                str(self.env.action_space))

        return self.model

    def build_TD3(self):
        from stable_baselines import TD3
        from stable_baselines.td3.policies import MlpPolicy
        from stable_baselines.ddpg.noise import NormalActionNoise
        
        self.allowed_action_spaces = [gym.spaces.Box]

        if(type(self.env.action_space) in self.allowed_action_spaces):
            n_actions = self.env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
            self.model = TD3(MlpPolicy, self.env, action_noise=action_noise, tensorboard_log=self.tf_board, **self.model_parameters)

        else:
            raise AttributeError(
                "TD3 does not support given environment.\n" + 
                "The current environment action space is: " + 
                str(self.env.action_space))


        return self.model
