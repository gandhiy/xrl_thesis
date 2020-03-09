import os
import sys
import gym
import pickle

from models.DQN import DQNAgent
from models.networks import MlpPolicy
from models.reward_functions import Identity, additive_SHAP


env = gym.make("CartPole-v0")
model = DQNAgent(env, 
                 MlpPolicy, 
                 Identity, 
                 model_name='test',
                 memory_size = 50000, 
                 batch_size=256, 
                 update_timesteps=256, 
                 logger_steps=100, 
                 learning_starts=0)


model.learn(500)

 

