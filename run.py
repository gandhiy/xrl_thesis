import gym
import tensorflow as tf 

from models.DQN import DQNAgent
from models.DDPG import DDPGAgent
from models.reward_functions import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *




env = gym.make("LunarLander-v2")

agent1 = DQNAgent(
env, 
Identity, 
RMSprop(1e-4, decay=0.9), 
model_name='lunarlander_testing',
batch_size=256,
warmup=0,
validation_logging=5,
validation_episodes=5,
layers=[128,128],
activation=ELU,
epsilon_min=0.01,
epsilon_decay=0.999,
tau=0.01,
reg=0.01,
gamma=0.999
)

env_cont = gym.make("LunarLanderContinuous-v2")

agent2 = DDPGAgent(
    env_cont, 
    Identity,
    critic_optimizer=RMSprop(1e-4, decay=0.9),
    actor_optimizer=RMSprop(1e-5, decay=0.75),
    model_name='lunarlander_testing',
    batch_size=64,
    warmup=150,
    actor_layers=[128, 256],
    critic_layers=[128, 256],
    actor_reg=0.02,
    critic_reg=0.02,
    activation='relu',
    epsilon_min=0,
    tau=0.01,
    gamma=0.999,
    validation_logging=10,
    validation_episodes=5,
    epsilon_episodes=100
)


# agent1.learn(1000)
agent2.learn(1000)