import os
import sys
import gym
import yaml
import pickle

from os.path import join
from argparse import ArgumentParser
from models.DQN import DQNAgent
from models.DDPG import DDPGAgent
from models.networks import MlpPolicy, CNNPolicy
from models.reward_functions import *

from pdb import set_trace as debug

def main(configs):
    
    
    env = gym.make(configs['env'])
    if(configs['policy'].casefold() == 'Mlp'.casefold()):
        policy = MlpPolicy
    elif(configs['policy'].casefold() == 'CNN'.casefold()):
        policy = CNNPolicy
    else:
        raise AttributeError("need to specify policy either ['Mlp', 'CNN'] ")

    if(configs['reward'].casefold() == 'identity_SHAP'.casefold()):
        reward = identity_SHAP
    elif(configs['reward'].casefold() == 'additive_SHAP'.casefold()):
        reward = additive_SHAP
    elif(configs['reward'].casefold() == 'identity'.casefold()):
        reward = Identity
    
    model_building_configs = configs['model_building_parameters']
    
    
    if(configs['model'].casefold() == "DQN".casefold()):
        model = DQNAgent(env, 
                    policy, 
                    reward,
                    **model_building_configs)
    elif(configs['model'].casefold() == 'DDPG'.casefold()):
        model = DDPGAgent(
                env, 
                reward,
                **model_building_configs)
    else:
        raise AttributeError("need to specify a model to train either ['DQN', 'DDPG']")


    if(configs['reload']):
        assert configs['saved_model'] is not None, "if reload is true, then saved_model must be set"
        path_to_saved_models = join('/'.join(model.save_path.split('/')[:-1]), configs['saved_model'])
        model.load(path_to_saved_models)
    
    
    model.learn(configs['timesteps'])
    with open(join(model.save_path, 'training_parameters.yaml'), 'w') as f:
        yaml.dump(configs, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--configs', required=True, help='configuration file')
    args = parser.parse_args()

    with open(args.configs, 'rb') as f:
        configs = yaml.load(f)

    main(configs)


 

