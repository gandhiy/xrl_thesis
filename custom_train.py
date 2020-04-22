import os
import sys
import gym
import yaml
import pickle

from os.path import join
from argparse import ArgumentParser
from models.DQN import DQNAgent
from models.DDPG import DDPGAgent
from models.PPO import PPOAgent
from models.reward_functions import *

from pdb import set_trace as debug

def main(configs):
    
    
    env = gym.make(configs['env'])
    if('curriculum' in configs['env'].casefold()):
        env.init(configs['episodes'])

    
    if(configs['reward'].casefold() == 'identity'):
        reward = Identity
    elif(configs['reward'].casefold() == 'dqn_shap'):
        reward = dqn_shap
    elif(configs['reward'].casefold() == 'ddpg_shap'):
        reward = ddpg_shap
    elif(configs['reward'].casefold() == 'mountaincar_curriculum'):
        reward = mountaincar_curriculum
    elif(configs['reward'].casefold() == 'dqn_shap_curriculum'):
        reward = dqn_shap_curriculum
    elif(configs['reward'].casefold() == 'ppo_shap'):
        reward = ppo_shap
    else:
        raise AttributeError("need to specify the reward function")

    model_building_configs = configs['model_building_parameters']
    if(configs['model'].casefold() == "DQN".casefold()):
        model = DQNAgent(
                env, 
                reward,
                **model_building_configs)
    elif(configs['model'].casefold() == 'DDPG'.casefold()):
        model = DDPGAgent(
                env, 
                reward,
                **model_building_configs)
    elif(configs['model'].casefold() == 'PPO'.casefold()):
        model = PPOAgent(
                env,
                reward,
                **model_building_configs)
    else:
        raise AttributeError("need to specify a model to train either ['DQN', 'DDPG']")

    

    if(configs['reload']):
        assert configs['saved_model'] is not None, "if reload is true, then saved_model must be set"
        path_to_saved_models = join('/'.join(model.save_path.split('/')[:-1]), configs['saved_model'])
        model.load(path_to_saved_models)
    
    

    model.learn(configs['episodes'])
    with open(join(model.save_path, 'training_parameters.yaml'), 'w') as f:
        yaml.dump(configs, f)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--configs', required=True, help='configuration file')
    args = parser.parse_args()

    with open(args.configs, 'rb') as f:
        configs = yaml.load(f)

    main(configs)


 

