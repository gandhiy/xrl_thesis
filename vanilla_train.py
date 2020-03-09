"""
Main python file to run training.
"""


import os
import gym
import yaml
import shap
import numpy as np
import pandas as pd
import stable_baselines

from argparse import ArgumentParser
from core.tools import test_model
from core.config_handler import Config_Handler
from core.model_builder import model_builder
from core.customcallbacks import CustomCallback
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from pdb import set_trace as debug
def generate_callback(d, handler):
    checkpoint_callback = CheckpointCallback(
        save_freq = d['callback_parameters']['save_freq'], 
        save_path = os.path.join(handler.save_folder, 'checkpoints'))
    eval_callback  = EvalCallback(
        gym.make(d['environment']),
        best_model_save_path = handler.log_dir,
        log_path = handler.log_dir,
        eval_freq = d['callback_parameters']['eval_freq']
    )
    custom_callback = CustomCallback(d['callback_parameters'], None)
    return CallbackList([checkpoint_callback, eval_callback, custom_callback])


def main(config):

    handler = Config_Handler(config)
    csv_list = []
    
    # stop entire training
    try:
        for i,d in  enumerate(handler.generate_dictionaries()):
            
            callback = generate_callback(d, handler)
            builder = model_builder(d, os.path.join(handler.log_dir, str(i)))
            model = builder.switcher[d['model']]()
            
            # stop the model learning
            try:
                model.learn(
                    total_timesteps=d['learning_parameters']['timesteps'],
                    callback=callback)
                    
            except KeyboardInterrupt:
                print("Incomplete Model Save")
                model.save(os.path.join(handler.save_folder, 'incomplete_run_{}'.format(i)))
        
            if(d['save_model']):
                model.save(os.path.join(handler.save_folder, 'run_{}'.format(i)))
            
            results = test_model(builder.env, model, d)
            results['environment'] = d['environment']

    
    except KeyboardInterrupt:
        df = pd.DataFrame(csv_list)
        df.to_csv(os.path.join(handler.save_folder, 'incomplete_results.csv'))
    
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--configuration_file', required=True, help='Path to configuration file.')
    args = parser.parse_args()
    
    main(args.configuration_file)
