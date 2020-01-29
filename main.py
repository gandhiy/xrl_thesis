"""
Main python file to run training.
"""


import os
import yaml
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from core.tools import test_model
from core.config_handler import Config_Handler
from core.model_builder import model_builder



def callback():
    pass


def main(config):

    handler = Config_Handler(config)
    csv_list = []
    for i,d in  enumerate(handler.generate_dictionaries()):
        

        builder = model_builder(d, os.path.join(handler.log_dir, str(i)))
        model = builder.switcher[d['model']]()
        
        
        try:
            model.learn(
                total_timesteps=d['learning_parameters']['timesteps'],
                log_interval=d['learning_parameters']['log_interval'])
        except KeyboardInterrupt:
            print("Incomplete Model Save")
            model.save(os.path.join(handler.save_folder, 'incomplete_run_{}'.format(i)))
    
    
        model.save(os.path.join(handler.save_folder, 'run_{}'.format(i)))

        results = test_model(builder.env, model, d)
        csv_list.append(results)
    
    df = pd.DataFrame(csv_list)
    df.to_csv(os.path.join(handler.save_folder, 'results.csv'))
        
    

        

        
        



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--configuration_file', required=True, help='Path to configuration file.')
    args = parser.parse_args()

    main(args.configuration_file)


