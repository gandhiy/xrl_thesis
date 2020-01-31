"""
Main python file to run training.
"""


import os
import yaml
import shap
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from core.tools import test_model
from core.config_handler import Config_Handler
from core.model_builder import model_builder


from pdb import set_trace as debug


def callback(_locals, _globals):
    global shap_values, file
    if(_locals['self'].num_timesteps >= 1): # and _locals['self'].num_timesteps%100 == 0):
        callback_model = _locals['self']
        def f(P):
            return np.array([callback_model.predict(x)[0] for x in P], dtype='float32')
        replay = callback_model.replay_buffer.sample(50) # will not work for all of the model types
        X,y = replay[0], replay[1]
        X_train, X_test = X[:-1], X[-1]
        explainer = shap.KernelExplainer(f, X_train)
        shap_values.append(explainer.shap_values(X_test, nsample=100))
        file.write(str(shap_values))



def main(config):

    handler = Config_Handler(config)
    csv_list = []
    for i,d in  enumerate(handler.generate_dictionaries()):
        

        builder = model_builder(d, os.path.join(handler.log_dir, str(i)))
        model = builder.switcher[d['model']]()
        
        
        try:
            model.learn(
                total_timesteps=d['learning_parameters']['timesteps'],
                log_interval=d['learning_parameters']['log_interval'], 
                callback=callback)
                
        except KeyboardInterrupt:
            print("Incomplete Model Save")
            model.save(os.path.join(handler.save_folder, 'incomplete_run_{}'.format(i)))
    
        if(d['save_model']):
            model.save(os.path.join(handler.save_folder, 'run_{}'.format(i)))
        

        results = test_model(builder.env, model, d)
        csv_list.append(results)
    
    df = pd.DataFrame(csv_list)
    df.to_csv(os.path.join(handler.save_folder, 'results.csv'))
        
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--configuration_file', required=True, help='Path to configuration file.')
    args = parser.parse_args()
    shap_values = []
    file = open("test_shap.txt", 'w')

    main(args.configuration_file)

    file.close()
