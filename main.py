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
    global shap_vals
    if(_locals['self'].num_timesteps >= 1 and _locals['self'].num_timesteps%256 == 0): 
        
        callback_model = _locals['self']    

        def f(P):
            return np.array([callback_model.predict(x)[0] for x in P], dtype='float32')

        try:
            replay = callback_model.replay_buffer.sample(50)
            X,_ = replay[0], replay[1]
            X_train, X_test = X[:-1], X[-1]
        except AttributeError:
            seg = _locals['seg_gen'].__next__()
            observations = seg['observations']
            X_train, X_test = observations[:49], observations[-1]

        explainer = shap.KernelExplainer(f, X_train)
        shap_values = explainer.shap_values(X_test, nsample=10)
        shap_vals.append(shap_values)
    
    return True


def main(config):

    handler = Config_Handler(config)
    csv_list = []
    for i,d in  enumerate(handler.generate_dictionaries()):
        

        builder = model_builder(d, os.path.join(handler.log_dir, str(i)))
        model = builder.switcher[d['model']]()
        
        
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
        s = np.array(shap_vals)


        for i in range(s.shape[1]):
            if(len(s.shape) > 2):
                for j in range(s.shape[2]):
                    results['action_{}_obs_{}_mean'.format(i,j)] = np.mean(s[:,i,j])
                    results['action_{}_obs_{}_stddev'.format(i,j)] = np.std(s[:,i,j])
            else:
                results['action_0_obs_{}_mean'.format(i)] = np.mean(s[:,i])
                results['action_0_obs_{}_std'.format(i)] = np.std(s[:,i])
        csv_list.append(results)

    
    df = pd.DataFrame(csv_list)
    df.to_csv(os.path.join(handler.save_folder, 'results.csv'))
    
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--configuration_file', required=True, help='Path to configuration file.')
    args = parser.parse_args()
    shap_vals = []

    file = open("test_shap.txt", 'w')

    main(args.configuration_file)

    file.close()
