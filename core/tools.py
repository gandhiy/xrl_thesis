"""
Extra tools 
"""


import numpy as np
import pickle
import tensorflow as tf

def test_model(env, model, parameters):
    """
     Test a given environment with a trained (not necessarily) model under specific parameters

     env -> gym environment
     model -> trained stable_baselines model
     parameters -> parameters used to initialize training and training
    """
    
    testing_parameters = parameters['testing_parameters']
    
    # metrics
    reward_list = []
    episode_rewards = []
    episode_lengths = []
    num_episodes = 0
    eps_length = 0


    obs = env.reset()
    for _ in range(testing_parameters['iterations']):
        eps_length += 1
        action, _ = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        reward_list.append(rewards)
        
        if(dones):
            episode_rewards.append(rewards)
            episode_lengths.append(eps_length)
            num_episodes += 1
            eps_length = 0
            # obs = env.reset()

        if(testing_parameters['render']):
            env.render()

    results = {}
    results['run_name'] = parameters['run_name']
    results.update(parameters['learning_parameters'])
    results.update(parameters['model_parameters'])
    results.update(testing_parameters)
    results['reward_average'] = np.mean(reward_list)
    results['num_episodes'] = num_episodes

    if(len(episode_lengths) > 0):
        results['avg_eps_len'] = np.mean(episode_lengths)
    else: 
        results['avg_eps_len'] = eps_length
    
    if(len(episode_rewards) > 0):
        results['avg_eps_rew'] = np.mean(episode_rewards)
    else:
        results['avg_eps_rew'] = rewards
    
    
    return results


def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def tfSummary(tag, val):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
