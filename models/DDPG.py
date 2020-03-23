import os
import pickle
import imageio
import numpy as np
import tensorflow as tf

from copy import deepcopy
from time import time
from .base import base
from os.path import join
from .networks import DDPGActor as Actor
from .networks import DDPGCritic as Critic
from core.tools import summary, Ornstein_Uhlenbeck_Noise
from tensorflow.keras.models import load_model
from core.replay_experience import Transition

from pdb import set_trace as debug

class DDPGAgent(base):
    def __init__(
        self, env, reward_class, model_name='temp', batch_size=256, memory_size=1028, gamma=0.95, epsilon = 1.0, 
        epsilon_min=0.01, epsilon_decay=0.995, exploration_fraction=0.1, decay_timesteps=100, update_timesteps=50, 
        tau=0.01, actor_lr = 0.001, critic_lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, clipping=0.5, logger_steps = 500, 
        learning_starts = 500, action_replay=False, render=False, explainer_updates=256, explainer_summarizing=25, sigma = 0.2, 
        mu = 0, theta=0.15, summarize_shap=True, num_to_explain=5, val_eps = 10, val_numtimesteps = 1000, making_a_gif=250, 
        gif_length = 500, save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models', gifs = False, save_step = 1000,
        actor_layers = [64,64], critic_layers=[64,64]
        ):

        super(DDPGAgent, self).__init__(
            env, 1.0, beta_1, beta_2, tau, batch_size, gamma, memory_size, 
            epsilon, epsilon_min, epsilon_decay, exploration_fraction, decay_timesteps, update_timesteps,
            logger_steps, learning_starts, action_replay, render, model_name, save_paths, val_eps, 
            val_numtimesteps, gifs, making_a_gif, gif_length, save_step

        )

        # inspired by stable-baselines code base
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        if(len(self.env.action_space.shape) > 0):
            envShape = self.env.action_space.shape 
        else:
            envShape = self.env.action_space.n





        ### Set up behavior Q ###
        self.behavior_q = Critic(self.env.observation_space.shape, envShape)
        self.behavior_q.init_model(critic_layers)
        self.behavior_q.build_opt(self.critic_lr, self.beta_1, self.beta_2)
        
        ### Set up target Q ###
        self.target_q = Critic(self.env.observation_space.shape, envShape)
        self.target_q.init_model(critic_layers)
        self.target_q.build_opt(self.critic_lr, self.beta_1, self.beta_2)


        ### Set up behavior pi ###
        self.behavior_pi = Actor(self.env.observation_space.shape, envShape, self.env.action_space.high)
        self.behavior_pi.init_model(actor_layers)
        self.behavior_pi_AdamOpt = self.behavior_pi.build_opt(self.actor_lr, self.beta_1, self.beta_2, clipping)
        self.get_critic_grad = self.behavior_pi.get_grads(self.behavior_q.model)

        ### Set up target pi ###
        self.target_pi = Actor(self.env.observation_space.shape, envShape, self.env.action_space.high)
        self.target_pi.init_model(actor_layers)
        _ = self.target_pi.build_opt(self.actor_lr, self.beta_1, self.beta_2, clipping)

        self.transfer_weights()
        



        # Set up tb logging
        files = [f for f in os.listdir(self.save_path) if 'DDPG' in f]
        self.save_path = join(self.save_path, 'DDPG{}'.format(len(files) + 1))
        logdir = join(self.save_path, 'tensorboard_logs')
        self.writer = summary(tf.summary.FileWriter(logdir))

        # explainer parameters
        self.explainer = None
        self.explainer_updates = explainer_updates
        self.num_explainer_summaries = explainer_summarizing
        self.shap_summary = summarize_shap
        self.num_to_explain = num_to_explain

        # reward funciton set
        self.reward_class = reward_class
        self.reward_function = None

        # OU Noise
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.exploration_noise = Ornstein_Uhlenbeck_Noise(envShape, self.mu, self.theta, self.sigma)

        

    def _action_replay(self, at, st):
        for _ in range(3):
            st = self.act_once(at, st)
        return st

    def act(self, st):
        

        at = self._action(st)
        at += self.exploration_noise.noise()
        
        if(self.action_replay): # repeat the same action 3 times
            return self._action_replay(at, st)
        else:
            return self.act_once(at, st)

    def _action(self, st):
        return self.behavior_predict(st, single_obs=True)[0]

    def shap_predictor(self):
        return self.behavior_pi.model

    def _asdf_test(self):
        x = np.random.rand(100, self.env.observation_space.shape[0])
        print(np.linalg.norm(self.target_pi.predict(x) - self.behavior_pi.predict(x)))

    def transfer_weights(self):
        self.target_q.transfer_weights(self.behavior_q.model, self.tau)
        self.target_pi.transfer_weights(self.behavior_pi.model, self.tau)    

    def behavior_predict(self, state, single_obs=False):
        return self.behavior_pi.predict(state)


    def target_predict(self, state, single_obs=False):
        return self.target_pi.predict(state)
        


    def update_on_batch(self):        
        """
         primed => target model
         non-primed => behavior model
         y_j = r_j = \gamma*Q'(s_{j+1}, \mu'(s_{j+1} | \theta^{\mu'}) | \theta^{Q'})
         L_Q = \frac{1}{N} \sum_j (y_j - Q(s_j, a_j | \theta^Q))^2 ==> squared bellman update
         \div_{\theta^{\mu}} J = \frac{1}{N} \sum_j \div_a Q(s_j, \mu(s_j | \theta^{\mu}) | \theta^Q)
        """
        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        
        states = np.array(batch.state)
        actions = np.array(batch.action)
        states_tp1 = np.array(batch.next_state)
        actions_tp1 = self.target_predict(states_tp1)
        
        state_avgs = np.mean(states, axis=0)
        for i in range(states.shape[1]):
            self.state[f'env_info/obs_{i}_average'] = state_avgs[i]

        state_std = np.std(states, axis=0)
        for j in range(states.shape[1]):
            self.state[f'env_info/obs_{j}_std'] = state_std[j]


            
        #### CRITIC UPDATE ####
        
        # bellman update
        mask = np.ones(self.batch_size) * ([not l for l in batch.done])
        mask = mask.reshape((-1, 1))

        y = self.target_q.model.predict([states_tp1, actions_tp1])
        y *= self.gamma
        y *= mask
        
        # apply shap updates here if desired
        r, self._parameter_dict = self.reward_function(batch, **self._parameter_dict)
        
        y += np.array(r).reshape((-1, 1))
        history = self.behavior_q.model.fit([states, actions], y, verbose=0)


        self.state['training/critic_accuracy'] = history.history['acc'][0]
        self.state['training/critic_loss'] = history.history['loss'][0]
        self.state['training/num_episodes'] = self._num_episodes

        #### ACTOR UPDATE ####
        acts = self.behavior_pi.predict(states)
        
        # gets the gradient of the critic output with respect to the input actions
        action_grads = self.get_critic_grad([states, acts])
        # apply gradients 
        grads = self.behavior_pi_AdamOpt([batch.state, np.array(action_grads).reshape(-1, self.env.action_space.shape[0])])
        
        self.state['training/actor_loss'] = grads[0]  
        
        



    def learn(self, total_timesteps):
        self.reward_function = self.reward_class(self.shap_predictor()).reward_function
        st = self.env.reset()
        assert self.learning_starts < total_timesteps

        
        self.__best_val_score = -10000000

        for tt in range(total_timesteps):
            start = time()
            self.update_dictionary()
            self.state['timestep'] = tt
            st = self.act(st) 
            
            
            if(self.memory.can_sample(self.batch_size) and tt > self.learning_starts):
                self.update_on_batch()   
                
                if (tt+1)%self.update_timesteps == 0:
                    self.transfer_weights()
                    
                if (tt+1)%self.logging_step == 0:
                    self.validate()

                if (tt+1)%self.gif_logger_step == 0 and self.gif:
                    self.create_gif(frames=self.gif_frames, save = join(self.save_path, f'gifs/timesteps_{tt}'))

                if (tt+1)%self.save_log == 0:
                    p = join(self.save_path, f'timesteps_{tt + 1}')
                    os.makedirs(p, exist_ok=True)
                    self.save(p)

            
            self.state['training/time_per_iteration'] = time() - start
            self.writer.update(self.state)
            self.state = {} # reset logging for each step (not everything needs to be plotted each timestep)
        
        self.env.close()
        


    def create_gif(self, frames=500, fps=29, save='model'):
        """
         Makes a gif of an agent
        """
        images = []
        obs = self.env.reset()
        try:
            img = self.env.render(mode='rgb_array')
        except NotImplementedError:
            img = self.env.env.render(mode='rgb_array')

        eps_rew = 0
        episode = 1
        for _ in range(frames):
            images.append(img)
            action = self.predict(obs)[0]
            obs, rew, done, _ = self.env.step(action)
            eps_rew += rew
            try:
                img = self.env.render(mode='rgb_array')
            except NotImplementedError:
                img = self.env.env.render(mode='rgb_array')
            if(done):
                print(f"Episode {episode} with reward {eps_rew}")
                episode += 1
                eps_rew = 0
                obs = self.env.reset()
        
        if('.gif' not in save):
            save += '.gif'
        imageio.mimsave(save, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
        self.env.close()



    def validate(self):
        env_target = deepcopy(self.env)
        env_behavior = deepcopy(self.env)
        env_dictionary = {
            'target': env_target,
            'behavior': env_behavior
        }

        for k,v in env_dictionary.items():
            env = v
            obs = env.reset()
            if(self.render and k == 'target'):
                try:
                    env.render(mode='rgb_array')
                except NotImplementedError:
                    env.env.render(mode='rgb_array')
            episode_rewards = []
            episode_lengths = []

            for _ in range(self.num_validation_episode):
                eps_count = 0
                reward = 0
                done = False
                while not done:
                    eps_count += 1
                    if(k == 'target'):
                        a = self.target_predict(obs)[0]
                    else:
                        a = self.behavior_predict(obs)[0]
                    
                    obs, r, done, _ = env.step(a)

                    if(self.render and k == 'target'):
                        try:
                            env.render(mode='rgb_array')
                        except NotImplementedError:
                            env.env.render(mode='rgb_array')

                    reward += r
                
                obs = env.reset()
                episode_rewards.append(reward)
                episode_lengths.append(eps_count)

            env.close()
            del(env)
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            
            if(avg_reward > self.__best_val_score and k == 'target'):
                self.__best_val_score = avg_reward
                p = join(self.save_path, 'best_model')
                os.makedirs(p, exist_ok=True)
                self.save(p)

            self.state[f'validation/average_{k}_reward'] = avg_reward
            self.state[f'validation/average_{k}_episode_length'] = avg_length
        self.state['validation/best_average_reward'] = self.__best_val_score


    """def validate__(self):
        env_target = deepcopy(self.env)
        env_behavior = deepcopy(self.env)
        reward_t = 0
        reward_b = 0
        episode_rewards_t = []
        episode_rewards_b = []
        episode_lengths_t = []
        episode_lengths_b = []

        obs_t = env_target.reset()
        obs_b = env_behavior.reset()
        
        #initialize rendering
        if(self.render):
            try:
                env_target.render(mode='rgb_array')
                
            except NotImplementedError:
                env_target.env.render(mode='rgb_array')
                

        # Run target validation
        for _ in range(self.num_validation_episode):
            for j in range(self.num_validation_timesteps):

                a_t = self.target_predict(obs_t)[0] # get action
                a_b = self.target_predict(obs_b)
                obs_t, r_t, done_t, _ = env_target.step(a_t)
                obs_b, r_b, done_b, _ = env_behavior.step(a_b)

                if(self.render):
                    try:
                        env_target.render(mode='rgb_array') 
                    except NotImplementedError:
                        env_target.env.render(mode='rgb_array')
                reward_t += r_t
                reward_b += r_b


                if(done_t):
                    episode_rewards.append(reward_t)
                    episode_lengths.append(j+1)

                    reward = 0
                    obs = env.reset()
                    
                    break
        env.close()
        del(env)

        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        if(avg_reward > self.__best_val_score):
            self.__best_val_score = avg_reward
            p = join(self.save_path, 'best_model')
            os.makedirs(p, exist_ok=True)
            self.save(p)
        
        self.state['validation/best_average_reward'] = self.__best_val_score
        self.state['validation/average_reward'] = avg_reward
        self.state['validation/average_episode_length'] = avg_length"""
        
        

    def save(self, path):
        self.behavior_pi.model.save(join(path, 'behavior_pi.h5'))
        self.behavior_q.model.save(join(path, 'behavior_q.h5'))
        self.target_pi.model.save(join(path, 'target_pi.h5'))
        self.target_q.model.save(join(path, 'target_q.h5'))


    def load(self, path):
        self.behavior_pi.model = load_model(join(path, 'behavior_pi.h5'))
        self.behavior_q.model = load_model(join(path, 'behavior_q.h5'))
        self.target_pi.model = load_model(join(path, 'target_pi.h5'))
        self.target_q.model = load_model(join(path, 'target_q.h5'))