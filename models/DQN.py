import os
import pickle
import imageio
import numpy as np 
import tensorflow as tf
import tensorflow.keras as keras

from copy import deepcopy
from time import time
from .base import base
from os.path import join 
from core.tools import summary
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from core.replay_experience import Transition

from pdb import set_trace as debug



class DQNAgent(base):
    def __init__(
        self, env, policy, reward_class, model_name='temp', batch_size = 256, memory_size=1028, gamma = 0.95, epsilon = 1.0,
        epsilon_min = 0.01, epsilon_decay = 0.995, decay_timesteps=500, exploration_fraction=0.1, update_timesteps= 50, tau=0.01, 
        learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.99, logger_steps=500, learning_starts = 1000, double=True,
        action_replay=False, render=False, explainer_updates = 256, explainer_summarizing=25, val_eps = 10,
        val_numtimesteps = 1000, summarize_shap = True, num_to_explain = 5, making_a_gif=250, gif_length = 500,
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models', gifs = False, save_step = 1000):
        
        super(DQNAgent, self).__init__(
            env, learning_rate, beta_1, beta_2, tau, batch_size, gamma, memory_size,
            epsilon, epsilon_min, epsilon_decay, exploration_fraction, decay_timesteps, update_timesteps,
            logger_steps, learning_starts, action_replay, render, model_name, save_paths,
            val_eps, val_numtimesteps, gifs, making_a_gif, gif_length, save_step
            )
        

        self.double=double
        
        if(len(self.env.action_space.shape)>0):
            ## Behavior Model
            self.behavior = policy(self.env.observation_space.shape, self.env.action_space.shape).model
            self.behavior.compile(optimizer = self._build_opt(), loss = 'mse', metrics=['accuracy'])
            ## Target model
            self.target = policy(self.env.observation_space.shape, self.env.action_space.shape).model
            self.target.compile(optimizer=self._build_opt(), loss='mse', metrics=['accuracy'])
        else:
            ## Behavior Model
            self.behavior = policy(self.env.observation_space.shape, self.env.action_space.n).model
            self.behavior.compile(optimizer = self._build_opt(), loss = 'mse', metrics=['accuracy'])
            ## Target Model
            self.target = policy(self.env.observation_space.shape, self.env.action_space.n).model
            self.target.compile(optimizer = self._build_opt(), loss= 'mse', metrics=['accuracy'])
        self.transfer_weights()
        

        files = [f for f in os.listdir(self.save_path) if 'DQN' in f]

        self.save_path = join(self.save_path, 'DQN{}'.format(len(files) + 1))
        logdir = join(self.save_path, 'tensorboard_logs')
        self.writer = summary(tf.summary.FileWriter(logdir))
        self.state = {}
        if(self.gif):
            gifdir = join(self.save_path, 'gifs')
            os.makedirs(gifdir, exist_ok=True)
        
        # explainer parameters
        self.explainer = None
        self.explainer_updates = explainer_updates
        self.num_explainer_summaries = explainer_summarizing
        self.shap_summary = summarize_shap
        self.num_to_explain = num_to_explain


        # learn using the behavior not target
        self.reward_class = reward_class
        self.reward_function = None



    def _build_opt(self):
        return Adam(
            learning_rate = self.learning_rate,
            beta_1 = self.beta_1,
            beta_2 = self.beta_2,
            clipvalue=0.5,
        )
    
    def transfer_weights(self):
        # soft update
        self.target.set_weights(
            [self.tau*l1 + (1 - self.tau)*l2 for l1, l2 in zip(self.target.get_weights(), self.behavior.get_weights())]
        )

    def shap_predictor(self):
        return self.behavior


    def behavior_predict(self, state, single_obs=False):
        if(single_obs):
            state= np.expand_dims(state, axis=0)
        return self.behavior.predict(state)
    
    def target_predict(self, state, single_obs=False):
        if(single_obs):
            state = np.expand_dims(state, axis=0)
        return self.target.predict(state)

    

    def act(self, st):
        # pick an action
        if np.random.rand() <=self.epsilon:
            at = self.env.action_space.sample()
        else:
            at = np.argmax(self.behavior_predict(st, single_obs=True))
        

        if(self.action_replay): # repeat the same action 3 times
            for _ in range(3):
                st = self.act_once(at, st)
            return st
        else:
            return self.act_once(at, st)

    def update_on_batch(self):
        """
         A single update based on the 
        """
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        mask = np.ones(self.batch_size) * ([not l for l in batch.done])
        if(self.double):
            y = self.gamma * np.amax(self.target_predict(np.array(batch.next_state)), axis=1)
        else:
            y = self.gamma * np.amax(self.behavior_predict(np.array(batch.next_state)), axis=1)
        y *= mask

        tmp, self._parameter_dict = self.reward_function(batch, **self._parameter_dict)
        y += tmp 
        
        # N x num_actions
        target = self.behavior_predict(np.array(batch.state))
        target[np.arange(self.batch_size), batch.action] = y
        history = self.behavior.fit(np.array(batch.state), target, verbose=0)
        
        self.state['training/accuracy'] = history.history['acc'][0]
        self.state['training/loss'] = history.history['loss'][0]
        self.state['training/num_episodes'] = self._num_episodes
        
        
    

    def learn(self, total_timesteps=10000):
        self.reward_function = self.reward_class(self.shap_predictor()).reward_function
        st = self.env.reset()
        assert self.learning_starts < total_timesteps        
        best_val_score = -np.inf
        
        for tt in range(total_timesteps):
            start = time()
            self.state['timestep'] = tt
            self.update_dictionary()
            st = self.act(st)
            
            if(self.memory.can_sample(self.batch_size) and tt > self.learning_starts):
                self.update_on_batch()
                
                if (tt+1)%self.update_timesteps == 0:
                    self.transfer_weights()
                    
                if((tt+1)%self.logging_step == 0):    
                    val_avg_rew, val_avg_eps = self.validate()
                    if(val_avg_rew > best_val_score):
                        best_val_score = val_avg_rew
                        p = join(self.save_path, 'best_model')
                        os.makedirs(p, exist_ok = True)
                        self.save(p)
                    
                    self.state[f'validation/average_{self.num_validation_episode}_episode_reward'] = val_avg_rew
                    self.state['validation/average_episode_length'] = val_avg_eps
                    
                if (tt+1)%self.gif_logger_step == 0 and self.gif:
                    self.create_gif(frames=self.gif_frames, save = join(self.save_path, f'gifs/timesteps_{tt}'))
                                
                if (tt+1)%self.save_log == 0:
                    p = join(self.save_path, f'timesteps_{tt + 1}')
                    os.makedirs(p, exist_ok = True)
                    self.save(p)
            if(tt < self.exploration_fraction * total_timesteps):
                if self.epsilon > self.epsilon_min and tt%self.decay_timestep == 0:
                    self.epsilon *= self.epsilon_decay
            
            
            self.state['training/epsilon'] = self.epsilon
            self.state['training/time_per_iteration'] = time() - start
            self.writer.update(self.state)
        self.env.close()

    

    def create_gif(self, frames=500, fps=29, save='model'):
        """
         Makes a gif of an agent
        """
        images = []
        obs = self.env.reset()
        img = self.env.render(mode='rgb_array')

        eps_rew = 0
        episode = 1
        for _ in range(frames):
            images.append(img)
            action = np.argmax(self.target_predict(obs, single_obs=True))
            obs, rew, done, _ = self.env.step(action)
            eps_rew += rew
            img = self.env.render(mode='rgb_array')
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
        env = deepcopy(self.env)
        reward = 0
        episode_length = 0
        obs = env.reset()
        for _ in range(self.num_validation_episode):
            for j in range(self.num_validation_timesteps):
                a = np.argmax(self.target_predict(obs, single_obs=True))
                obs, r, done, _ = env.step(a)
                reward += r
                if(done):
                    obs = env.reset()
                    episode_length += (j + 1)
                    break
        
        return reward/self.num_validation_episode, episode_length/self.num_validation_episode
        

    def save(self, path):
        self.behavior.save(join(path, 'behavior.h5'))
        self.target.save(join(path, 'target.h5'))

    def load(self, path):
        self.behavior = load_model(join(path, 'behavior.h5'))
        self.target = load_model(join(path, 'target.h5'))
    

