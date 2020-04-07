import os
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from copy import deepcopy
from .base import base
from os.path import join
from core.tools import summary, Ornstein_Uhlenbeck_Noise, Gaussian_Noise, Zero_Noise
from models.networks import DDPGCritic as Critic
from models.networks import DDPGActor as Actor
from core.replay_experience import Transition
from tensorflow.keras.models import load_model

from pdb import set_trace as debug



class DDPGAgent(base):
    def __init__(
        self, 
        env, 
        reward_class, 
        model_name='temp', 
        batch_size = 256, memory_size=50000,
        gamma = 0.99, tau = 0.001, noise_scale = 0.01,
        actor_lr = 0.01, critic_lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, critic_epochs=1,
        validation_logging = 25, warmup = 25, render = False, 
        sigma = 0.2, mu = 0, theta = 0.15, validation_episodes = 5,
        save_gifs = False, gif_logging_episodes=100, gif_frames=1000, 
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models',
        actor_layers = [400,300], critic_layers=[400,300], actor_reg = 0.01, critic_reg = 0.01,
        save_episode = 100, tb_log=True, verbose=0):

        super(DDPGAgent, self).__init__(
            env, model_name, save_paths, 1.0, beta_1, beta_2, critic_epochs, tau, batch_size, gamma, 
            memory_size, validation_logging, warmup, render, validation_episodes, save_gifs,
            gif_logging_episodes, gif_frames, save_episode, verbose)

        self.tb_log = tb_log

        if(len(self.env.action_space.shape) > 0):
            self.num_actions = self.env.action_space.shape[0]
        else:
            self.num_actions = self.env.action_space.n

        self.critic = Critic(self.env.observation_space.shape, self.num_actions, critic_layers, critic_reg)
        self.target_critic = Critic(self.env.observation_space.shape, self.num_actions, critic_layers, critic_reg)
        self.critic.initialize(critic_lr, beta_1, beta_2)
        self.get_grads = self.critic.get_grads()


        self.actor = Actor(self.env.observation_space.shape, self.num_actions, actor_layers, actor_reg, range_high=self.env.action_space.high, range_low=self.env.action_space.low)
        self.target_actor = Actor(self.env.observation_space.shape, self.num_actions, actor_layers, actor_reg, range_high=self.env.action_space.high, range_low=self.env.action_space.low)
        self.actor_optimizer = self.actor.initialize(actor_lr, beta_1, beta_2)
        
        
        self.transfer_weights()


        files = [f for f in os.listdir(self.save_path) if 'DDPG' in f]
        self.save_path = join(self.save_path, f'DDPG{len(files) + 1}')
        logdir = join(self.save_path, 'tensorboard_logs')
        if(self.tb_log):
            self.writer = summary(tf.summary.FileWriter(logdir))


        self.explainer = None

        self.reward_class = reward_class
        self.reward_function = None
        self.per_step_reward = []

        self.mu = mu 
        self.sigma = sigma
        self.theta = theta
        # self.exploration_noise = Ornstein_Uhlenbeck_Noise(self.num_actions, self.mu, self.theta, self.sigma)
        self.exploration_noise = Gaussian_Noise(self.num_actions, scale=noise_scale*self.env.action_space.high)
        self.__best_val_score = -1000000

        self.csv = []
        
        

    def transfer_weights(self):
        self.target_critic.transfer_weights(self.critic.model, self.tau)
        self.target_actor.transfer_weights(self.actor.model, self.tau)

    def _action(self, obs):
        return self.actor.predict(obs)

    
    def environment_step(self, obs, done, warming_up):
        self.environment_iteration += 1
        self.count += 1
        at = self._action(obs)
        if(warming_up):
            p = self.episode_number/self.total_episodes
            n = self.exploration_noise.noise()
            at = at*p + (1 - p)*n
            self.state['training/noise'] = ((1-p)*n, self.environment_iteration)
        else:
            self.state['training/noise'] = (0, self.environment_iteration)
        obs_t, rt, done, _ = self.env.step(at)
        self.state['training/per_step_reward'] = (rt, self.environment_iteration)
        self.per_step_reward.append(rt)
        trajectory = [obs, at, obs_t, done, rt]
        obs = obs_t
        if done:
            self.count = 0
            total_reward = np.sum(self.per_step_reward)
            self.state['training/episode_reward'] = ((total_reward - self._exp_episode_reward)/self._std_episode_reward, self.episode_number)
            self.per_step_reward = []
            obs = self.env.reset()
            trajectory[2] = obs
        self.memory.push(trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4])
        return obs, done

    def batch_update(self):
        batch = Transition(*zip(*self.memory.sample(self.batch_size)))

        states = np.array(batch.state)
        actions = np.array(batch.action)
        next_state = np.array(batch.next_state)


        ### Critic ###
        mask = np.ones(self.batch_size) * ([not l for l in batch.done])
        mask = mask.reshape((-1, 1))


            

        r, self._parameter_dict = self.reward_function(batch, **self._parameter_dict)
        y = self.target_critic.batch_predict(next_state, self.target_actor.batch_predict(next_state))
        y *= self.gamma
        y *= mask
        y += np.array(r).reshape((-1, 1))


        history = self.critic.fit(states, actions, y, self.epochs)
        self.state['training/critic_loss'] = (np.mean(history.history['loss']), self.training_iteration)
        self.state['training/critic_error'] = (np.mean(history.history['mean_absolute_error']), self.training_iteration)

        q_grads_wrt_a = self.get_grads([states, self.actor.batch_predict(states)]) # value of the target policy over the behavior policy
        mean_grads = self.actor_optimizer([states, np.array(q_grads_wrt_a).reshape(-1, self.num_actions)])
        self.state['training/actor_loss'] = (mean_grads[0], self.training_iteration)



    def learn(self, episodes=1000):
        self.reward_function = self.reward_class(self.actor.model).reward_function
        obs = self.env.reset()
        self.total_episodes = episodes + self.warmup
        self.episode_number = 0
        self.update_dictionary()
        self.training_iteration = 0
        self.environment_iteration = 0
        for e in tqdm(range(self.total_episodes)):
            self.state['episode'] = self.episode_number = e
            done = False
            self.count = 0
            while not done:
                warming_up = self.warmup > self.episode_number
                obs, done = self.environment_step(obs, done, warming_up)

                ### Algorithm Functions ###
                if(self.memory.can_sample(self.batch_size) and not warming_up):
                    self.training_iteration += 1
                    self.batch_update()
                    self.transfer_weights()

                    # do this at the begging of the episode
                    if(self.episode_number%self.validation_logging == 0 and self.count == 0):
                        self.validate()
                        
                    if(self.save_gif and self.episode_number%self.gif_logging == 0 and self.count == 0):
                        self.create_gif(frames = self.gif_frames, save=join(self.save_path, f'gifs/episode_{self.episode_number+1}'))

                    if(self.episode_number%self.save_log == 0 and self.count == 0):
                        p = join(self.save_path, f'episode_{self.episode_number+1}')
                        os.makedirs(p, exist_ok=True)
                        self.save(p)

                    

                self.update_dictionary()
                if self.tb_log:
                    self.writer.update(self.state)
                
                
        
    def validate(self):
        env = deepcopy(self.env)
        obs = env.reset()
        if self.render:
            try:
                env.render(mode='rbg_array')
            except NotImplementedError:
                env.env.render(mode='rgb_array')
        episode_rewards = []
        episode_lengths = []
        for v in range(self.num_validation_episode):            
            done = False
            count = 0
            reward = 0
            while not done:
                count += 1
                action = self.target_actor.predict(obs)
                obs, rt, done, _ = env.step(action)
                if(self.verbose > 1):
                    print(f"Validation Episode: {v} \t\t Step: {count} \t Action: {action}")
                if self.render:
                    try:
                        env.render(mode='rgb_array')
                    except NotImplementedError:
                        env.env.render(mode='rgb_array')
                reward += rt
            obs = env.reset()
            episode_rewards.append(reward)
            episode_lengths.append(count)
        env.close()
        del(env)

        avg_total_reward = np.mean(episode_rewards)
        episode_rewards = (episode_rewards - self._exp_episode_reward)/self._std_episode_reward
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)

        if(avg_reward > self.__best_val_score):
            self.__best_val_score = avg_reward
            p = join(self.save_path, 'best_model')
            os.makedirs(p, exist_ok=True)
            self.save(p)
        
        self.state[f'validation/average_total_reward'] = (avg_total_reward, self.episode_number)
        self.state[f'validation/average_relative_reward'] = (avg_reward, self.episode_number)
        self.state[f'validation/average_episode_length'] = (avg_length, self.episode_number)
        self.state[f'validation/best_average_reward'] = (self.__best_val_score, self.episode_number)





    def create_gif(self, frames=500, fps=60, save='model'):
        env = deepcopy(self.env)
        images = []
        obs = env.reset()
        
        try:
            img = env.render(mode='rgb_array')
        except NotImplementedError:
            img = env.env.render(mode='rgb_array')
        
        for _ in range(frames):
            images.append(img)
            action = self.target_actor.predict(obs)
            obs, _, done, _ = env.step(action)
            try:
                img = env.render(mode='rgb_array')
            except NotImplementedError:
                img = env.env.render(mode='rgb_array')

            if(done):
                obs = env.reset()
        
        if('.gif' not in save):
            save += '.gif'
        imageio.mimsave(save, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=fps)
        


    def save(self, path):
        self.actor.model.save(join(path, 'actor.h5'))
        self.target_actor.model.save(join(path, 'target_actor.h5'))
        self.critic.model.save(join(path, 'critic.h5'))
        self.target_critic.model.save(join(path, 'target_critic.h5'))


                
    def load(self, path):
        self.actor.model = load_model(join(path, 'actor.h5'))
        self.target_actor.model = load_model(join(path, 'target_actor.h5'))
        self.critic.model = load_model(join(path, 'critic.h5'))
        self.target_critic.model = load_model(join(path, 'target_critic.h5'))

            
            
            
            
            
