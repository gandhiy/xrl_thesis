## https://github.com/LuEE-C/PPO-Keras/blob/8c61f59339b7dae2585357ba6427037f1ceca84a/Main.py#L74
import os
import gym
import imageio
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from copy import deepcopy
from .base import base
from os.path import join
from core.tools import summary
from models.networks import PPOActor, PPOCritic
from tensorflow.keras.models import load_model

from pdb import set_trace as debug

class PPOAgent(base):
    def __init__(
        self, env, reward_class, model_name='temp',clip_value=0.2, noise=1.0,
        batch_size=256, buffer_size=5000, gamma = 0.99, entropy_weight=5e-3,
        critic_lr = 0.001, actor_lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, 
        validation_logging = 25, warmup = 10, render = False, validation_episodes=5,
        save_gifs = False, gif_logging_episodes=100, gif_frames=1000, 
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models/',
        critic_batch_size = 32, actor_batch_size=32, critic_epochs = 1, actor_epochs=1,
        actor_layers = [64,64], critic_layers = [64,64], save_episodes = 100,
        tb_log = True, verbose=0, explainer_samples= -1
    ):
        super(PPOAgent, self).__init__(
            env, model_name, save_paths, 1.0, beta_1, beta_2, 1, 1.0, 
            batch_size, gamma, buffer_size, validation_logging, warmup,
            render, validation_episodes, save_gifs, gif_logging_episodes, 
            gif_frames, save_episodes, verbose)
        
        self.buffer_size = buffer_size
        self.tb_log = tb_log
        self.continuous = isinstance(self.env.action_space, (gym.spaces.box.Box, ))
        if(len(self.env.action_space.shape) > 0):
            self.num_actions = self.env.action_space.shape[0]
        else:
            self.num_actions = self.env.action_space.n


        self.critic = PPOCritic(self.env.observation_space.shape, critic_layers)
        self.critic.initialize(critic_lr, beta_1, beta_2, clip_value)
        self.critic_batch_size = critic_batch_size
        self.critic_epochs = critic_epochs

        
        self.actor = PPOActor(self.env.observation_space.shape, self.num_actions, actor_layers, self.continuous)
        self.actor.initialize(actor_lr, beta_1, beta_2, clip_value, entropy_weight, noise)
        self.actor_batch_size = actor_batch_size
        self.actor_epochs = actor_epochs


        files = [f for f in os.listdir(self.save_path) if 'PPO' in f]
        self.save_path = join(self.save_path, f'PPO{len(files) + 1}')
        logdir = join(self.save_path, 'tensorboard_logs')
        if(self.tb_log):
            self.writer = summary(tf.summary.FileWriter(logdir))
        
        self.explainer = None
        self.reward_class = reward_class
        self.reward_function = None
        if(explainer_samples < 0):
            self.samples = self.batch_size
        else:
            self.samples = explainer_samples


        self.per_step_reward = []
        self.__best_val_score = -1000000

    def generate_vf(self):
        self.state['training/episode_reward'] = (np.sum(self.reward), self.episode_number)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.gamma


    def batch_update(self):
        rewards, self._parameter_dict = self.reward_function(self.memory, **self._parameter_dict)
        
        obs = np.array(self.memory[0])
        action = np.array(self.memory[1])
        prediction = np.array(self.memory[2])
        prediction = np.squeeze(prediction)
        reward = np.array(rewards).reshape((len(rewards), 1))

    
        old_prediction = prediction

        V = self.critic.predict(obs)

        advantage = reward - V
        actor_history = self.actor.fit(obs, advantage, old_prediction, action, self.actor_batch_size, self.actor_epochs)
        critic_history = self.critic.fit(obs, reward, self.critic_batch_size, self.critic_epochs)

        a_loss = np.mean(actor_history.history['loss'])
        c_loss = np.mean(critic_history.history['loss'])
        self.state['training/actor_loss'] = (a_loss, self.training_iteration)
        self.state['training/critic_loss'] = (c_loss, self.training_iteration)
        self.memory = [[], [], [], []]


    def learn(self, episodes=1000):
        self.reward_function = self.reward_class(self.actor.model).reward_function
        self.total_episodes = episodes + self.warmup
        self.episode_number = 0
        self.update_dictionary()
        self.training_iteration = 0
        self.environment_iteration = 0
        

        obs = self.env.reset()
        self.reward = []
        self.memory = [[], [], [], []]

        for e in tqdm(range(self.total_episodes)):
            self.state['episode'] = self.episode_number = e
            done = False
            self.count = 0
            tmp_batch = [[], [], []]
            self.reward = []
            while not done:
                warming_up = self.warmup > self.episode_number
                action, action_matrix, predicted_action = self.actor.get_action(obs)
                obs_t, rt, done, _ = self.env.step(action)
                self.per_step_reward.append(rt)
                self.reward.append(rt)
                tmp_batch[0].append(obs)
                tmp_batch[1].append(action_matrix)
                tmp_batch[2].append(predicted_action)
                obs = obs_t

                if done:
                    self.generate_vf()
                    self.memory[0].extend(tmp_batch[0])
                    self.memory[1].extend(tmp_batch[1])
                    self.memory[2].extend(tmp_batch[2])
                    self.memory[3].extend(self.reward)
                    self.per_step_reward = []
                    obs = self.env.reset()
                    

                if(len(self.memory[0]) > self.buffer_size and not warming_up):
                    self.batch_update()
                    self.training_iteration += 1

                self.update_dictionary()
                if self.tb_log:
                    self.writer.update(self.state)

            if(self.episode_number%self.validation_logging == 0 and not warming_up):
                self.validate()

            if(self.save_gif and self.episode_number%self.gif_logging == 0 and not warming_up):
                self.create_gif(frames = self.gif_frames, save=join(self.save_path, f'gifs/episode_{self.episode_number+1}'))
                
            if(self.episode_number%self.save_log == 0 and not warming_up):
                p = join(self.save_path, f'episode_{self.episode_number+1}')
                os.makedirs(p, exist_ok=True)
                self.save(p)


                

    def validate(self):
        env = deepcopy(self.env)
        obs = env.reset()
        if self.render:
            try:
                env.render(mode='rgb_array')
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
                action = self.actor.predict(obs)
                obs, rt, done, _ = env.step(action)
                if(self.verbose > 1):
                    print(f"Validation Episode: {v} \t \t Step: {count} \t {action}")
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
            img = env.env.render(mode = 'rgb_array')
        
        for _ in range(frames):
            images.append(img)
            action = self.actor.predict(obs)
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
        self.actor.model.save_weights(join(path, 'actor.h5'))
        self.critic.model.save_weights(join(path, 'critic.h5'))

    def load(self, path):
        self.actor.model.load_weights(join(path, 'actor.h5'))
        self.critic.model.load_weights(join(path, 'critic.h5'))

