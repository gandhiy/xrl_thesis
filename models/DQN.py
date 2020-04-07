import os 
import imageio
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from copy import deepcopy
from .base import base
from os.path import join
from core.tools import summary
from models.networks import DQNPolicy
from core.replay_experience import Transition
from tensorflow.keras.models import load_model

from pdb import set_trace as debug

class DQNAgent(base):
    def __init__(
        self, env, reward_class, model_name='temp', 
        batch_size=256, memory_size=50000, gamma=0.995, tau = 0.001,
        start_epsilon=1.0, epsilon_min = 0.001, epsilon_decay = 0.995, 
        warmup=25, learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.99, epochs=1,
        render = False, validation_logging = 25, validation_episodes = 5, 
        save_gifs = False, save_gifs_every_n_episodes = 100, gif_frames=1000,
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models', save_episodes = 
        100, layers = [64,64], verbose=0, tb_log = True):

        super(DQNAgent, self).__init__(
        env, model_name, save_paths, learning_rate, beta_1, beta_2, epochs, tau, batch_size,
        gamma, memory_size, validation_logging, warmup, render, validation_episodes, 
        save_gifs, save_gifs_every_n_episodes, gif_frames, save_episodes, verbose)
        

        self.tb_log = tb_log

        if(len(self.env.action_space.shape) > 0):
            self.num_actions = self.env.action_space.shape[0]
        else:
            self.num_actions = self.env.action_space.n

        self.critic = DQNPolicy(self.env.observation_space.shape, self.num_actions, layers = layers)
        self.critic.initialize(self.learning_rate, self.beta_1, self.beta_2)

        self.target = DQNPolicy(self.env.observation_space.shape, self.num_actions, layers = layers)
        self.target.transfer_weights(self.critic, self.tau)

        files = [f for f in os.listdir(self.save_path) if 'DQN' in f]
        self.save_path = join(self.save_path, f'DQN{len(files) + 1}')
        logdir = join(self.save_path, 'tensorboard_logs')
        if(self.tb_log):
            self.writer = summary(tf.summary.FileWriter(logdir))

        self.explainer = None
        self.reward_class = reward_class
        self.reward_function = None
        self.per_step_reward = []

        self.epsilon = start_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_factor = epsilon_decay
        self.__best_val_score = -1000000


    def _action(self, obs):
        if(np.random.rand() > self.epsilon):
            return self.critic.predict(obs)
        else:
            return self.env.action_space.sample()

    def environment_step(self, obs, done):
        self.environment_iteration += 1
        self.count += 1
        at = self._action(obs)
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
        mask = np.ones(self.batch_size) * ([not l for l in batch.done])
        
        r, self._parameter_dict = self.reward_function(batch, **self._parameter_dict)
        y = self.gamma * np.amax(self.critic.model.predict(np.array(batch.next_state)), axis=1)
        y *= mask
        y += r

        target = self.target.model.predict(np.array(batch.state))
        target[np.arange(self.batch_size), batch.action] = y
        history = self.critic.fit(np.array(batch.state), target)

        self.state['training/accuracy'] = (history.history['acc'][0], self.training_iteration)
        self.state['training/loss'] = (history.history['loss'][0], self.training_iteration)
        

    def learn(self, episodes=1000):
        self.reward_function = self.reward_class(self.critic.model).reward_function
        self.total_episodes = episodes + self.warmup
        self.update_dictionary()
        self.episode_number = 0
        self.training_iteration = 0
        self.environment_iteration = 0

        obs = self.env.reset()
        for e in tqdm(range(self.total_episodes)):
            self.state['episode'] = self.episode_number = e
            done = False
            self.count = 0
            while not done:
                warming_up = self.warmup > self.episode_number
                obs, done = self.environment_step(obs, done)

                if(self.memory.can_sample(self.batch_size) and not warming_up):
                    
                    
                    for _ in range(self.epochs):   
                        self.training_iteration += 1                     
                        self.batch_update()
                    self.target.transfer_weights(self.critic, self.tau)

                    if(self.episode_number%self.validation_logging == 0 and self.count == 1):
                        self.validate()

                    if(self.save_gif and self.episode_number%self.gif_logging == 0):
                        self.create_gif(frames = self.gif_frames, save=join(self.save_path, f'gifs/episode_{self.episode_number + 1}'))

                    if(self.episode_number%self.save_log == 0 and self.count == 0):
                        p = join(self.save_path, f'episode_{self.episode_number+1}')
                        os.makedirs(p, exist_ok=True)
                        self.save(p)
                    
                if(self.epsilon > self.epsilon_min and warming_up):
                    self.epsilon *= self.epsilon_decay_factor
                else:
                    self.epsilon = self.epsilon_min
                self.state['training/epsilon'] = (self.epsilon, self.environment_iteration)

                self.update_dictionary()
                if self.tb_log:
                    self.writer.update(self.state)


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
                action = self.target.predict(obs)
                obs, rt, done, _  = env.step(action)
                if(self.verbose > 1):
                    print(f"Validation Episode: {v} \t\t Step: {count} \t Action: {action}")
                if self.render:
                    try:
                        env.render(mode='rgb_array')
                    except:
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

    def create_gif(self, frames = 500, fps = 60, save='model'):
        env = deepcopy(self.env)
        images = []
        obs = env.render()

        try:
            img = env.render(mode='rgb_array')
        except NotImplementedError:
            img = env.env.render(mode='rgb_array')
        
        for _ in range(frames):
            images.append(img)
            action = self.target.predict(obs)
            obs, reward, done, _ = env.step(action)
            try:
                img = env.render(mode='rgb_array')
            except NotImplementedError:
                img = env.env.render(mode='rgb_array')
            
            if done:
                obs = env.reset()
        
        if('.gif' not in save):
            save += '.gif'

        imageio.mimsave(save, [np.array(img) for i,img in enumerate(images) if i%2 == 0], fps=fps)

    def save(self, path):
        self.critic.model.save(join(path, 'critic.h5'))
        self.target.model.save(join(path, 'target.h5'))

    def load(self, path):
        self.critic.model = load_model(join(path, 'critic.h5'))
        self.target.model = load_model(join(path, 'target.h5'))

        