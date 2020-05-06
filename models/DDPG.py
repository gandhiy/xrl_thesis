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
from tensorflow.keras.layers import ELU


from pdb import set_trace as debug



class DDPGAgent(base):
    """
     Class for training a DDPG Agent on a continuous environment

     NON-DEFAULT ARGUMENTS
     * env(gym.env): Continuous action space environment
     
     * reward_class(reward_functions.reward): reward function to train DDPG, see models/reward_functions.py for more information

     * critic_optimizer(tf.keras.optimizer): optimizing model for updating the critic
     
     * actor_optimizer(tf.keras.optimizer): optimizing model for updating the actor
     
     DEFAULT ARGUMENTS
     * model_name(string): name to save model by (default -> 'temp')
     * batch_size(int): number of samples to train on (default -> 256)
     * memory_size(int): size of the replay buffer (default -> 50000)
     * gamma (float): discount factor (default -> 0.99)
     * tau (float): soft actor critic update parameter (default -> 0.001),
     * warmup (int): number of environment warmup episodes to set up replay buffer (default -> 25)
     * epsilon_min (float): minimum noise value during training (default -> 0.001)
     * epsilon_episodes (int): number of episodes to decay epsilon
     * sigma (float): OU noise sigma factor (default -> 0.2)
     * mu (float): OU noise mean factor (default -> 0)
     * theta (float): OU noise speed factor (default -> 0.15)
     * validation_logging (int): number of episodes between validation testing (default -> 25) 
     * validation_episodes (int): number of episodes to validate on (default -> 5)
     * save_paths (string/file path): place to save model (default -> .)
     * actor_layers (array): length and size of actor network  i.e. number of nodes in layer 1 is given by layers[0], nodes in layer 2 is given by layers[1] if defined, etc. (default -> [400, 300])
     * critic_layers (array): length and size of critic network (default -> [400, 300])
     * actor_reg (float): actor weight regularization (default -> 0.01)
     * critic_reg (float): critic weight regularization (default -> 0.01)
     * activation (string or tensorflow.keras.layers): nonlinear activation function for both critic and actor
     * save_episodes (int): intermediary saving episode step (default -> 100)
     * tb_log (bool): whether to set up a tensorboard for logging (default -> True)
     * verbose (int): verbose output (default -> 0)
     * explainer_samples (int): number of samples to explain on from batch; must be less than or equal to batch size and -1 implies batch size (default -> -1)
     * RANDOM_SEED: seeded learning
    """

    def __init__(
        self, env, reward_class, critic_optimizer, actor_optimizer,
        model_name='temp', batch_size = 256, memory_size=50000, gamma = 0.99, tau = 0.001, 
        warmup = 25,  epsilon_min = 0.001, epsilon_episodes=0, sigma = 0.2, mu = 0, 
        theta = 0.15, validation_logging = 25,  validation_episodes = 5,
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models',
        actor_layers = [400,300], critic_layers=[400,300], actor_reg = 0.01, critic_reg = 0.01,
        activation=ELU, save_episodes = 100, tb_log=True, verbose=0, explainer_samples = -1,
        RANDOM_SEED=1234):

        
        
        super(DDPGAgent, self).__init__(
            env, model_name, save_paths, tau, batch_size, gamma, 
            memory_size, validation_logging, warmup, validation_episodes, 
            save_episodes, verbose, tb_log, RANDOM_SEED)

        
        if(len(self.env.action_space.shape) > 0):
            self.num_actions = self.env.action_space.shape[0]
        else:
            self.num_actions = self.env.action_space.n

        self.critic = Critic(self.env.observation_space.shape, self.num_actions, critic_layers, critic_reg, activation=activation)
        self.target_critic = Critic(self.env.observation_space.shape, self.num_actions, critic_layers, critic_reg, activation=activation)
        self.critic.initialize(critic_optimizer)
        self.get_grads = self.critic.get_grads()


        self.actor = Actor(self.env.observation_space.shape, self.num_actions, actor_layers, actor_reg, activation=activation, range_high=self.env.action_space.high, range_low=self.env.action_space.low)
        self.target_actor = Actor(self.env.observation_space.shape, self.num_actions, actor_layers, actor_reg, activation=activation, range_high=self.env.action_space.high, range_low=self.env.action_space.low)
        self.actor_optimizer = self.actor.initialize(actor_optimizer)
        self.reward_function = reward_class(self.actor.model.predict).reward_function
        
        self.transfer_weights()


        files = [f for f in os.listdir(self.save_path) if 'DDPG' in f]
        self.save_path = join(self.save_path, f'DDPG{len(files) + 1}')
        logdir = join(self.save_path, 'tensorboard_logs')
        if(self.tb_log):
            self.writer = summary(tf.summary.FileWriter(logdir))


        self.explainer = None
        if(explainer_samples < 0):
            self.samples = self.batch_size
        else:
            self.samples = explainer_samples

        self.per_step_reward = []

        self.mu = mu 
        self.sigma = sigma
        self.theta = theta
        self.epsilon_min = epsilon_min
        self.epsilon_episodes = epsilon_episodes
        self.exploration_noise = Ornstein_Uhlenbeck_Noise(self.num_actions, self.mu, self.theta, self.sigma)
        self.__best_val_score = -1000000



    def transfer_weights(self):
        self.target_critic.transfer_weights(self.critic.model, self.tau)
        self.target_actor.transfer_weights(self.actor.model, self.tau)

    def _action(self, obs):
        return self.actor.predict(obs)

    def predict(self, obs):
        return self.target_actor.predict(obs)
    
    def environment_step(self, obs, done, warming_up):
        self.environment_iteration += 1
        self.count += 1
        at = self._action(obs)
        

        # four possible combinations of noise: 
        # 1.) noise during warmup only: warming_up > 0 & epsilon_episodes =0
        # 2.) noise during training only: warming_up = 0 & epsilon_episodes > 0
        # 3.) noise during warmup and during training: warming_up > 0 & epsilon_episodes > 0 
        # 4.) no noise during warmup or training: warming_up = 0 & epsilon_episodes = 0

        if(warming_up):
            p = self.episode_number/self.total_episodes
            n = self.exploration_noise.noise()
        else:
            p = max(self.epsilon_min, 1 - self.episode_number/(self.epsilon_episodes + 1e-9)) 
            n = self.exploration_noise.noise()

        for i in range(len(n)):
            self.state[f'noise/OU_noise_act_{i+1}'] = (n[i], self.environment_iteration)
            self.state[f'noise/decay_factor'] = (p, self.environment_iteration)
            self.state[f'noise/action_{i+1}_noise'] = (p*n[i], self.environment_iteration)
            
        at = at + (p*n)
        
        obs_t, rt, done, _ = self.env.step(at)
        
        self.state['training/per_step_reward'] = (rt, self.environment_iteration)
        self.per_step_reward.append(rt)
        
        trajectory = [obs, at, obs_t, done, rt]
        obs = obs_t
        if done:
            self.count = 0
            self.exploration_noise.reset()
            self.state['training/episode_reward'] = (np.sum(self.per_step_reward), self.episode_number)
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
        y = mask * self.gamma * self.target_critic.batch_predict(next_state, self.target_actor.batch_predict(next_state)) + np.array(r).reshape((-1, 1))


        history = self.critic.fit(states, actions, y)
        self.state['training/critic_loss'] = (np.mean(history.history['loss']), self.training_iteration)


        q_grads_wrt_a = self.get_grads([states, self.actor.batch_predict(states)]) # value of the target policy over the behavior policy
        mean_grads = self.actor_optimizer([states, np.array(q_grads_wrt_a).reshape(-1, self.num_actions)])
        self.state['training/actor_loss'] = (np.mean(mean_grads[0]), self.training_iteration)

    def learn(self, episodes=1000):
        self.total_episodes = episodes + self.warmup
        self.episode_number = 0
        self.training_iteration = 0
        self.environment_iteration = 0
        self.update_dictionary()

        obs = self.env.reset()
        for e in tqdm(range(self.total_episodes)):
            self.episode_number = e
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
                        

                    if(self.episode_number%self.save_log == 0 and self.count == 0):
                        p = join(self.save_path, f'episode_{self.episode_number}')
                        os.makedirs(p, exist_ok=True)
                        self.save(p)

                    

                self.update_dictionary()
                if self.tb_log:
                    self.writer.update(self.state)
                elif not self.tb_log and done:
                    print(f"Episode Reward: {self.state['training/episode_reward'][0]}")
                        
    def validate(self):
        obs = self.v_env.reset()
        episode_rewards = []
        for _ in range(self.num_validation_episode):            
            done = False
            reward = 0
            while not done:
                action = self.target_actor.predict(obs)
                obs, rt, done, _ = self.v_env.step(action)
                reward += rt
            obs = self.v_env.reset()
            episode_rewards.append(reward)
        avg_reward = np.mean(episode_rewards)
        if(avg_reward > self.__best_val_score):
            self.__best_val_score = avg_reward
            p = join(self.save_path, 'best_model')
            os.makedirs(p, exist_ok=True)
            self.save(p)
        
        self.state[f'validation/average_reward'] = (avg_reward, self.episode_number)
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

            
            
            
            
            
