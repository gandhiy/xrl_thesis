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
    """
     Class for a DQN Agent. Initialize agent and then call
     agent.learn(episodes=N). Only works with discrete space 
     action environments.

     NON-DEFAULT ARGUMENTS
     * env(gym.env): Continuous action space environment
     
     * reward_class(reward_functions.reward): reward function to train DDPG, see models/reward_functions.py for more information

     * opt (tf.keras.optimizer): optimizer for the DQN neural network

     DEFAULT ARGUMENTS
     * model_name (string): name to save model under (default -> 'temp')
     * batch_size(int): number of samples to train on (default -> 256)
     * memory_size(int): size of the replay buffer (default -> 50000)
     * gamma (float): discount factor (default -> 0.99)
     * tau (float): soft actor critic update parameter (default -> 0.001)
     * epsilon_min (float): minimum epsilon value for generating noise (default -> 0.001) 
     * epsilon_decay (float): decay rate for epsilon (default -> 0.995)
     * warmup (int): number of environment warmup episodes to set up replay buffer (default -> 25)
     * validation_logging (int): number of episodes between validation logging (default -> 25)
     * validation_episodes (int): number of episodes to validate on (default -> 5)
     * save_paths (string/file path): path to save model at (default -> .)
     * save_episodes (int): number of episodes between model saves (default -> 100)
     * layers (array): each element represents the number of nodes at layer i (default -> [64, 64])
     * reg (float): L2 neural network layer regularization term (default = 0.01)
     * activation (string, keras.activations): the activation function for the neural network (default -> 'relu')
     * verbose (int): verbose output (default -> 0)
     * tb_log (boolean): whether to log information to the tensorboard (default -> True)
     * explainer_samples (int): number of samples to explain on from batch; must be less than or equal to batch size and -1 implies batch size (default -> -1)
     * RANDOM_SEED (int): default -> 1234

    """


    def __init__(
        self, env, reward_class, opt, policy=None, model_name='temp', 
        batch_size=256, memory_size=50000, gamma=0.995, tau = 0.001,
        epsilon_min = 0.001, epsilon_decay = 0.995, 
        warmup=25, validation_logging = 25, validation_episodes = 5, 
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models', save_episodes = 
        100, layers = [64,64], activation='relu', reg=0.01, verbose=0, tb_log = True, explainer_samples = -1,
        curriculum_balance = 1.0, RANDOM_SEED=1234):

        super(DQNAgent, self).__init__(
        env, model_name, save_paths, tau, batch_size,
        gamma, memory_size, validation_logging, warmup, validation_episodes, 
        save_episodes, verbose, tb_log, RANDOM_SEED)
        

        if(len(self.env.action_space.shape) > 0):
            self.num_actions = self.env.action_space.shape[0]
        else:
            self.num_actions = self.env.action_space.n
        
        
        if policy is None:
            self.critic = DQNPolicy(self.env.observation_space.shape, self.num_actions, layers = layers, activation=activation, reg=reg)
        else:
            self.critic = policy # allows for initialized policies outside of the network

        self.critic.initialize(opt)

        if policy is None:
            self.target = DQNPolicy(self.env.observation_space.shape, self.num_actions, layers = layers, activation=activation, reg=reg)
        else:
            self.target = deepcopy(policy)
        self.target.transfer_weights(self.critic, self.tau)

        self.reward_function = reward_class(self.critic.model.predict).reward_function

        files = [f for f in os.listdir(self.save_path) if 'DQN' in f]
        self.save_path = join(self.save_path, f'DQN{len(files) + 1}')
        if(self.tb_log):
            logdir = join(self.save_path, 'tensorboard_logs')
            self.writer = summary(tf.summary.FileWriter(logdir))

        self.explainer = None
        self.per_step_reward = []

        if(explainer_samples < 0):
            self.samples = self.batch_size
        else:
            self.samples = explainer_samples
        
        self.curriculum_balance = curriculum_balance
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay_factor = epsilon_decay
        self.__best_val_score = -1000000


    def _action(self, obs):
        if(np.random.rand() > self.epsilon):
            return self.critic.predict(obs)
        else:
            return self.env.action_space.sample()

    def predict(self, obs):
        return self.critic.predict(obs)


    def environment_step(self, obs, done):
        self.environment_iteration += 1
        self.count += 1
        at = self._action(obs)
        obs_t, rt, done, _ = self.env.step(at)
        self.per_step_reward.append(rt)
        trajectory = [obs, at, obs_t, done, rt]
        obs = obs_t

        self.state['training/per_step_reward'] = (rt, self.environment_iteration)
        if done:
            self.count = 0
            self.state['training/episode_reward'] = ((np.sum(self.per_step_reward)), self.episode_number) 
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
        self.total_episodes = episodes + self.warmup
        self.update_dictionary()
        self.episode_number = 0
        self.training_iteration = 0
        self.environment_iteration = 0

        obs = self.env.reset()
        for e in tqdm(range(self.total_episodes)):
            self.episode_number = e
            done = False
            self.count = 0
            while not done:
                warming_up = self.warmup > self.episode_number
                obs, done = self.environment_step(obs, done)

                if(self.memory.can_sample(self.batch_size) and not warming_up):
                    
                    
                    self.training_iteration += 1                     
                    self.batch_update()

                    self.target.transfer_weights(self.critic, self.tau)

                    if(self.episode_number%self.validation_logging == 0 and self.count == 1):
                        self.validate()

                    if(self.episode_number%self.save_log == 0 and self.count == 0):
                        p = join(self.save_path, f'episode_{self.episode_number+1}')
                        os.makedirs(p, exist_ok=True)
                        self.save(p)
                    
                    if(self.epsilon > self.epsilon_min):
                        self.epsilon *= self.epsilon_decay_factor
                
                self.state['noise/epsilon'] = (self.epsilon, self.environment_iteration)

                self.update_dictionary()
                if self.tb_log:
                    self.writer.update(self.state)

    def validate(self):
        
        obs = self.v_env.reset()
        episode_rewards = []
        for v in range(self.num_validation_episode):
            done = False
            reward = 0
            while not done:
                action = self.target.predict(obs)
                obs, rt, done, _  = self.v_env.step(action)
                if(self.verbose > 1):
                    print(f"Validation Episode: {v} \t\t Step: {count} \t Action: {action}")
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

    def save(self, path):
        self.critic.model.save(join(path, 'critic.h5'))
        self.target.model.save(join(path, 'target.h5'))

    def load(self, path):
        self.critic.model = load_model(join(path, 'critic.h5'))
        self.target.model = load_model(join(path, 'target.h5'))

        

