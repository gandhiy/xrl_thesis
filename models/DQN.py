import shap
import imageio
import numpy as np 
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from core.replay_experience import Transition, ReplayMemory

from pdb import set_trace as debug

class DQNAgent:
    def __init__(
        self, env, policy, reward_class, batch_size = 256, memory_size=1028, gamma = 0.95, epsilon = 1.0,
        epsilon_min = 0.01, epsilon_decay = 0.995, exploration_fraction=0.1, update_timesteps= 50, tau=0.01, 
        learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.99, logger_steps=500, learning_starts = 1000, 
        explainer_updates = 256, explainer_summarizing=25, summarize_shap = True, num_to_explain = 5):
        

        ## Optimizer
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        ## Environment Parameters
        self.reward_function = reward_class().reward_func 
        self.env = env
        self.batch_size = batch_size
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n 
        self.tau = tau
        self.learning_starts = learning_starts

        
        ## Behavior Model
        self.behavior = policy(self.state_size, self.action_size).model
        self.behavior.compile(optimizer = self._build_opt(), loss = 'mse')
        ## Target model
        self.target = policy(self.state_size, self.action_size).model
        self.target.compile(optimizer=self._build_opt(), loss='mse')
        self.transfer_weights()
        
        ## DQN Parameters
        self.memory = ReplayMemory(capacity=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exploration_fraction = exploration_fraction
        self.update_timesteps = update_timesteps
        self.logging_step = logger_steps
        
        ## logging information
        self._num_episodes = 0
        self._eps_rew = 0
        self._eps_rew_list = []
        self._mean_eps_rew = 0
        self._current_timestep = 0
        
        # explainer parameters
        self.explainer = None
        self.explainer_updates = explainer_updates
        self.num_explainer_summaries = explainer_summarizing
        self.shap_summary = summarize_shap
        self.num_to_explain = num_to_explain


        # put all of the class parameters (minus itself) inside 
        # of this variable and update it using rfunc
        self.__parameter_dict = None
    
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

    def update_dictionary(self):
        self.__parameter_dict = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}


        
    def _predict(self, state):
        return self.target.predict(state)
    
    def predict(self, state):
        return np.argmax(self._predict(state))
        
    def act(self, st):
        if np.random.rand() <=self.epsilon:
            at = self.env.action_space.sample()
        else:
            at = np.argmax(self.behavior.predict(np.expand_dims(st, axis=0)))
        
        snext, rt, done, _ = self.env.step(at)
        if done:
            self._num_episodes += 1
            self._eps_rew_list.append(self._eps_rew)
            self._mean_eps_rew = sum(self._eps_rew_list)/len(self._eps_rew_list)
            self._eps_rew = 0
            snext = self.env.reset()
        self._eps_rew += rt
        self.memory.push(st, at, snext, done, rt)
        return snext
    

    def learn(self, total_timesteps=10000):
        st = self.env.reset()
        assert self.learning_starts < total_timesteps
        
        for tt in range(total_timesteps):
            self.update_dictionary()
            self._current_timestep = tt
            st = self.act(st)
            
            if(self.memory.can_sample(self.batch_size) and tt > self.learning_starts):
                ## learning starts

                # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                batch = Transition(*zip(*self.memory.sample(self.batch_size))) 

                mask = np.ones(self.batch_size)*([not l for l in batch.done])
                y = self.gamma * np.amax(self._predict(np.array(batch.next_state)), axis=1) # uses target model
                y *= mask
            
                """ 
                iik
                x_train = np.array(batch.state)
                x_test = np.array(batch.next_state)[0:5]
                

                if(self.explainer is None or tt%self.explainer_updates == 0):
                    x_train = shap.kmeans(x_train, self.num_explainer_summaries)
                    if(self.shap_summary):
                        self.explainer = shap.KernelExplainer(self.shap_predict, x_train)


                shap_vals = self.explainer.shap_values(x_test, nsamples=10, l1_reg='aic', silent=True)
                
                y += np.sum((np.sum(shap_vals, axis=0)/len(shap_vals)))
                """
                tmp, self.__parameter_dict = self.reward_function(batch, **self.__parameter_dict)
                y += tmp 
                
                # N x num_actions
                target = self.behavior.predict(np.array(batch.state))
                target[np.arange(self.batch_size), batch.action] = y
                loss = self.behavior.evaluate(np.array(batch.state), target, verbose=0)
                self.behavior.fit(np.array(batch.state), target, verbose=0)
                
                
                if (tt+1)%self.update_timesteps == 0:
                    self.transfer_weights()
                    
                    
                if((tt+1)%self.logging_step == 0):
                    print(
                        f"Episodes: {self._num_episodes} \t Average Reward {self._mean_eps_rew:.4f}",
                        f"\t Loss: {loss:.4f} \t Timesteps: {tt+1}"
                    )
                    self._num_episodes = 0
                    
                                
            if self.epsilon > self.epsilon_min and tt < (self.exploration_fraction*total_timesteps):
                self.epsilon *= self.epsilon_decay
                        
    def create_gif(self, frames=500, fps=29, save='model'):
        """
         Makes a gif of an agent
        """
        images = []
        obs = self.env.reset()
        img = self.env.render(mode='rgb_array')

        eps_rew = 0
        episode = 1
        for i in range(frames):
            images.append(img)
            action = self.predict(np.expand_dims(obs, axis=0))
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

                    
