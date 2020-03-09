import numpy as np
import imageio

from .base import base
from .networks import Actor, Critic
from core.replay_experience import Transition


class DDPGAgent(base):
    def __init__(
        self, env, reward_class, batch_size=256, memory_size=1028, gamma=0.95, epsilon = 1.0, 
        epsilon_min=0.01, epsilon_decay=0.995, exploration_fraction=0.1, update_timesteps=50, 
        tau=0.01, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.99, logger_steps = 500,
        learning_starts = 500, action_replay=False, render=False, explainer_updates=256, explainer_summarizing=25,
        summarize_shap=True, num_to_explain=5, val_eps = 10, val_numtimesteps = 1000,
        save_paths = '/Users/yashgandhi/Documents/xrl_thesis/saved_models'):

        super(DDPGAgent, self).__init__(
            env, learning_rate, beta_1, beta_2, tau, batch_size, gamma, epsilon,
            epsilon_min, epsilon_decay, exploration_fraction, update_timesteps,
            logger_steps, action_replay, render, val_eps, val_numtimesteps
        
        )

        self.argmax = False
        
        # setup models and optimizers
        self.behavior_q = Critic(self.env.observation_space.shape, self.env.action_space.shape)
        self.behavior_q.init_model()
        self.behavior_q.build_opt(self.learning_rate, self.beta_1, self.beta_2)
        
        self.target_q = Critic(self.env.observation_space.shape, self.env.action_space.shape)
        self.target_q.init_model()

        # self.target_q.build_opt(self.learning_rate, self.beta_1, self.beta_2) not entirely necessary since we'll be updating the weights heuristically
        
        self.behavior_pi = Actor(self.env.observation_space.shape, self.env.action_space.shape, self.env.action_space.high)
        self.behavior_pi.init_model()
        self.behavior_pi_AdamOpt = self.behavior_pi.build_opt(self.learning_rate, self.beta_1, self.beta_2)
        self.get_critic_grad = self.behavior_pi.get_grads(self.behavior_q.model)
        
        self.target_pi = Actor(self.env.observation_space.shape, self.env.action_space.shape, self.env.action_space.high)
        self.target_pi.init_model()
        self.transfer_weights()
        


        # explainer parameters
        self.explainer = None
        self.explainer_updates = explainer_updates
        self.num_explainer_summaries = explainer_summarizing
        self.shap_summary = summarize_shap
        self.num_to_explain = num_to_explain

        
        self.reward_function = reward_class(self.shap_predict).reward_func






    def act(self, st):
        
        if np.random.rand() <=self.epsilon:
            at = self.env.action_space.sample()
        else:
            at = self.behavior_pi.predict(st)[0]

        
        if(self.action_replay): # repeat the same action 3 times
            for _ in range(3):
                st = self.act_once(at, st)
            return st
        else:
            return self.act_once(at, st)
        
    def shap_predict(self, st):
        return self.behavior_predict(st)


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
        ret = []
        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        
        states = np.array(batch.state)
        actions = np.array(batch.action)
        states_tp1 = np.array(batch.next_state)
        actions_tp1 = self.target_predict(states_tp1)
        
        #### CRITIC UPDATE ####
        
        # bellman update
        mask = np.ones(self.batch_size)*([not l for l in batch.done])
        mask = mask.reshape((-1, 1))

        y = self.target_q.model.predict([states_tp1, actions_tp1])
        y *= self.gamma
        y *= mask
        
        # apply shap updates here if desired
        tmp, self._parameter_dict = self.reward_function(batch, **self._parameter_dict)
        
        y += np.array(tmp).reshape((-1, 1))
        ret.append(self.behavior_q.model.evaluate([states, actions], y, verbose=0)) # loss of critic model
        self.behavior_q.model.fit([states, actions], y, verbose=0)

        
        #### ACTOR UPDATE ####
        acts = self.behavior_pi.predict(states)
        
        # gets the gradient of the critic output with respect to the input actions
        action_grads = self.get_critic_grad([states, acts])
        # apply gradients 
        self.behavior_pi_AdamOpt([batch.state, np.array(action_grads).reshape(-1, self.env.action_space.shape[0])])
       
        
        return ret



    def learn(self, total_timesteps):
        st = self.env.reset()
        assert self.learning_starts < total_timesteps
        
        for tt in range(total_timesteps):
            
            self.update_dictionary()
            self._current_timestep = tt
            st = self.act(st)
            
            if(self.memory.can_sample(self.batch_size) and tt > self.learning_starts):
                loss = self.update_on_batch()   
                if (tt+1)%self.update_timesteps == 0:
                    self.transfer_weights()
                    
            
                if (tt+1)%self.logging_step == 0:
                    
                    print(
                        f"Episodes: {self._num_episodes} \t Average Reward {self._mean_eps_rew:.4f}",
                        f"\t Critic Loss: {loss[0]:.2E} \t Timesteps: {tt+1:.1f}"
                    )
                    
            if self.epsilon > self.epsilon_min and tt < (self.exploration_fraction*total_timesteps):
                self.epsilon *= self.epsilon_decay
                        
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
            action = self.predict(obs)[0]
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


##### ADD ACTION REPLAY IN BOTH DDPG #####