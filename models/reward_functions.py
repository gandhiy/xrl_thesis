import shap
import numpy as np


from pdb import set_trace as debug


class Identity:
    def __init__(self, predictor=None):
        pass
    
    def reward_function(self, batch, **kwargs):
        try:
            return batch.reward, kwargs
        except AttributeError:
            return batch[3], kwargs


class reward:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def update_predictor(self,new_predictor):
        self.predictor = new_predictor
    
    def shap_predict(self, P):
        return self.predictor.predict(P)

   

class SHAP(reward):
    def __init__(self, predictor):
        super(SHAP, self).__init__(predictor)


    def get_shap_vals(self, batch, **kwargs):
        ind = np.random.choice(np.arange(len(batch.state)), size=(kwargs['samples']))
        x_train = np.array(batch.state)[ind]
        x_test = np.array(batch.state)[ind]
        self.actions = np.array(batch.action)[ind]

        kwargs['explainer'] = shap.KernelExplainer(self.shap_predict, x_train)
        return kwargs['explainer'].shap_values(x_test, nsamples=50, l1_reg='aic', silent=True)

    def plot_shap_vals(self, vals):
        raise NotImplementedError
       

class dqn_shap(SHAP):
    def __init__(self, predictor):
        super(dqn_shap, self).__init__(predictor)

    def reward_function(self, batch, **kwargs):
        #(action_space x samples x obs_space)
        self.t = kwargs['state']['training_iteration']
        vals = np.array(self.get_shap_vals(batch, **kwargs))
        shap_vals = []
        for i,a in enumerate(self.actions):
            shap_vals.append(vals[a, i])
        
        kwargs['state'].update(self.plot_shap_vals(shap_vals))
        total_val = np.mean(np.abs(shap_vals))
        kwargs['state']['training/shap_reward'] = (total_val, self.t)
        return total_val, kwargs


    def plot_shap_vals(self, vals):
        out_state = {}
        
        per_obs_mean = np.mean(np.abs(vals), axis=0)
        for j, obs in enumerate(per_obs_mean):
            out_state[f'shap_vals/obs_{j}'] = (obs, self.t)
        return out_state

class ddpg_shap(SHAP):
    def __init__(self, predictor):
        super(ddpg_shap, self).__init__(predictor)

    def reward_function(self, batch, **kwargs):
        self.t = kwargs['state']['training_iteration']
        vals = np.array(self.get_shap_vals(batch, **kwargs))
        shap_vals = np.squeeze(vals)
        
        kwargs['state'].update(self.plot_shap_vals(shap_vals))
        total_val = np.mean(np.abs(shap_vals))
        kwargs['state']['training/shap_reward'] = (total_val, self.t)
        return total_val, kwargs

    def plot_shap_vals(self, vals):
        out_state = {}

        per_obs_mean = np.mean(np.abs(vals), axis=0)
        for j, obs in enumerate(per_obs_mean):
            out_state[f'shap_vals/obs_{j}'] = (obs, self.t)
        return out_state
    

class dqn_shap_curriculum(SHAP):
    def __init__(self, predictor):
        super(dqn_shap_curriculum, self).__init__(predictor)
        self.dqn_shap = dqn_shap(predictor)
        self.Identity = Identity(predictor)

    def reward_function(self, batch, **kwargs):
        if kwargs['state']['timestep'] < .33 * kwargs['_total_timesteps']:
            self.dqn_shap.reward_function(batch, **kwargs)


class mountaincar_curriculum:
    def __init__(self, predictor):
        pass

    def reward_function(self, batch, **kwargs):
        p = kwargs['episode_number']/kwargs['total_episodes']
        t = kwargs['training_iteration']
        states = np.array(batch.state)
        if p < 0.20:
            r = 10*np.abs(states[:, 1])
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        elif p < 0.4:
            r = 10*np.abs(states[:, 1]) + states[:, 0]
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        elif p < 0.6:
            r = states[:, 0]
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        else:
            r = np.array(batch.reward)
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        
    
