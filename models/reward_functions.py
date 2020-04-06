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
        x_train = np.array(batch.state)
        x_test = np.array(batch.state)
        kwargs['explainer'] = shap.KernelExplainer(self.shap_predict, x_train)
        return kwargs['explainer'].shap_values(x_test, nsamples=50, l1_reg='aic', silent=True)

    def plot_shap_vals(self, vals):
        out_state = {}
        for i, a_list in enumerate(vals):
            per_obs_mean = np.mean(np.abs(a_list), axis=0)
            for j, obs in enumerate(per_obs_mean):
                out_state[f'shap_vals/obs_{j}_act_{i}'] = obs
        return out_state
       

class dqn_shap(SHAP):
    def __init__(self, predictor):
        super(dqn_shap, self).__init__(predictor)

    def reward_function(self, batch, **kwargs):
        #(action_space x samples x obs_space)
        vals = np.array(self.get_shap_vals(batch, **kwargs))
        shap_vals = []
        for i,a in enumerate(batch.action):
            shap_vals.append(vals[a, i])
        
        kwargs['state'].update(self.plot_shap_vals(shap_vals))
        total_val = np.sum(np.abs(shap_vals))
        kwargs['state']['training/shap_reward'] = total_val
        return total_val, kwargs


    def plot_shap_vals(self, vals):
        out_state = {}
        
        per_obs_mean = np.sum(np.abs(vals), axis=0)
        for j, obs in enumerate(per_obs_mean):
            out_state[f'shap_vals/obs_{j}'] = obs
        return out_state



class dqn_shap_curriculum(SHAP):
    def __init__(self, predictor):
        super(dqn_shap_curriculum, self).__init__(predictor)
        self.dqn_shap = dqn_shap(predictor)
        self.Identity = Identity(predictor)

    def reward_function(self, batch, **kwargs):
        if kwargs['state']['timestep'] < .33 * kwargs['_total_timesteps']:
            self.dqn_shap.reward_function(batch, **kwargs)
