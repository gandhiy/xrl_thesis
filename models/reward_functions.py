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
    

    
  
class SHAP(reward):
    def __init__(self, predictor):
        super(SHAP, self).__init__(predictor)
        
    def get_shap_vals(self, batch, **kwargs):
        
        ind = np.random.choice(np.arange(len(batch.state)), size=(kwargs['samples']), replace = False)
        x_train = np.array(batch.state)[ind]
        x_test = np.array(batch.state)[ind]
        
        try:
            self.actions = np.array(batch.action)[ind]
        except:
            self.actions = np.array(batch[3])[ind]

        if(kwargs['episode_number'] % kwargs['save_log'] == 0):
            kwargs['explainer'] = shap.KernelExplainer(self.predictor, x_train)
        
        return kwargs['explainer'].shap_values(x_test, nsamples=50, l1_reg='aic', silent=True)

    def plot_shap_vals(self, vals):
        raise NotImplementedError
       

class dqn_shap(SHAP):
    def __init__(self, predictor):
        super(dqn_shap, self).__init__(predictor)

    def reward_function(self, batch, plot_reward=True, **kwargs):
        #(action_space x samples x obs_space)
        self.t = kwargs['state']['training_iteration']
        vals = np.array(self.get_shap_vals(batch, **kwargs))
        shap_vals = []
        for i,a in enumerate(self.actions):
            shap_vals.append(vals[a, i])
        
        kwargs['state'].update(self.plot_shap_vals(shap_vals))
        total_val = np.sum(np.abs(shap_vals))
        if(plot_reward):
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

    def reward_function(self, batch, plot_reward=True, **kwargs):
        self.t = kwargs['training_iteration']
        shap_vals = np.array(self.get_shap_vals(batch, **kwargs)).squeeze()
        
        kwargs['state'].update(self.plot_shap_vals(shap_vals))
        total_val = np.sum(np.abs(shap_vals))

        if(plot_reward):
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

    def reward_function(self, batch, **kwargs):
        p = kwargs['episode_number']/kwargs['total_episodes']
        t = kwargs['training_iteration']

    
        if p <= 0.45:
            shap_reward, kwargs  = self.dqn_shap.reward_function(batch, False, **kwargs)
            kwargs['state']['training/shap_curriculum_reward'] = (shap_reward, t)
            return shap_reward, kwargs
        elif p <= 0.75:
            px = (10/3.0)*(p - 0.45)
            qx = 1 - px
            shap_reward, kwargs  = self.dqn_shap.reward_function(batch, False, **kwargs)
            r = px*shap_reward + qx*(kwargs['curriculum_balance']*np.mean(np.array(batch.reward)))
            kwargs['state']['training/shap_curriculum_reward'] = (r, t)
            return r, kwargs
        else:
            kwargs['state']['training/shap_curriculum_reward'] = (np.mean(np.array(batch.reward)), t)
            return batch.reward, kwargs
            
class ddpg_shap_curriculum(SHAP):
    def __init__(self, predictor):
        super(ddpg_shap_curriculum, self).__init__(predictor)
        self.ddpg_shap = ddpg_shap(predictor)

    def reward_function(self, batch, **kwargs):
        p = kwargs['episode_number']/kwargs['total_episodes']
        t = kwargs['training_iteration']

    
        if p <= 0.45:
            shap_reward, kwargs  = self.ddpg_shap.reward_function(batch, False, **kwargs)
            kwargs['state']['training/shap_curriculum_reward'] = (shap_reward, t)
            return shap_reward, kwargs
        elif p <= 0.75:
            px = (10/3.0)*(p - 0.45)
            qx = 1 - px
            shap_reward, kwargs  = self.ddpg_shap.reward_function(batch, False, **kwargs)
            r = px*shap_reward + qx*(kwargs['curriculum_balance']*np.mean(np.array(batch.reward)))
            kwargs['state']['training/shap_curriculum_reward'] = (r, t)
            return r, kwargs
        else:
            kwargs['state']['training/shap_curriculum_reward'] = (np.mean(np.array(batch.reward)), t)
            return batch.reward, kwargs


class mountaincar_curriculum:
    def __init__(self, predictor):
        pass

    def reward_function(self, batch, **kwargs):
        p = kwargs['episode_number']/kwargs['total_episodes']
        t = kwargs['training_iteration']
        states = np.array(batch.state)
        if p <= 0.25:
            r = 10*states[:, 1]
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        elif p <= 0.55:
            px = (10/3.0)*(p - 0.25)
            qx = 1 - px
            r = 10*(qx)*states[:, 1] + px*states[:, 0]
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        elif p <= 0.75:
            r = states[:, 0]
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        else:
            r = np.array(batch.reward)
            kwargs['state']['training/curriculum_reward'] = (np.mean(r), t)
            return r, kwargs
        
class lunarlander_curriculum:
    def __init__(self, predictor):
        pass

    def reward_function(self, batch, plot_reward=True, **kwargs):
        p = kwargs['episode_number']/kwargs['total_episodes']
        t = kwargs['training_iteration']
        states = np.array(batch.state)

        