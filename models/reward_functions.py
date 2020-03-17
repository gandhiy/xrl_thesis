import shap
import numpy as np


from pdb import set_trace as debug


class Identity:
    def __init__(self, predictor=None):
        pass
    
    def reward_function(self, batch, **kwargs):
        return batch.reward, kwargs



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
        x_test = np.array(batch.next_state)[0:kwargs['num_to_explain']]
        if(kwargs['explainer'] is None or kwargs['num_explainer_summaries']):
            x_train = shap.kmeans(x_train, kwargs['num_explainer_summaries'])
            kwargs['explainer'] = shap.KernelExplainer(self.shap_predict, x_train)


        return kwargs['explainer'].shap_values(x_test, nsamples=10, l1_reg='aic', silent=True)


class additive_SHAP(SHAP):
    def __init__(self, predictor):
        super(additive_SHAP, self).__init__(predictor)

    def reward_function(self, batch, **kwargs):

        shap_vals = self.get_shap_vals(batch, **kwargs)
        kwargs['state'].update(self.plot_shap_vals(shap_vals))

        total_val = np.mean(np.abs(shap_vals))
        kwargs['state']['training/shap_reward'] = total_val
    
        return total_val, kwargs
    
    def plot_shap_vals(self, vals):
        out_state = {}
        for i, a_list in enumerate(vals):
            per_obs_mean = np.mean(np.abs(a_list), axis=0)
            for j, obs in enumerate(per_obs_mean):
                out_state[f'shap_vals/obs_{j}_act_{i}'] = obs
        return out_state