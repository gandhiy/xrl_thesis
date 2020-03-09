import shap
import numpy as np

from core.tools import tfSummary

from pdb import set_trace as debug

"""
Old methods
class rfunc:
    def __init__(self, type):
        self.type = type

    def reward_func(self, batch, **kwargs):
        raise NotImplementedError

    def get_type(self):
        return self.type

class additive_DQN_SHAP(rfunc):
    def __init__(self):
        super(additive_DQN_SHAP, self).__init__('additive_DQN_SHAP')

    def shap_predict(self, P):
        l = []
        for x in P:
            l.append(np.argmax(self.behavior.predict(np.expand_dims(x, axis=0))))
        return np.array(l, dtype='float32')

    def reward_func(self, batch, **kwargs):
        self.behavior = kwargs['behavior']
        x_train = np.array(batch.state)
        x_test = np.array(batch.next_state)[0:kwargs['num_to_explain']]
        if(kwargs['explainer'] is None or kwargs['_current_timestep']%kwargs['explainer_updates'] == 0):
            x_train = shap.kmeans(x_train, kwargs['num_explainer_summaries'])
            kwargs['explainer'] = shap.KernelExplainer(self.shap_predict, x_train)
        shap_vals = kwargs['explainer'].shap_values(x_test, nsamples=10, l1_reg='aic', silent=True)
        
        return np.sum((np.sum(shap_vals, axis=0)/len(shap_vals))), kwargs        

class additive_DDPG_SHAP(rfunc):
    def __init__(self):
        super(additive_DDPG_SHAP, self).__init__('additive_DDPG_SHAP')

    def shap_predict(self, P):
        l = []
        for x in P:
            l.append(self.behavior_pi.predict(x))

        return np.array(l, dtype='float32')

    def reward_func(self, batch, **kwargs):
        self.behavior_pi = kwargs['behavior_pi']
        x_train = np.array(batch.state)
        x_test = np.array(batch.next_state)[0:kwargs['num_to_explain']]
        if(kwargs['explainer'] is None or kwargs['num_explainer_summaries']):
            x_train = shap.kmeans(x_train, kwargs['num_explainer_summaries'])
            kwargs['explainer'] = shap.KernelExplainer(self.shap_predict, x_train)
        shap_vals = kwargs['explainer'].shap_values(x_test, nsamples=10, l1_reg='aic', silent=True)

        return np.sum((np.sum(shap_vals, axis=0)/len(shap_vals))), kwargs
"""

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
        l = []
        for x in P:
            l.append(self.predictor(x))
        return np.array(l, dtype='float32')


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
        total_val = np.sum((np.sum(shap_vals, axis=0)/len(shap_vals)))
        tv = tfSummary('training/shap_reward', total_val)
        kwargs['summary_writer'].add_summary(tv, kwargs['_current_timestep'])
        return total_val, kwargs
    

