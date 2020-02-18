import shap
import numpy as np

class rfunc:
    def __init__(self, type):
        self.type = type

    def reward_func(self, batch, **kwargs):
        raise NotImplementedError

    def get_type(self):
        return self.type

class Identity(rfunc):
    def __init__(self):
        super(Identity, self).__init__('Identity')

    def reward_func(self, batch, **kwargs):
        return batch.reward, kwargs


class additive_SHAP(rfunc):
    def __init__(self):
        super(additive_SHAP, self).__init__('additive_SHAP')

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


            
        
    


