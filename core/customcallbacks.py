import shap
import numpy as np
import tensorflow as tf

from stable_baselines.common.callbacks import BaseCallback

from pdb import set_trace as debug

class CustomCallback(BaseCallback):
    def __init__(self, d, process_shap_values, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.dictionary = d
        self.is_tb_set = False
        self.shap_vals = []

    def f(self, p):
        if(self.model.action_probability(p[0]) is None):
            return np.array([self.model.predict(x)[0] for x in p], dtype='float32')
        else:
            
            return np.array([self.model.action_probability(x) for x in p], dtype='float32')

    def generate_samples(self):
        try:
            replay = self.model.replay_buffer.sample(self.dictionary['shap_samples'] + self.dictionary['test_samples'])
            X, _ = replay[0], replay[1]
        except AttributeError:
            seg = self.locals['seg_gen'].__next__()
            X = seg['observations']

        return X[:self.dictionary['shap_samples']], X[:-self.dictionary['test_samples']]

    def _on_step(self) -> bool:

        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self.model.episode_reward))
                self.model.summary = tf.summary.merge_all()                
            self.is_tb_set = True
        
        if(self.model.replay_buffer.can_sample(self.dictionary['shap_samples'] + self.dictionary['test_samples']) and self.num_timesteps % self.dictionary['log_shap_vals'] == 0):
            X_train, X_test = self.generate_samples()
            if(self.dictionary['summarize']):
                assert self.dictionary['num_summaries'] is not None, f"If summarize is True, num_summaries needs to be set"
                X_train = shap.kmeans(X_train, self.dictionary['num_summaries'])
            explainer = shap.KernelExplainer(self.f, X_train)
            shap_values = explainer.shap_values(X_test, nsample=10, l1_reg='aic', silent=True)

            for i,a_list in enumerate(shap_values):
                per_obs_mean = np.mean(np.abs(a_list), axis=0)
                for j,obs in enumerate(per_obs_mean):
                    self.locals['writer'].add_summary(
                        tf.Summary(value = [tf.Summary.Value(tag=f'shap_vals/obs_{j}_act_{i}', simple_value=obs)]),
                        self.num_timesteps
                    )

            

        # for i in range(shap_values.shape[1]):
        #     if(len(shap_values.shape) > 2):
        #         for j in range(shap_values.shape[2]):
        # for i in range(s.shape[1]):
        #     if(len(s.shape) > 2):
        #         for j in range(s.shape[2]):
        #             results['action_{}_obs_{}_mean'.format(i,j)] = np.mean(s[:,i,j])
        #             results['action_{}_obs_{}_stddev'.format(i,j)] = np.std(s[:,i,j])
        #     else:
        #         results['action_0_obs_{}_mean'.format(i)] = np.mean(s[:,i])
        #         results['action_0_obs_{}_std'.format(i)] = np.std(s[:,i])


        return True


