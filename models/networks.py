import keras
import numpy as np
import tensorflow as tf
import keras.backend as K


from keras.models import Sequential, Model, load_model
from keras.layers import Dense, GaussianNoise, Input, concatenate
from keras.layers import BatchNormalization, Flatten, Lambda, Conv2D
from keras.layers import Softmax, MaxPool2D
from keras.optimizers import Adam
from keras.initializers import RandomUniform


# for DQN
class MlpPolicy:
    def __init__(self, 
        in_size, 
        act_size, 
        model_params = [128, 64]
    ):
        self.model = Sequential()
        self.model.add(Dense(model_params[0], input_shape = in_size, activation='relu'))
        
        for m in model_params[1:]:
            self.model.add(Dense(m, activation='relu'))
        
        self.model.add(Dense(act_size, activation='softmax'))

class CNNPolicy:
    def __init__(self, 
        in_size, 
        act_size, 
        model_params = [128, 64]
    ):
        self.model = Sequential()
        self.model.add(Conv2D(model_params[0], 3, input_shape=in_size, data_format='channels_last', activation='relu'))
        self.model.add(MaxPool2D())
        for m in model_params[1:]:
            self.model.add(Conv2D(m, 3, activation='relu'))
            self.model.add(MaxPool2D())

        self.model.add(Flatten())
        self.model.add(Dense(act_size))
        


# DDPG Actor & Critic networks
class Critic:
    def __init__(self, obs_shape, act_shape, *args):
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        
    def __init_model__(self, model_params = [128, 64]):
        
        inp_1 = Input((self.obs_shape))
        inp_2 = Input((self.act_shape))
        
        x = Dense(model_params[0], activation='relu')(inp_1)
        x = concatenate([x, inp_2])
        for m in model_params[1:]:
            x = Dense(m, activation='relu')(x)
            x = BatchNormalization()(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        self.model = Model([inp_1, inp_2], out)
        
    def __build_opt__(self, lr, b1, b2):
        return Adam(
            learning_rate=lr,
            beta_1 = b1,
            beta_2 = b2,
            clipvalue=0.5
        )
    
    def init_model(self):
        self.__init_model__()
    
    
    def build_opt(self, learning_rate, beta_1, beta_2):
        self.model.compile(optimizer=self.__build_opt__(learning_rate, beta_1, beta_2), loss='mse')
   
    
    def predict(self, st, at):
        if(len(st.shape) < 2):
            assert len([st]) == len(at), 'mismatch between number of samples'
            return self.model.predict([[st], at])
        else:
            assert len(st) == len(at), 'mistmatch between number of samples'
            return self.model.predict([st, at])

    
    def transfer_weights(self, model, tau):
        self.model.set_weights(
            [tau*l1 + (1-tau)*l2 for l1, l2 in zip(self.model.get_weights(), model.get_weights())]
        )


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

   

class Actor:
    def __init__(self, obs_shape, act_shape, act_range):
        self.obs_shape = obs_shape
        
        assert len(act_shape) < 2, "Only Box environment allowed"

        self.act_shape = act_shape[0] 
        self.act_range = act_range
        
    def __init_model__(self, model_params=[128, 64]):
        inp = Input((self.obs_shape))
        x = Dense(model_params[0], activation='relu')(inp)
        for m in model_params[1:]:
            x = Dense(m, activation='relu')(x)
            x = BatchNormalization()(x)
        
        x = GaussianNoise(1.0)(x)
        
        # puts action out vals between -1 and 1
        out = Dense(self.act_shape, activation='tanh', kernel_initializer=RandomUniform())(x)
        
        # set to the correct range
        out = Lambda(lambda i: i * self.act_range)(out)
        self.model= Model(inp, out)
        
    def __build_opt__(self, lr, b1, b2):
        # build function to apply gradients to actor model
        act_grads = K.placeholder(shape=(None, self.act_shape))
        clipped_grads = K.clip(act_grads, -0.5, 0.5)
        mean_grad = K.sum(clipped_grads, axis=0)
        update_params = tf.gradients(self.model.output, self.model.trainable_weights, -clipped_grads)
        grads = zip(update_params, self.model.trainable_weights)
        
        
        
        return K.function(
            inputs=[self.model.input, act_grads], outputs=[mean_grad],
            updates=[tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2).apply_gradients(grads)][1:]
        )
    
    def get_grads(self, qmodel):
        return K.function(
            inputs=[qmodel.input[0], qmodel.input[1]],
            outputs=K.gradients(qmodel.output, qmodel.input[1])
        )
    
    def init_model(self):
        self.__init_model__()
        
    def build_opt(self, learning_rate, beta_1, beta_2):
        return self.__build_opt__(learning_rate, beta_1, beta_2)
        
    def predict(self, st):
        if(len(st.shape) < 2):
            return self.model.predict(np.expand_dims(st, axis=0))
        else:
            return self.model.predict(st)
        
    def transfer_weights(self, model, tau):
        self.model.set_weights(
            [tau*l1 + (1-tau)*l2 for l1, l2 in zip(self.model.get_weights(), model.get_weights())]
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)