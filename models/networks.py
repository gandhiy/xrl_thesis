import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

sys.path.append("..")

from core.tools import ppo_loss, ppo_loss_continuous
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GaussianNoise, Input, concatenate
from tensorflow.keras.layers import BatchNormalization, Flatten, Lambda, Conv2D
from tensorflow.keras.layers import Softmax, MaxPool2D, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform, VarianceScaling

from pdb import set_trace as debug

# for DQN
class DQNPolicy:
    def __init__(self, in_state, in_actions, layers):
        self.obs = in_state
        self.act = in_actions
        self.state_input = Input(shape = self.obs)

        x = Dense(layers[0], activation='relu')(self.state_input)
        for l in layers[1:]:
            x = Dense(l, activation='relu')(x)
        
        out = Dense(self.act, activation='softmax')(x)
        self.model = Model(inputs = [self.state_input], outputs=[out])

    def initialize(self, lr, b1, b2):
        self.model.compile(
            optimizer=Adam(learning_rate=lr, beta_1 = b1, beta_2 = b2, clipvalue = 0.5),
            loss = 'mse',
            metrics = ['accuracy'],
        )
    
    def transfer_weights(self, behavior_model, tau):
        self.model.set_weights(
            [((1 - tau)*l1) + (tau*l2) for l1, l2 in zip(self.model.get_weights(), behavior_model.model.get_weights())]
        )

    def predict(self, st):
        return np.argmax(self.model.predict(np.expand_dims(st, axis=0)))

    def batch_predict(self, st):
        return np.argmax(self.model.predict(st), axis=1)

    def fit(self, states, y):
        return self.model.fit([states], y, verbose=0)

# PPO Actor & Critic networks
class PPOActor:
    def __init__(self, in_state, in_actions, layers = [128, 128], continuous=False):
        self.continuous = continuous
        self.obs = in_state
        self.act = in_actions
        
        self.state_input = Input(shape=self.obs)
        self.advantage = Input(shape=(1, ))
        self.old_prediction = Input(shape=(self.act, ))

        x = Dense(layers[0], activation='tanh')(self.state_input)
        for l in layers[1:]:
            x = Dense(l, activation='tanh')(x)
        
        if(self.continuous):
            activation = 'tanh'
        else:
            activation = 'softmax'
        out_actions = Dense(self.act, activation=activation, name='output')(x)
        

        self.model = Model(inputs=[self.state_input, self.advantage, self.old_prediction], outputs = [out_actions])
        
    def initialize(self, lr, b1, b2, clip=0.2, c2=5e-3, noise=1.0):
        if(self.continuous):
            l = ppo_loss_continuous(
                advantage=self.advantage,
                old_prediction=self.old_prediction,
                noise=noise,
                clip=0.2,
                c2=5e-3
            )
        else:
            l = ppo_loss(
                advantage=self.advantage,
                old_prediction=self.old_prediction,
                clip=clip,
                c2=c2
            )
        self.model.compile(
            optimizer=Adam(learning_rate=lr, beta_1=b1, beta_2=b2), 
            loss = [l])

    def get_action(self, obs, val=False, noise=1.0):
        dummy_value = np.zeros((1,1))
        dummy_action = np.zeros((1, self.act))
        if(len(self.obs) == 1):
            p = self.model.predict([obs.reshape(1, self.obs[0]), dummy_value, dummy_action])
        else:
            p = self.model.predict([obs.reshape(1, *self.obs), dummy_value, dummy_action])
            
        if(self.continuous):
            if(val):
                action = action_matrix = p[0]
            else:
                action = action_matrix = p[0] + np.random.normal(loc=0, scale=noise, size=p[0].shape)
        else:
            if(val):
                action = np.argmax(p[0])
            else:
                action = np.random.choice(self.act, p=np.nan_to_num(p[0]))
            action_matrix = np.zeros(self.act)
            action_matrix[action] = 1
        return action, action_matrix, p

    def predict(self, st):
        return self.get_action(st, val=True)[0]

    def fit(self, obs, adv, old_pred, act, bs, epochs = 1):
        return self.model.fit(
            [obs, adv, old_pred], 
            act, 
            batch_size = bs,
            epochs = epochs,
            shuffle = True, 
            verbose = 0
        )


class PPOCritic:
    def __init__(self, in_state, layers=[128,128]):
        self.obs = in_state

        self.state_input = Input(shape=self.obs)

        x = Dense(layers[0], activation='tanh')(self.state_input)
        for l in layers[1:]:
            x = Dense(l, activation='tanh')(x)

        self.out = Dense(1)(x)
        self.model = Model(inputs=[self.state_input], outputs=[self.out])

    def initialize(self, lr, b1, b2, clip=0.2):
        self.model.compile(optimizer=Adam(learning_rate=lr, beta_1=b1, beta_2=b2, clipvalue=clip), loss='mse', metrics=['mae'])
        
    def predict(self, st):
        return self.model.predict(st)

    def fit(self, obs, rew, bs, epochs=1):
        return self.model.fit(
            [obs], 
            rew, 
            batch_size = bs,
            epochs = epochs, 
            shuffle = True,
            verbose = 0)


class DDPGCritic:
    def __init__(self, in_state, in_actions, layers=[128,128], reg=0.01):
        self.obs = in_state
        self.act = in_actions
        self.init = VarianceScaling()
        self.reg = l2(reg)
        self.state_input = Input(shape = self.obs)
        self.act_input = Input(shape = (self.act, ))

        st = BatchNormalization()(self.state_input)
        st = Dense(layers[0], activation=None, kernel_initializer=self.init, kernel_regularizer = self.reg)(st)
        st = ELU()(st)
        
        at = BatchNormalization()(self.act_input)
        at = Dense(layers[0], activation=None, kernel_initializer=self.init, kernel_regularizer = self.reg)(at)
        at = ELU()(at)
        

        x = concatenate([st, at], axis=1)
        for l in layers[1:]:
            x = Dense(l, activation=None, kernel_initializer=self.init, kernel_regularizer = self.reg)(x)
            x = ELU()(x)

        out = Dense(1, activation='linear')(x)
        

        self.model = Model(inputs =[self.state_input, self.act_input], outputs=[out])
        
    def initialize(self, lr, b1, b2):
        self.model.compile(
            optimizer=Adam(learning_rate=lr, beta_1=b1, beta_2=b2, clipvalue=0.5),
            loss='mse',
            metrics=['mae']
        )
    
    def batch_predict(self, st, at):
        return self.model.predict([st, at])
    
    def predict(self, st, at):
        return self.model.predict([[st], at])


    def transfer_weights(self, behavior_model, tau):
        self.model.set_weights(
            [((1-tau)*l1) + (tau*l2) for l1, l2 in zip(self.model.get_weights(), behavior_model.get_weights())])
    
    def fit(self, states, actions, y, epochs = 1):
        return self.model.fit([states, actions], y, verbose=0, epochs=epochs)

    def get_grads(self):
        return K.function(
            inputs = self.model.input,
            outputs = K.gradients(self.model.output, self.model.input[1])
        )

class DDPGActor:
    def __init__(self, in_state, in_actions, layers=[128, 128], reg=0.01, range_high=1, range_low=-1):
        self.obs = in_state
        self.act = in_actions
        self.init = VarianceScaling()
        self.reg = l2(reg)

        self.state_input = Input(shape = self.obs)
        x = BatchNormalization()(self.state_input)
        x = Dense(layers[0], activation=None, kernel_initializer = self.init, kernel_regularizer = self.reg)(x)        
        x = ELU()(x)
        

        for l in layers[1:]:
            x = Dense(l, activation=None, kernel_initializer = self.init, kernel_regularizer = self.reg)(x)
            x = ELU()(x)


        x = Dense(self.act, activation='tanh', use_bias=False)(x)        
        # move to action range
        out = Lambda(lambda i: range_high * i)(x)
        self.model = Model(inputs = [self.state_input], outputs=[out])

    def initialize(self, lr, b1, b2
    ):
        self.opt = Adam(lr, b1, b2, clipvalue=0.5)
        self.model.compile(
            optimizer=self.opt, 
            loss = 'mse'
        )
        

        act_grads = K.placeholder(shape=(None, self.act))
        mean_grad = K.mean(-act_grads, axis=0) #policy loss
        
        update_params = tf.gradients(self.model.output, self.model.trainable_weights, -act_grads)
        grads = zip(update_params, self.model.trainable_weights)
        
        return K.function(
            inputs=[self.model.input, act_grads], outputs=[mean_grad],
            updates=[self.opt.apply_gradients(grads)]
        )

    def predict(self, obs):
        return self.model.predict(np.expand_dims(obs, axis=0))[0]
    
    def batch_predict(self, obs):
        return self.model.predict(obs)

    def transfer_weights(self, behavior_model, tau):
        self.model.set_weights(
            [((1-tau)*l1) + (tau*l2) for l1, l2 in zip(self.model.get_weights(), behavior_model.get_weights())])
        


