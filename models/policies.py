import tensorflow

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

class MlpPolicy:
    def __init__(self, in_size,act_size):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape = (in_size, ), activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(act_size, activation='softmax'))

