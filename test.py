import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Seqeuntial
from tensorflow.keras.layers import Dense


m = Sequential([
Dense(32, input_shape=(784, ), activation='relu'), 
Dense(32, activation='relu'),
Dense(64),
Dense(1)])

m.compile('adam', 'mse')

x_train = np.random.rand(100000, 784)
y_train = np.random.rand(100000, 1)

m.fit(x_train, y_train, batch_size=64, epochs=10)

