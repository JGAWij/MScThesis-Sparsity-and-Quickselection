from concrete_autoencoder import ConcreteAutoencoderFeatureSelector

from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = np.reshape(x_train, (len(x_train), -1))
#x_test = np.reshape(x_test, (len(x_test), -1))
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)



def decoder(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(784)(x)
    return x

selector = ConcreteAutoencoderFeatureSelector(K = 20, output_function = decoder, num_epochs = 800)

selector.fit(x_train, x_train, x_test, x_test)