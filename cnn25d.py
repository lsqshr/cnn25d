import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.advanced_activations import SReLU
from data import flatten_blocks
from matplotlib import pyplot as plt


def _make_cnn(in_shape, binary=True, optimizer='rmsprop'):
    '''
    Make the CNN Model

    Parameters:
    in_shape: the numpy shape of the input matrix
    binary: True if the ground truth labels are binary
    '''

    model = Sequential()
    model.add(
        Convolution2D(
            32, 3, 3, border_mode='same', input_shape=in_shape[1:]))
    model.add(SReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.4))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(SReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.4))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(SReLU())

    if not binary:
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=optimizer)
    else:
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    return model


class Cnn25D(object):
    '''
    Simple 2.5D CNN with input and ground truth loaded in memory
    '''

    def __init__(self, in_shape, binary=False, optimizer='rmsprop'):
        self._binary = binary
        self.model = _make_cnn(in_shape, optimizer)

    def train(self, x, y, epoch, model_path='cache.h5'):
        x = x.astype('float32')
        y = y.astype('float32')

        self._normalize_feats(x)

        self._history = self._model.fit(x,
                                        y,
                                        batch_size=64,
                                        nb_epoch=epoch,
                                        validation_split=0.3,
                                        shuffle=True)
        self._model.save(model_path)

    def normalize_feats(self, x):
        x /= x.max()
        return x

    def predict(self, x):
        return self._model.predict(x)

    def im_predict(self, x, idx, shape):
        im = np.zeros(shape)
        if x.ndim == 4:
            p = self.predict(x)
        elif x.ndim == 6:
            nsample, nrotate, nscale, kernelsz, _, _ = x.shape
            x = flatten_blocks(x, None)

            p = self.predict(x)
            p = p.argmax(axis=-1)
            p = p.reshape(nsample, nrotate)
            p = p.mean(axis=-1)

            if self._binary:
                p = p > 0.5

        print('== pshape:', p.shape)
        print('== idx shape:', idx.shape)
        print('== img shape:', shape)

        for i in tqdm(range(idx.shape[0])):
            im[tuple(idx[i, :])] = p[i]

        return im

    def plot(self):
        from IPython.display import SVG, display
        from keras.utils.visualize_util import model_to_dot
        display(SVG(model_to_dot(self._model).create(prog='dot', format='svg')))
    
    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    def plot_history(self):
        # summarize history for loss
        plt.figure()
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper right')
        plt.savefig('history.eps')
