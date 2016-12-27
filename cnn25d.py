import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.advanced_activations import SReLU
from matplotlib import pyplot as plt
from keras.models import load_model
from patch25d import *
import argparse


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

    def __init__(self, binary=False):
        self._binary = binary

    def train(self, x, y, epoch, model_path='cache.h5', optimizer='rmsprop'):
        x = x.astype('float32')
        y = y.astype('float32')

        x, y = flatten_blocks(x, y)

        if self._binary:
            y_act = np.zeros((y.shape[0], 2))
            for i, gidx in enumerate(y):
                y_act[i, math.floor(gidx)] = 1
            y = y_act
        else:
            y = self.normalize_feats(y)

        x = self.normalize_feats(x)

        # Make the CNN model
        self._model = _make_cnn(x.shape, optimizer)

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

    def load_model_from_h5(self, h5path):
        self._model = load_model(h5path)

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


class Cnn25DH5(Cnn25D):
    '''
    2.5D CNN that takes a h5 file as input
    The h5 files contains the 2.5D patches sampled
    from different 3D images using patch25d
    '''

    def __init__(self, binary=False):
        super(Cnn25DH5, self).__init__(binary)

    def simple_train_h5(self,
                        h5db,
                        testidx,
                        nsample_each=5000,
                        nepoch=10,
                        model_path='cache.h5',
                        optimizer='rmsprop'):
        '''
        Train 2.5D CNN from the 2.5D patches in a h5 file
        '''
        assert (isinstance(testidx, list))

        nimg = h5db.get_im_num()
        trainidx = [i for i in range(nimg)]
        trainidx = [i for i in trainidx if i not in testidx]

        if any([t > nimg - 1 for t in testidx]):
            raise Exception('There are one or more test idx out of bound')

        all_train_x = []
        all_train_y = []
        for idx in trainidx:
            x, y, _ = h5db.select_patches_from(idx, nsample_each, self._binary)
            all_train_x.append(x)
            all_train_y.append(y)

        all_train_x = np.concatenate(all_train_x, axis=0)
        all_train_y = np.concatenate(all_train_y, axis=0)

        # Shuffle the indices
        random_idx = np.arange(all_train_y.shape[0])
        np.random.shuffle(random_idx)
        all_train_x = np.squeeze(all_train_x[random_idx, :, :, :, :, :])
        all_train_y = np.squeeze(all_train_y[random_idx, :])

        self.train(all_train_x, all_train_y, nepoch, model_path, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to train 2.5D CNN.')

    parser.add_argument(
        '-i',
        '--inh5',
        type=str,
        default=None,
        required=True,
        help='The input file. A image file (*.tif, *.nii, *.mat). '
    )

    parser.add_argument(
        '-o',
        '--model_cache',
        type=str,
        default='model.h5',
        required=False,
        help='The h5 file to cache the trained model.')

    parser.add_argument(
        '-e',
        '--epoch',
        type=int,
        default=30,
        required=False,
        help='Number of epochs to train 2.5D CNN')

    parser.add_argument(
        '--nsample_each',
        type=int,
        default=5000,
        help='''Number of samples to draw from each image for training.
                Default 5000.''')

    # Arguments for soma detection
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=False)

    # Arguments for soma detection
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)

    # Arguments for soma detection
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--no-binary', dest='binary', action='store_false')
    parser.set_defaults(binary=True)

    args = parser.parse_args()

    cnn = Cnn25DH5(binary=args.binary)
    h5db = Patch25DB()
    h5db.connect(args.inh5, mode='r')
    cnn.simple_train_h5(
        h5db,
        [1,],
        nsample_each=args.nsample_each,
        nepoch=args.epoch,
        model_path='cache.h5',
        optimizer='rmsprop')
