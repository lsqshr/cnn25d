import numpy as np
from scipy.stats import gmean
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.advanced_activations import ELU
from matplotlib import pyplot as plt
from keras.models import load_model
from patch25d import *

from rivuletpy.utils.io import *
import argparse

from matplotlib import pyplot as plt


def _make_cnn(in_shape, binary=True, optimizer='rmsprop'):
    '''
    Make the CNN Model

    Parameters:
    in_shape: the numpy shape of the input matrix
    binary: True if the ground truth labels are binary
    '''

    model = Sequential()
    model.add(BatchNormalization(input_shape=in_shape[1:]))
    model.add(
        Convolution2D(
            128, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(GaussianDropout(0.25))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(GaussianDropout(0.25))

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
            y = y / y.max()

        x = self.normalize_feats(x)

        # Make the CNN model
        self._model = _make_cnn(x.shape, self._binary, optimizer)

        self._history = self._model.fit(x,
                                        y,
                                        batch_size=64,
                                        nb_epoch=epoch,
                                        validation_split=0.1,
                                        shuffle=True)
        self._model.save(model_path)

    def normalize_feats(self, x):
        # x = x - x.mean() / x.std()
        x = x - x.mean() / (x.max() - x.min())
        return x

    def predict(self, x):
        return self._model.predict(x)

    def im_predict(self, x, idx, shape):
        # x = x.astype('float32')
        im = np.zeros(shape)
        if x.ndim == 4:
            p = self.predict(x)
        elif x.ndim == 6:
            nsample, nrotate, nscale, kernelsz, _, _ = x.shape
            x = flatten_blocks(x, None)
            # x = self.normalize_feats(x)
            p = self.predict(x)
            p = p.reshape((nsample, nrotate * nscale))
            if p.shape[1] >= 2:
                p = p.mean(axis=-1)

            if self._binary:
                p = p.argmax(axis=-1)
                p = p > 0.5

        for i in tqdm(range(idx.shape[0])):
            im[math.floor(idx[i, 0]), math.floor(idx[i, 1]), math.floor(idx[i, 2])] = p[i]

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
        print('Loading model from h5path')
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
                        model_path='model.h5',
                        optimizer='rmsprop',
                        use_cache=False):
        '''
        Train 2.5D CNN from the 2.5D patches in a h5 file
        '''

        assert(isinstance(testidx, list))

        if use_cache:
            train_x, train_y = h5db.get_cached_train()
        else:
            nimg = h5db.get_im_num()
            trainidx = [i for i in range(nimg)]
            trainidx = [i for i in trainidx if i not in testidx]

            print(testidx)
            if any([t > nimg - 1 for t in testidx]) and (len(testidx) == 1 and
                                                         testidx[0] != -1):
                raise Exception('There are one or more test idx out of bound')

            train_x = []
            train_y = []
            for idx in trainidx:
                print('== Collect patches from image ', idx)
                x, y, _ = h5db.select_patches_from(idx, nsample_each, self._binary)
                train_x.append(x)
                train_y.append(y)

            print('== All the patches collected')

            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)

            print('nsample_each:%d\ttrain_x:%d' % (nsample_each, train_x.shape[0]))

            # Shuffle the indices
            random_idx = np.arange(train_y.shape[0])
            np.random.shuffle(random_idx)
            train_x = train_x[random_idx, :, :, :, :, :]
            train_y = np.squeeze(train_y[random_idx, :])
            h5db.cache_train(train_x, train_y)

        train_x = self.normalize_feats(train_x)
        self.train(train_x, train_y, nepoch, model_path, optimizer)

    def im_predict_from_h5(self,
                           h5db,
                           testidx,
                           model_path=None,
                           predicted_path=None):
        x, y, c = h5db.get_all_patches_from(testidx)
        shape = h5db.get_im_shape(testidx)

        # Load model from file if stated
        if model_path is not None:
            self.load_model_from_h5(model_path)
        im_predicted = self.im_predict(x, c, shape)
        return im_predicted

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
        '--predicted_path',
        type=str,
        default=None,
        required=False,
        help='The path to write predicted tiff. '
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

    parser.add_argument(
        '--test_idx',
        type=int,
        default=-1,
        help='''Which image is left for testing.
                Default 0.''')

    # Arguments for soma detection
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)

    # Arguments for soma detection
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)

    parser.add_argument('--no-binary', dest='binary', action='store_false')
    parser.set_defaults(binary=True)

    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.set_defaults(cache=False)

    args = parser.parse_args()

    cnn = Cnn25DH5(binary=args.binary)
    h5db = Patch25DB()
    h5db.connect(args.inh5, mode='r+')  # It might write the h5 file to cache the train data

    if args.train:
        cnn.simple_train_h5(
            h5db,
            [args.test_idx, ],
            nsample_each=args.nsample_each,
            nepoch=args.epoch,
            model_path=args.model_cache,
            optimizer='rmsprop',
            use_cache=args.cache)
        cnn.plot_history()

        # DEBUG
        im = cnn.im_predict_from_h5(h5db, 0,
                               model_path=None if args.train else args.model_cache)

        im2save = im.copy()
        im2save[im2save < 0] = 0
        im2save[im2save > 0.5] = 0.5
        im2save /= im2save.max()
        im2save *= 150
        writetiff3d('predicted.tif' if args.predicted_path is None else args.predicted_path,
                    im2save.astype('uint8'))


        f, ax = plt.subplots(1, 2)
        ax[0].imshow(im.max(-1))
        ax[0].set_title('predicted')
        ax[1].imshow(im.max(-1) > 0)
        ax[1].set_title('region')
        plt.show()

    if args.test:
        im = cnn.im_predict_from_h5(h5db, 0,
                               model_path=None if args.train else args.model_cache)

        im2save = im.copy()
        im2save[im2save < 0] = 0
        im2save[im2save > 0.5] = 0.5
        im2save /= im2save.max()
        im2save *= 150
        writetiff3d('predicted.tif' if args.predicted_path is None else args.predicted_path,
                    im2save.astype('uint8'))

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(im.max(-1))
        ax[0].set_title('predicted')
        ax[1].imshow(im.max(-1) > 0)
        ax[1].set_title('region')
        plt.show()
