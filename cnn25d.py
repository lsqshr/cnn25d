import pickle
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.callbacks import TensorBoard
from patch25d import *
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from rivuletpy.utils.io import *
from nets import NetBuilder
import dtcwt


class Learner(object):
    def __init__(self, binary=False, model_path=None):
        self._binary = binary
        self._model_path = model_path

    def _normalize_feats(self, x):
        x = x - x.mean() / (x.max() - x.min())
        return x

    def _init_data(self, x, y):
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

        x = self._normalize_feats(x)

        return x, y

    def predict(self, x):
        # To be implemented in subclasses
        pass

    def im_predict(self, x, idx, shape):
        im = np.zeros(shape)
        if x.ndim == 4:
            p = self.predict(x)
        elif x.ndim == 6:
            nsample, nrotate, nscale, kernelsz, _, _ = x.shape
            x = flatten_blocks(x, None)
            # x = self._normalize_feats(x)
            p = self.predict(x)
            p = p.reshape((nsample, nrotate * nscale))
            if p.shape[1] >= 2:
                p = p.mean(axis=-1)

            if self._binary:
                p = p.argmax(axis=-1)
                p = p > 0.5

        for i in tqdm(range(idx.shape[0])):
            im[math.floor(idx[i, 0]),
               math.floor(idx[i, 1]),
               math.floor(idx[i, 2])] = p[i]

        return im, p


class Cnn25D(Learner):
    '''
    Simple 2.5D CNN with input and ground truth loaded in memory
    '''

    def __init__(self,
                 binary=False,
                 block_type='basic',
                 epoch=10,
                 model_path=None,
                 optimizer='rmsprop'):
        super(Cnn25D, self).__init__(binary, model_path)
        self._block_type = block_type  # Can be basic/residual
        self._epoch = epoch
        self._optimizer = optimizer

    def train(self, x, y):
        x, y = self._init_data(x, y)

        # Make the CNN model
        # self._model = _make_cnn(x.shape, self._binary, optimizer)
        builder = NetBuilder(
            'softmax' if self._binary else 'linear',
            2 if self._binary else 1,
            block_type=self._block_type,
            nb_row=3,
            nb_col=3,
            nb_filter=64,
            ndense=128,
            dropout=0.25)
        self._model = builder.build(x.shape[1:])
        self._model.compile(
            loss="categorical_crossentropy" if self._binary else "mse",
            optimizer=self._optimizer)
        tb = TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True)

        self._history = self._model.fit(x,
                                        y,
                                        batch_size=64,
                                        nb_epoch=self._epoch,
                                        validation_split=0.1,
                                        shuffle=True,
                                        callbacks=[tb])
        self._save_model()

    def save_model(self):
        self._model.save(self._model_path)

    def predict(self, x):
        return self._model.predict(x)

    def plot(self, fname='model.png'):
        from keras.utils.visualize_util import plot
        plot(self._model, to_file=fname)

    def load_model(self, modelpath):
        print('Loading model from h5path')
        self._model = load_model(modelpath)

    def plot_history(self, imgpath):
        # summarize history for loss
        plt.figure()
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper right')
        plt.savefig(imgpath)


class WaveletSVM3D(Learner):
    '''
    A learner for 3D blocks with 3D Wavelet features and SVR/SVM learner
    '''
    def __init__(self, binary, model_path='model.pkl', nlevels=2):
        super(WaveletSVM3D, self).__init__(binary, model_path)
        self._nlevels = nlevels

    def train(self, x, y):
        # This class only take care of 3D blocks
        assert (x.shape[-1] == x.shape[-2] and
                x.shape[-1] == x.shape[-3])
        x, y = self._init_data(x, y)

        print('Extracting 3D wavelet features...')
        x = self.wavelet3d_feats(x)
        self._model = SVR(C=1.0, epsilon=0.2, verbose=True)
        print('Fitting SVR...')
        self._model.fit(x, y)
        print('End Fitting SVR...')

    def predict(self, x):
        x = self.wavelet3d_feats(x)
        return self._model.predict(x)

    def wavelet3d_feats(self, x):
        wt = dtcwt.Transform3d()

        if x.shape[1] % 2 != 0:  # Cut one border for wavelet
            x = x[:, :-1, :-1, :-1]

        flat = []
        for i in tqdm(range(x.shape[0])):
            b = x[i, :, :, :]
            trans = wt.forward(b, nlevels=self._nlevels)
            for t in trans.highpasses:
                for l in range(t.shape[-1]):
                    C = t[:, :, :, l]
                    bad = C <= 0.2 * C.max()
                    C[bad] = 0
                    t[:, :, :, l] = C
            wvec = np.concatenate([t.flatten() for t in trans.highpasses])
            flat.append(wvec)

        x = np.stack(flat)
        return x

    def save_model(self):
        pickle.dump(self._model, self._model_path)

    def load_model(self, model_path):
        self._model = pickle.load(modelpath)


class LearnH5(object):
    '''
    Learner that takes a h5 file as input
    The h5 files contains the 2.5D patches sampled
    from different 3D images using patch25d
    '''

    def __init__(self, learner):
        self._learner = learner

    def simple_train_h5(self,
                        h5db,
                        testidx,
                        nsample_each=5000,
                        use_cache=False):
        '''
        Train 2.5D CNN from the 2.5D patches in a h5 file
        by loading all data into memory in front
        '''

        assert(isinstance(testidx, list))

        if use_cache:
            train_x, train_y = h5db.get_cached_train()
        else:
            nimg = h5db.get_im_num()
            trainidx = [i for i in range(nimg)]
            trainidx = [i for i in trainidx if i not in testidx]

            if any([t > nimg - 1 for t in testidx]) and (len(testidx) == 1 and
                                                         testidx[0] != -1):
                raise Exception('There are one or more test idx out of bound')

            train_x = []
            train_y = []
            for i, idx in enumerate(trainidx):
                print('== Collect patches from image %d/%d' % (i, len(trainidx)))
                x, y, _ = h5db.select_patches_from(idx, nsample_each, self._learner._binary)
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

        self._learner.train(train_x, train_y)

    def im_predict_from_h5(self,
                           h5db,
                           testidx,
                           model_path=None,
                           predicted_path=None,
                           metrics=True):
        x, y, c = h5db.get_all_patches_from(testidx)
        shape = h5db.get_im_shape(testidx)

        # Load model from file if stated
        if model_path is not None:
            self._learner.load_model(model_path)
        im, p = self._learner.im_predict(x, c, shape)

        # Try to write the metrics in the model cache
        cache = h5py.File(model_path, 'r+')
        metrics = {}
        metrics['evs'] = explained_variance_score(y, p)
        metrics['mae'] = mean_absolute_error(y, p)
        metrics['mse'] = mean_squared_error(y, p)
        metrics['r2'] = r2_score(y, p)

        for key in metrics:
            dpath = '/metrics/%d/%s' % (testidx, key)
            if dpath in cache:
                m = cache[dpath]
                m[()] = metrics[key]
            else:
                cache[dpath] = metrics[key]

        cache.close()

        return im, metrics

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
        '--model_type',
        type=str,
        default='seqcnn',
        required=False,
        help='The type of CNN to use (seqcnn/residual/wavelet_svm). Default seqcnn. '
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
        default='model',
        required=False,
        help='The prefix for the cached model, stats and history plot.')

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
        nargs='+',
        default=[-1,],
        help='''Which image is left for testing.
                Default 0.''')

    parser.add_argument(
        '--patch_type',
        type=str,
        default='25d',
        required=False,
        help='The type of extracted patch. Options are \'25d\', \'nov\' and \'3d\'. Default \'25d\'')

    # Arguments for soma detection
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)

    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)

    # Arguments for soma detection
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)

    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.set_defaults(binary=False)

    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.set_defaults(cache=False)

    args = parser.parse_args()

    if args.model_type in ('seqcnn', 'residual'):
        model = Cnn25D(
            args.binary,
            args.model_type,
            epoch=args.epoch,
            model_path=args.model_cache + '.h5',
            optimizer='rmsprop')
    elif args.model_type == 'wavelet_svm':
        model = WaveletSVM3D(
            args.binary,
            model_path=args.model_cache + '.pkl')

    lh5 = LearnH5(model)
    h5db = Patch25DB(patch_type=args.patch_type)
    # NOTE: It might write the h5 file to cache the train data
    h5db.connect(args.inh5, mode='r+')  

    if args.train:
        lh5.simple_train_h5(
            h5db,
            args.test_idx,
            nsample_each=args.nsample_each,
            use_cache=args.cache)

        # Save the training history to cache h5 for cnn models
        if args.model_type in ('seqcnn', 'residual'):
            lh5._learner.plot(args.model_cache+'.png')
            lh5._learner.plot_history(args.model_cache+'.eps')

            cache = h5py.File(args.model_cache+'.h5')
            if '/history/loss' in cache:
                loss = cache['/history/loss']
                loss[()] = lh5._learner._history.history['loss']
            else:
                cache['/history/loss'] = lh5._learner._history.history['loss']
            if '/history/val_loss' in cache:
                val_loss = cache['/history/val_loss']
                val_loss[()] = lh5._learner._history.history['val_loss']
            else:
                cache['/history/val_loss'] = lh5._learner._history.history['val_loss']
            cache.close()

    if args.test:
        for tidx in args.test_idx:
            im, metrics = lh5.im_predict_from_h5(
                h5db, tidx,
                model_path=args.model_cache+'.h5')

            print(metrics)

            im2save = im.copy()
            im2save[im2save < 0.25] = 0
            im2save /= im2save.max()
            im2save *= 200
            writetiff3d(args.model_cache + '.%d.tif' % tidx if args.predicted_path is None
                        else args.predicted_path + '.' + str(tidx) + '.tif',
                        im2save.astype('uint8'))

            if args.plot:
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(im.max(-1))
                ax[0].set_title('predicted')
                ax[1].imshow(im.max(-1) > 0)
                ax[1].set_title('region')

    if args.plot:
        plt.show()
