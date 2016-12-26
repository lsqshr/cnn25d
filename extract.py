from data import *
# from train import *
from matplotlib import pyplot as plt
from rivuletpy.utils.io import *
# from keras.models import load_model
from scipy.ndimage.interpolation import zoom
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to perform the Rivulet2 tracing algorithm.')
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default=None,
        required=True,
        help='The input file. A image file (*.tif, *.nii, *.mat).')
    parser.add_argument(
        '-o',
        '--h5',
        type=str,
        default=None,
        required=True,
        help='The h5 file to write.')
    parser.add_argument(
        '-s',
        '--swc',
        type=str,
        default=None,
        required=True,
        help='The input swc file.'  )
    parser.add_argument(
        '-z',
        '--zoom_factor',
        type=float,
        default=1.,
        help='''The factor to zoom the image to speed up the whole thing.
                Default 1.'''
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.,
        help='Threshold used for segmenting the image. Default 0.')
    parser.add_argument(
        '-n',
        '--nsample',
        type=int,
        default=-1,
        help=''''Number of samples to extract.
              If nsample<0, extract as many as possible
              according to the threshold. Default -1'''
    )
    parser.add_argument(
        '--template',
        type=str,
        default=None,
        help='Template image to use for normalising the image.')

    args = parser.parse_args()
    print('After parse args')

    # NSAMPLE_EXTRACT = -1
    # NSAMPLE_TRAIN = -1
    # K = 7
    RADII = [7, 9, 11, 13, 15]
    # EPOCH = 200
    NROTATE = 3
    BINARY = True

    # template = '/home/siqi/hpc-data1/Data/Gold166-JSON/zebrafishadult/1.tif'
    fpath = args.file
    swcpath = args.swc

    if args.template:
        template_img = loadimg(args.template)

    print('Loading image files')
    imgvox = loadimg(fpath)
    swc = loadswc(swcpath)

    if args.zoom_factor != 1.:
        imgvox = zoom(imgvox, args.zoom_factor)
        swc[:, 2:5] *= args.zoom_factor

    # Pad Image
    img = Image3D(imgvox)
    img.pad(max(RADII) * 2)
    img.binarize(args.threshold)
    print('Making Distance Transform Map...')
    distmap = DistanceMap3D(swc, imgvox.shape, binary=BINARY)
    distmap.pad(max(RADII) * 2)

    db = BlockDB(20000)
    db.connect(h5file=args.h5)
    db.extract_image(
        os.path.split(args.file)[1], img, distmap,
        K, RADII, NROTATE, nsample=args.nsample)

# if os.path.exists('blocks.h5'):
# print('Load Blocks from blocks.h5...')
# h5f = h5py.File('blocks.h5', 'r')
# x = h5f['input/x'][:]
# y = h5f['input/y'][:]
# idx = h5f['input/idx'][:]
# h5f.close()
# else:
# print('Normalising Image...')
# img.gradient_based_normalise(template_img)
# print('Extracting Blocks...')
# extractor = BlockExtractor(
#     augment=True, nsample=NSAMPLE_EXTRACT, K=K, radii=RADII, nrotate=NROTATE)
# x, y, idx = extractor.extract(img.get_data(), distmap)

# NSAMPLE_EXTRACT = NSAMPLE_EXTRACT if NSAMPLE_EXTRACT > 0 else x.shape[0]

# x_flat, y_flat = flatten_blocks(x, y)

# assert(y.shape[0] == x.shape[0])

# if not os.path.exists('cache.h5'):
#     assert(y.shape[0] == x.shape[0])
#     NSAMPLE_TRAIN = NSAMPLE_TRAIN if NSAMPLE_TRAIN > 0 else x.shape[0]
#     x_flat = x_flat[:NSAMPLE_TRAIN, :, :, :]
#     y_flat = y_flat[:NSAMPLE_TRAIN]
#     learner = RivealLearner(x_flat.shape, BINARY)

#     if BINARY:
#         print(y_flat)
#         y_act = np.zeros((y_flat.shape[0], 2))
#         for i in range(y_flat.shape[0]):
#             y_act[i, math.floor(y_flat[i])] = 1
#         y_flat = y_act

#     learner.train(x_flat, y_flat, EPOCH)
#     learner.plot_history()
# else:
#     learner = RivealLearner(x_flat.shape, K)
#     learner.set_model(load_model('cache.h5'))

# # Predict
# im = learner.im_predict(x, idx, img.get_data().shape)
# f, ax = plt.subplots(1, 4)
# ax[0].imshow(img.get_data().max(axis=-1))
# ax[1].imshow(distmap.get_data().max(axis=-1))
# ax[2].imshow(im.max(axis=-1))
# ax[3].imshow((im > 0).max(axis=-1))
# plt.show()
