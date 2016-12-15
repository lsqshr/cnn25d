from data import *
from train import *
from matplotlib import pyplot as plt
from rivuletpy.utils.io import *
from keras.models import load_model

NSAMPLE_EXTRACT = -1
NSAMPLE_TRAIN = -1 
K = 7
RADII = [7, 9, 11, 13, 15]
EPOCH = 200 
NROTATE = 3
BINARY = True

template = '/home/siqi/hpc-data1/Data/Gold166-JSON/zebrafishadult/1.tif'
fpath = 'test.small.tif'
swcpath = 'test.small.swc'
imgvox = loadimg(fpath)
template_img = loadimg(template)
img = Image3D(imgvox)

# Pad Image
img.pad(40)
swc = loadswc(swcpath)
print('Making Distance Transform Map...')
distmap = DistanceMap3D(swc, imgvox.shape, binary=BINARY)
distmap.pad(40)

if os.path.exists('blocks.h5'):
    print('Load Blocks from blocks.h5...')
    h5f = h5py.File('blocks.h5', 'r')
    x = h5f['input/x'][:]
    y = h5f['input/y'][:]
    idx = h5f['input/idx'][:]
    h5f.close()
else:
    print('Normalising Image...')
    img.gradient_based_normalise(template_img)
    print('Extracting Blocks...')
    extractor = BlockExtractor(
        augment=True, nsample=NSAMPLE_EXTRACT, K=K, radii=RADII, nrotate=NROTATE)
    x, y, idx = extractor.extract(img.get_data(), distmap)

NSAMPLE_EXTRACT = NSAMPLE_EXTRACT if NSAMPLE_EXTRACT > 0 else x.shape[0]

x_flat, y_flat = flatten_blocks(x, y)

assert(y.shape[0] == x.shape[0])

if not os.path.exists('cache.h5'):
    assert(y.shape[0] == x.shape[0])
    NSAMPLE_TRAIN = NSAMPLE_TRAIN if NSAMPLE_TRAIN > 0 else x.shape[0]
    x_flat = x_flat[:NSAMPLE_TRAIN, :, :, :]
    y_flat = y_flat[:NSAMPLE_TRAIN]
    learner = RivealLearner(x_flat.shape, BINARY)

    if BINARY:
        print(y_flat)
        y_act = np.zeros((y_flat.shape[0], 2))
        for i in range(y_flat.shape[0]):
            y_act[i, math.floor(y_flat[i])] = 1
        y_flat = y_act

    learner.train(x_flat, y_flat, EPOCH)
    learner.plot_history()
else:
    learner = RivealLearner(x_flat.shape, K)
    learner.set_model(load_model('cache.h5'))

# Predict
im = learner.im_predict(x, idx, img.get_data().shape)
f, ax = plt.subplots(1, 4)
ax[0].imshow(img.get_data().max(axis=-1))
ax[1].imshow(distmap.get_data().max(axis=-1))
ax[2].imshow(im.max(axis=-1))
ax[3].imshow((im > 0).max(axis=-1))
plt.show()
