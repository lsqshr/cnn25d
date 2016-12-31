import math
import numpy as np
from scipy.ndimage.interpolation import zoom
from patch25d import Image3D, DistanceMap3D
from rivuletpy.utils.io import loadswc, loadimg, writetiff3d
import argparse


def im_auto_chunk(img,
                  distmap,
                  labelmap,
                  nchunk=3,
                  size=[100, 100, 50],
                  prefix='chunk'):
    print('Working on: ', prefix)
    bimg = distmap._data > 0

    # Randomly choose a centre that is inside the central region
    central_region = np.zeros(bimg.shape)
    cx, cy, cz = [math.floor(s / 2 + 1) for s in bimg.shape]
    rx, ry, rz = [math.floor(s / 4) for s in bimg.shape]
    gx, gy, gz = np.meshgrid(
        np.arange(cx - rx, cx + rx + 1),
        np.arange(cy - ry, cy + ry + 1), np.arange(cz - rz, cz + rz + 1))
    central_region[gx.flatten(), gy.flatten(), gz.flatten()] = 1
    central_region = np.logical_and(central_region, bimg)
    idx = np.argwhere(bimg)
    idx2sample = idx[np.random.randint(0, idx.shape[0], (nchunk, )), :]

    size = np.asarray(size)
    pad_margin = size.max()

    if pad_margin > 0:
        img.pad(pad_margin)
        distmap.pad(pad_margin)
        labelmap.pad(pad_margin)

    for i, p in enumerate(idx2sample):
        centre = np.asarray(p)
        centre += pad_margin

        img_chunked, distmap_chunked, labelmap_chunked = im_chunk(
            img, distmap, labelmap, centre, size)

        # Save image to tiff
        print('Saving to', prefix + '.img.%d.tif' % i)
        writetiff3d(prefix + '.img.%d.tif' % i, img_chunked._data.astype('uint8'))
        np.save(prefix + '.dist.%d.npy' % i, distmap_chunked._data)
        np.save(prefix + '.label.%d.npy' % i, labelmap_chunked._data)


def im_chunk(img, distmap, labelmap, centre, size):

    img_chunked = img.copy()
    distmap_chunked = distmap.copy()
    labelmap_chunked = labelmap.copy()

    img_chunked.chunk(centre, size)
    distmap_chunked.chunk(centre, size)
    labelmap_chunked.chunk(centre, size)

    return img_chunked, distmap_chunked, labelmap_chunked


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to automatically chunk image according to its swc.')

    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default=None,
        required=True,
        help='The input file. A image file (*.tif, *.nii, *.mat). ')

    parser.add_argument(
        '--swc',
        type=str,
        default=None,
        required=False,
        help='The swc file to chunk. ')

    parser.add_argument(
        '-o',
        '--output_prefix',
        type=str,
        default=None,
        required=True,
        help='The prefix for the output tif and swc files.')

    parser.add_argument(
        '-c',
        '--centre',
        nargs='+',
        type=int,
        default=None,
        required=False,
        help='The centre of the chunk.')

    parser.add_argument(
        '-s',
        '--size',
        type=int,
        nargs='+',
        default=[100, 100, 50],
        help='''The size of the chunk.
                Default [100, 100, 50]''')

    parser.add_argument(
        '-z',
        '--zoom_factor',
        type=float,
        default=1.,
        help='''The factor to zoom the image to speed up the whole thing.
                Default 1.''')

    parser.add_argument(
        '-n',
        '--nchunk',
        type=int,
        default=3,
        help='Number of chunks')

    args = parser.parse_args()


    # Load and chunk the image
    imgvox = loadimg(args.file)
    swc = loadswc(args.swc)

    if args.zoom_factor != 1.:
        imgvox = zoom(imgvox, args.zoom_factor)

        if args.swc is not None:
            swc[:, 2:5] *= args.zoom_factor
    img = Image3D(imgvox)
    size = np.asarray(args.size)

    distmap = DistanceMap3D(swc, imgvox.shape, binary=False)
    labelmap = DistanceMap3D(swc, imgvox.shape, binary=True)

    if args.centre is not None:
        centre = np.asarray(args.centre)
        im_chunk(img, distmap, labelmap, centre, size)

        # Save image to tiff
        writetiff3d(prefix + '.img.tif', img._data.astype('uint8'))
        np.save(prefix + '.dist.npy', distmap._data)
        np.save(prefix + '.label.npy', labelmap._data)
    else:
        im_auto_chunk(img, distmap, labelmap, args.nchunk, args.size,
                      args.output_prefix)
