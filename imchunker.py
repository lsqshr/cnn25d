import os
import math
import numpy as np
from patch25d import Image3D, DistanceMap3D
from rivuletpy.utils.io import *
import argparse


def im_auto_chunk(img,
                  distmap,
                  labelmap,
                  threshold=0,
                  nchunk=3,
                  size=[100, 100, 50],
                  prefix='chunk'):
    bimg = img._data > 0

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

    for i, p in enumerate(idx2sample):
        img, distmap, labelmap = im_chunk(img, distmap, labelmap, p, size)

        # Save image to tiff
        writetiff3d(prefix + '.img.%d.tif' % i, img._data.astype('uint8'))
        np.save(prefix + '.dist.%d.npy' % i, distmap._data)
        np.save(prefix + '.label.%d.npy' % i, labelmap._data)


def im_chunk(img, distmap, labelmap, centre, size):
    pad_margin = (np.floor(np.asarray(size) / 2) + np.asarray(centre) - np.asarray(imgvox.shape)).max()
    if pad_margin > 0:
        img.pad(pad_margin)
        distmap.pad(pad_margin)
        labelmap.pad(pad_margin)

    img.chunk(args.centre, args.size)
    distmap.chunk(args.centre, args.size)
    labelmap.chunk(args.centre, args.size)

    if pad_margin > 0:
        img.unpad(pad_margin)
        distmap.unpad(pad_margin)
        labelmap.unpad(pad_margin)


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
        '-t',
        '--threshold',
        type=int,
        default=0,
        help='Threshold to select candidate')

    parser.add_argument(
        '-n',
        '--nchunk',
        type=int,
        default=3,
        help='Number of chunks')

    args = parser.parse_args()

    # Load and chunk the image
    imgvox = loadimg(args.file)
    img = Image3D(imgvox)
    size = np.asarray(args.size)

    swc = loadswc(args.swc)
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
        im_auto_chunk(img, distmap, labelmap, args.threshold, args.nchunk, args.size,
                      args.output_prefix)
