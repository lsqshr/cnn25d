# %load data.py
import math
from tqdm import tqdm
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import morphological_gradient, binary_dilation
import skfmm
import h5py
# from threading import Thread, Semaphore
from multiprocessing import Process, Semaphore
import json


class Image3D(object):
    def __init__(self, data=None):
        self._data = data
        self._binary = None

    def pad(self, margin):
        pimg = np.zeros(
            (self._data.shape[0] + 2 * margin,
             self._data.shape[1] + 2 * margin,
             self._data.shape[2] + 2 * margin))
        pimg[margin:margin + self._data.shape[0],
             margin:margin + self._data.shape[1],
             margin:margin + self._data.shape[2]] = self._data

        self._data = pimg

        if self._binary is not None:
            pimg.fill(0)
            pimg[margin:margin + self._binary.shape[0],
                 margin:margin + self._binary.shape[1],
                 margin:margin + self._binary.shape[2]] = self._binary
            self._binary = pimg

    def unpad(self, margin):
        pimg = np.zeros((self._data.shape[0] - 2 * margin,
                         self._data.shape[1] - 2 * margin,
                         self._data.shape[2] - 2 * margin))
        pimg = self._data[margin:margin + self._data.shape[0],
                          margin:margin + self._data.shape[1],
                          margin:margin + self._data.shape[2]]
        self._data = pimg

    def get_data(self):
        return self._data

    def get_binary(self):
        return self._binary

    def get(self, x, y, z):
        return self._data[math.floor(x), math.floor(y), math.floor(z)]

    def binarize(self, threshold):
        self._binary = self._data.copy()
        self._binary[self._data < threshold] = 0

    def gradient_based_normalise(self, ref_img):
        '''
        Normalise the image intensity with Gradient based Histogram matching.
        F. PICCININI, E. LUCARELLI, A. GHERARDI, A. BEVILACQUA,
        Multi-image based method to correct vignetting effect in
        light microscopy images, Journal of Microscopy, 2012, 248, 1, 6
        '''
        if self._data is None:
            raise Exception('The image is still empty!')

        p_ag_input, gx = self._hist2d(self._data)
        p_ag_ref, _ = self._hist2d(ref_img)
        cinput = _cdf(p_ag_input)
        cref = _cdf(p_ag_ref)

        # Histogram Match Table
        bins = gx[:-1] * 255
        res = np.interp(self._data, bins, cinput)
        res = np.interp(res, cref, bins)

        # Logarithm Transform
        C3, M = 50, 255
        res = np.log(C3 + res)
        res = res - res.min()
        self._data = M * res / res.max()

    def _hist2d(self, img):
        '''
        Compute the Average Gradient Weighted histogram of the input image
        '''

        # Compute the bivriate histogram of input
        input_grad = morphological_gradient(img, size=(3, 3, 3))
        input_hist2d, gx, gy = np.histogram2d(
            img.flatten() / img.max(),
            input_grad.flatten() / input_grad.max(),
            bins=256)

        # Compute the AG profile of the input histogram
        p_ohm = input_hist2d.sum(axis=1)  # Ordinary Histogram

        # Gradient Weighted
        gy_tiled = np.tile(gy[:-1], (256, 1))
        p_gw = (gy_tiled * input_hist2d).sum(axis=1)

        # Average Gradient Weighted
        p_ag = p_gw / (p_ohm + 1)

        return p_ag, gx

    def _hist_match(self, h1, h2, bins):
        cdf1 = _cdf(h1)
        cdf2 = _cdf(h2)

        ctable = {}
        for c1, b1 in zip(cdf1, bins):
            ctable[np.floor(b1)] = np.floor(b1)
            for c2, b2 in zip(cdf2, bins):
                if np.floor(c1) == np.floor(c2):
                    ctable[np.floor(b1)] = np.floor(b2)
                    break
        return ctable


def _cdf(h):
    c = np.cumsum(h)
    return 255 * c / c[-1]


class DistanceMap3D(Image3D):
    def __init__(self, swc, shape, binary):
        Image3D.__init__(self)
        self._binary_label = binary
        self._make_from_swc(swc, shape)

    def _make_from_swc(self, swc, shape):
        skimg = np.ones(shape)
        for i in range(swc.shape[0]):
            node = [math.floor(n) for n in swc[i, 2:5]]
            skimg[node[0], node[1], node[2]] = 0

        a, dm = 6, 5
        dt = skfmm.distance(skimg, dx=1)

        if self._binary_label:
            self._data = dt <= dm / 2
        else:
            zeromask = dt >= dm
            dt = np.exp(a * (1 - dt / dm)) - 1
            dt[zeromask] = 0
            self._data = dt


class BlockExtractor(object):
    def __init__(self,
                 K=7,
                 radii=(7, 11, 15),
                 nrotate=1):
        self._K = K  # Block Radius
        self._kernelsz = 2 * K + 1
        self._radii = radii  # Radii to sample at each location
        self._nrotate = nrotate  # Number of random rotations to perform
        self._init_grids()

    def set_input(self, bimg3d, distmap):
        self._bimg3d = bimg3d
        self._distmap = distmap

    def set_candidates(self, candidates):
        self._candidates = candidates

    def run(self):
        '''
        Extract 2.5D blocks from a 3D Image with the ground truth
        in the distance map at the central voxel
        '''

        # Get all candidate positions
        nsample = self._candidates.shape[0]
        imgvox = self._bimg3d.get_data()

        # Claim the memory for all 2.5D blocks
        # nblock = nsample * len(self._radii) * self._nrotate
        self._blocks = np.zeros(shape=(nsample, self._nrotate,
                                       len(self._radii),
                                       2 * self._K + 1,
                                       2 * self._K + 1, 3))
        self._dist = np.zeros((nsample, 1))  # Claim the memory for 2.5D blocks

        self._standard_grid = (np.arange(imgvox.shape[0]),
                               np.arange(imgvox.shape[1]),
                               np.arange(imgvox.shape[2]))

        # Start extracting blocks
        for i in range(nsample):
            bx, by, bz = self._candidates[i, :]
            self._dist[i] = self._distmap.get(bx, by, bz)
            for r in range(self._nrotate):
                for s in range(len(self._radii)):
                    self._transform_grids(self._radii[s] / self._K,
                                          (np.random.rand() * 2 * np.pi,
                                           np.random.rand() * 2 * np.pi,
                                           np.random.rand() * 2 * np.pi),
                                          (bx, by, bz))

                    # Sample the 2.5D block with the current grids
                    self._blocks[i, r, s, :, :, :] = self._sample(imgvox)
                    self._reset_grids()

    def get_outputs(self):
        return self._blocks, self._dist

    def get_candidates(self):
        return self._candidates

    def _sample(self, imgvox):
        xy_pts = np.stack(
            (self._grids[0][0].flatten(), self._grids[0][1].flatten(),
             self._grids[0][2].flatten())).T
        yz_pts = np.stack(
            (self._grids[1][0].flatten(), self._grids[1][1].flatten(),
             self._grids[1][2].flatten())).T
        xz_pts = np.stack(
            (self._grids[2][0].flatten(), self._grids[2][1].flatten(),
             self._grids[2][2].flatten())).T
        binterp = RegularGridInterpolator(self._standard_grid, imgvox)
        return np.stack(
            (binterp(xy_pts).reshape(self._kernelsz, self._kernelsz),
             binterp(yz_pts).reshape(self._kernelsz, self._kernelsz),
             binterp(xz_pts).reshape(self._kernelsz, self._kernelsz)),
            axis=-1)

    def _init_grids(self):
        width = 2 * self._K + 1

        x = np.linspace(-self._K, self._K + 1, width)
        y = np.linspace(-self._K, self._K + 1, width)
        z = np.linspace(-self._K, self._K + 1, width)

        # Make Grid 1 on XY plane
        grid_xy_x, grid_xy_y, grid_xy_z = np.meshgrid(x, y, 0)

        # Make Grid 2 on YZ plane
        grid_yz_x, grid_yz_y, grid_yz_z = np.meshgrid(
            0, y, z)

        # Make Grid 3 on XZ plane
        grid_xz_x, grid_xz_y, grid_xz_z = np.meshgrid(x, 0, z)

        self._grids_backup = [[grid_xy_x, grid_xy_y, grid_xy_z],
                              [grid_yz_x, grid_yz_y, grid_yz_z],
                              [grid_xz_x, grid_xz_y, grid_xz_z]]
        self._grids = [[None, None, None], [None, None, None],
                       [None, None, None]]
        self._reset_grids()

    def _reset_grids(self):
        for i in range(3):
            for j in range(3):
                self._grids[i][j] = self._grids_backup[i][j].copy()

    def _transform_grids(self, scale_ratio, rotation, translation):
        rx, ry, rz, rs, rt = self._make_transform(scale_ratio,
                                                  rotation, translation)
        gridshape = self._grids[0][0].shape

        for i in range(3):
            g = np.stack(
                (self._grids[i][0].flatten(), self._grids[i][1].flatten(),
                 self._grids[i][2].flatten(),
                 np.ones(self._grids[0][0].size)))
            g = g.T.dot(rx).T
            g = g.T.dot(ry).T
            g = g.T.dot(rz).T
            g = g.T.dot(rs).T
            g = g.T.dot(rt).T
            self._grids[i][0] = g[0, :].reshape(gridshape)
            self._grids[i][1] = g[1, :].reshape(gridshape)
            self._grids[i][2] = g[2, :].reshape(gridshape)

    def _make_transform(self, scale_ratio, rotation, translation):
        angle_x, angle_y, angle_z = rotation
        tx, ty, tz = translation

        # Rotation Mat X
        rx = np.asarray(
            [[1, 0, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x), 0],
             [0, np.sin(angle_x), np.cos(angle_x), 0], [0, 0, 0, 1]])

        # Rotation Mat Y
        ry = np.asarray(
            [[np.cos(angle_y), 0, np.sin(angle_y), 0], [0, 1, 0, 0],
             [-np.sin(angle_y), 0, np.cos(angle_y), 0], [0, 0, 0, 1]])

        # Rotation Mat Z
        rz = np.asarray([[np.cos(angle_z), -np.sin(angle_z), 0, 0],
                         [np.sin(angle_z), np.cos(angle_z), 0, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]])

        # Scale Matrix
        rs = np.asarray([[scale_ratio, 0, 0, 0], [0, scale_ratio, 0, 0],
                         [0, 0, scale_ratio, 0], [0, 0, 0, 1]])

        # Translation Matrix
        rt = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                         [tx, ty, tz, 1]])

        return rx, ry, rz, rs, rt


class BlockDB(object):
    '''
    A Simple Database system to store&query the
    2.5 blocks of voxels using h5 files
    '''

    def __init__(self, extract_batch_size=1000):
        self._extract_batch_size = extract_batch_size

    def connect(self, h5file=None, mode='a'):
        print('-- Trying to open h5 file at %s' % h5file)
        self._db = h5py.File(h5file, "a")

    def extract_image(self,
                      img_name,
                      imgvox,
                      swc,
                      threshold=0,
                      K=7,
                      radii=[7, 9, 11],
                      nrotate=1,
                      nsample=-1,
                      template_img=None,
                      sema=None):

        if sema:
            sema.acquire()

        # Pad Image
        img = Image3D(imgvox)
        img.binarize(threshold)
        if template_img is not None:
            print('Normalising intensity...')
            img.gradient_based_normalise(template_img)
        img.pad(max(radii) * 2)

        print('Making Distance Transform Map...')
        distmap = DistanceMap3D(swc, imgvox.shape, binary=True)
        distmap.pad(max(radii) * 2)

        print('Extracting 2.5D blocks from %s' % img_name)
        # Calls BlockExtractor
        candidates = self._get_candidates(img)
        nsample_to_extract = candidates.shape[0] if candidates.shape[
            0] < nsample or nsample < 0 else nsample

        # Create a new group for this image
        img_grp = self._db.create_group(img_name)
        meta = img_grp.create_group('meta')

        meta.create_dataset('K', data=np.asarray(K).reshape(1, ))
        meta.create_dataset('radii', data=np.asarray(radii))
        meta.create_dataset('nrotate', data=np.asarray(nrotate).reshape(1, ))
        meta.create_dataset(
            'nsample', data=np.asarray(nsample_to_extract).reshape(1, ))
        data_grp = img_grp.create_group('data')
        data_grp.create_dataset('x', (nsample_to_extract, nrotate, len(radii),
                                      2 * K + 1, 2 * K + 1, 3))
        data_grp.create_dataset('y', (nsample_to_extract, 1))
        data_grp.create_dataset('c', (nsample_to_extract, 3))

        # Extract blocks batch by batch
        # batch_end = self._extract_batch_size
        batch_start = 0

        # Setup the tqdm bar
        pbar = tqdm(total=nsample_to_extract)
        e = BlockExtractor(  # In python, Thread can only be started once
            K=K, radii=radii, nrotate=nrotate)
        e.set_input(img, distmap)
        while True:
            batch_end = batch_start + self._extract_batch_size
            batch_end = batch_end if batch_end <= nsample_to_extract else nsample_to_extract
            batch_candidates = candidates[batch_start:batch_end, :]
            e.set_candidates(batch_candidates)
            e.run()
            batch_start += self._extract_batch_size  # Move to next batch

            x, y = e.get_outputs()
            c = e.get_candidates()

            # Append the blocks to hdf5
            data_grp['x'][batch_start:batch_end, :, :, :, :, :] = x
            data_grp['y'][batch_start:batch_end, :] = y
            data_grp['c'][batch_start:batch_end] = c
            pbar.update(self._extract_batch_size)

            if batch_start > nsample_to_extract:
                break

        if sema:
            sema.release()

    def extract_from_json(self,
                          json_file,
                          K,
                          radii,
                          nrotate,
                          nsample,
                          template_img,
                          nthread=1):
        d = json.load(open(json_file, 'r'))
        root = os.path.join(os.path.split(json_file)[0], d['rootpath'])
        process_pool = []
        sema = Semaphore(value=nthread)
        for dataset in d['data']:
            for imgname in d['data'][dataset]:

                img = d['data'][dataset][imgname]
                imgvox = loadimg(os.path.join(root, img['imagepath']))
                swc = loadswc(os.path.join(root, img['groundtruth']))
                e = Process(
                    name=img,
                    target=self.extract_image,
                    args=(imgname, imgvox, swc, img['misc']['threshold'],
                          K, radii, nrotate, nsample,
                          template_img, sema))
                process_pool.append(e)
                e.start()

        for e in process_pool:
            e.join()

    def _get_candidates(self, img3d):
        bimg = img3d.get_binary()
        bimg = bimg > 0

        for i in range(3):
            bimg = binary_dilation(bimg)

        idx = np.argwhere(bimg)
        np.random.shuffle(idx)
        return idx


def flatten_blocks(x, y=None):
    # Transform the blocks to nsample*NROTATION X K X K X 3*NSCALE
    nsample, nrotate, nscale, kernelsz, _, _ = x.shape
    xnew = np.zeros((nsample * nrotate, kernelsz, kernelsz, 3 * nscale))

    if y is not None:
        # Assign value y to each observation
        y = np.tile(y, (1, nrotate))
        y = y.reshape((nsample * nrotate, 1))

    for i in range(nsample):
        for j in range(nrotate):
            for z in range(nscale):
                xnew[i * nrotate + j, :, :, z * 3:z * 3 + 3] = x[i, j,
                                                                 z, :, :, :]

    if y is not None:
        return xnew, y
    else:
        return xnew


if __name__ == '__main__':
    import os
    from rivuletpy.utils.io import loadimg, loadswc
    from scipy.ndimage.interpolation import zoom
    import argparse

    parser = argparse.ArgumentParser(
        description='Arguments to perform the Rivulet2 tracing algorithm.')

    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default=None,
        required=True,
        help='The input file. A image file (*.tif, *.nii, *.mat). It can also be a json file.')

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
        required=False,
        help='The input swc file. Not used if --file is a json file' )

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

    parser.add_argument(
        '--thread',
        type=int,
        default=1,
        help="Number of threads to extract blocks. Default -1")

    args = parser.parse_args()
    K = 7
    RADII = [7, 9, 11, 13, 15]
    NROTATE = 3

    # Extract 2.5D Blocks
    db = BlockDB(1000)
    db.connect(h5file=args.h5)

    template_img = None
    if args.template:
        template_img = loadimg(args.template)

    if os.path.splitext(args.file)[1] == '.json':
        print('Loading from json file', args.file)
        db.extract_from_json(args.file, K, RADII, NROTATE, args.nsample,
                             template_img, args.thread)
    else:
        print('Loading image file', args.file)
        imgvox = loadimg(args.file)
        swc = loadswc(args.swc)

        if args.zoom_factor != 1.:
            imgvox = zoom(imgvox, args.zoom_factor)
            swc[:, 2:5] *= args.zoom_factor

        print('imgvox.shape:', imgvox.shape)
        print('imgvox.shape:', template_img.shape)
        db.extract_image(os.path.split(args.file)[1],
                         imgvox,
                         swc,
                         threshold=args.threshold,
                         K=K,
                         radii=RADII,
                         nrotate=1,
                         nsample=args.nsample,
                         template_img=template_img)
