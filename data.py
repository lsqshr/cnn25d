# %load data.py
import math
from tqdm import tqdm
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import morphological_gradient, binary_dilation
import skfmm
import h5py


class Image3D(object):
    def __init__(self, data=None):
        self._data = data

    def pad(self, margin):
        pimg = np.zeros(
            (self._data.shape[0] + 2 * margin,
             self._data.shape[1] + 2 * margin,
             self._data.shape[2] + 2 * margin))
        pimg[margin:margin + self._data.shape[0],
             margin:margin + self._data.shape[1],
             margin:margin + self._data.shape[2]] = self._data
        self._data = pimg

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

    def get(self, x, y, z):
        return self._data[math.floor(x), math.floor(y), math.floor(z)]

    def binarize(self, threshold):
        self._data[self._data < threshold] = 0

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
        self._binary = binary
        self._make_from_swc(swc, shape)

    def _make_from_swc(self, swc, shape):
        skimg = np.ones(shape)
        for i in range(swc.shape[0]):
            node = [math.floor(n) for n in swc[i, 2:5]]
            skimg[node[0], node[1], node[2]] = 0

        a, dm = 6, 5
        dt = skfmm.distance(skimg, dx=1)

        if self._binary:
            self._data = dt <= dm / 2
        else:
            zeromask = dt >= dm
            dt = np.exp(a * (1 - dt / dm)) - 1
            dt[zeromask] = 0
            self._data = dt

class BlockExtractor(object):
    def __init__(self,
                 augment=True,
                 nsample=4000,
                 K=7,
                 radii=(7, 11, 15),
                 nrotate=1):
        self._nsample = nsample
        self._K = K  # Block Radius
        self._radii = radii  # Radii to sample at each location
        self._nrotate = nrotate  # Number of random rotations to perform
        self._init_grids()

    def extract(self, bimg3d, distmap, save_path="blocks.h5"):
        '''
        Extract 2.5D blocks from a 3D Image with the ground truth
        in the distance map at the central voxel
        '''

        # Get all candidate positions
        candidate_idx = self._get_candidates(bimg3d)
        nsample = candidate_idx.shape[0]
        print('#Candidate:', nsample)
        nsample = self._nsample if nsample > self._nsample and self._nsample != -1 else nsample
        print('#Extract:', nsample)

        # Claim the memory for all 2.5D blocks
        # nblock = nsample * len(self._radii) * self._nrotate
        blocks = np.zeros(shape=(nsample, self._nrotate, len(self._radii),
                                 2 * self._K + 1, 2 * self._K + 1, 3))
        dist = np.zeros((nsample, 1))  # Claim the memory for 2.5D blocks

        # Start extracting blocks
        for i in tqdm(range(nsample)):
            bx, by, bz = candidate_idx[i, :]
            dist[i] = distmap.get(bx, by, bz)
            for r in range(self._nrotate):
                for s in range(len(self._radii)):
                    self._transform_grids(self._radii[s] / self._K,
                                          (np.random.rand() * 2 * np.pi,
                                           np.random.rand() * 2 * np.pi,
                                           np.random.rand() * 2 * np.pi),
                                          (bx, by, bz))

                    # Sample the 2.5D block with the current grids
                    blocks[i, r, s, :, :, :] = self._sample(bimg3d)
                    self._reset_grids()

        idx = candidate_idx[:nsample, :]    
        hf = h5py.File(save_path, 'w')
        hf.create_dataset('input/x', data=blocks)
        hf.create_dataset('input/y', data=dist)
        hf.create_dataset('input/idx', data=idx)

        return blocks, dist, idx

    def _sample(self, bimg3d):
        standard_grid = (np.arange(bimg3d.shape[0]),
                         np.arange(bimg3d.shape[1]),
                         np.arange(bimg3d.shape[2]))
        xy_pts = np.stack((self._grids[0][0].flatten(),
                           self._grids[0][1].flatten(),
                           self._grids[0][2].flatten())).T
        yz_pts = np.stack((self._grids[1][0].flatten(),
                           self._grids[1][1].flatten(),
                           self._grids[1][2].flatten())).T
        xz_pts = np.stack((self._grids[2][0].flatten(),
                           self._grids[2][1].flatten(),
                           self._grids[2][2].flatten())).T
        # print('Point shape:', xy_pts.shape)
        binterp = RegularGridInterpolator(standard_grid, bimg3d)

        bxy = binterp(xy_pts).reshape(2*self._K+1, 2*self._K+1)
        byz = binterp(yz_pts).reshape(2*self._K+1, 2*self._K+1)
        bxz = binterp(xz_pts).reshape(2*self._K+1, 2*self._K+1)

        return np.stack((bxy, byz, bxz), axis=-1)

    def _get_candidates(self, bimg3d):
        bimg = bimg3d > 0

        for i in range(3):
            bimg = binary_dilation(bimg)

        idx = np.argwhere(bimg)
        np.random.shuffle(idx)
        return idx

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
        # print('== grade shape:', self._grids[0][0].shape)
        # print('== Transform:', trm)
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
                xnew[i * nrotate + j, :, :, z * 3:z * 3 + 3] = x[i, j, z, :, :, :]

    if y is not None:
        return xnew, y
    else:
        return xnew
