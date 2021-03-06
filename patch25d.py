import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import morphological_gradient, binary_dilation
import skfmm
import h5py
import multiprocessing as mp
from matplotlib import pyplot as plt


class Image3D(object):
    def __init__(self, data=None):
        self._data = data
        self._binary = None

    def copy(self):
        return Image3D(self._data.copy())

    def pad(self, margin):
        pimg = np.zeros((self._data.shape[0] + 2 * margin,
                         self._data.shape[1] + 2 * margin,
                         self._data.shape[2] + 2 * margin))
        pimg[margin:margin + self._data.shape[0], margin:margin + self._data.
             shape[1], margin:margin + self._data.shape[2]] = self._data

        self._data = pimg

        if self._binary is not None:
            pimg.fill(0)
            pimg[margin:margin + self._binary.shape[0], margin:margin +
                 self._binary.shape[1], margin:margin + self._binary.shape[
                     2]] = self._binary
            self._binary = pimg

    def unpad(self, margin):
        pimg = np.zeros((self._data.shape[0] - 2 * margin,
                         self._data.shape[1] - 2 * margin,
                         self._data.shape[2] - 2 * margin))
        pimg = self._data[margin:margin + self._data.shape[
            0], margin:margin + self._data.shape[1], margin:margin +
                          self._data.shape[2]]
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

    def chunk(self, centre, size):
        '''
        Chunk the image by given a 3D centre and a size of the chunk
        Assume the image has been zero padded
        '''
        cx, cy, cz = centre
        rx, ry, rz = [math.floor(s / 2) for s in size]
        self._data = self._data[cx - rx:cx + rx + 1, cy - ry:cy + ry + 1, cz -
                                rz:cz + rz + 1]


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

        # Add nodes the current swc to make sure there is
        # at least one node in each voxel on a branch
        idlist = swc[:, 0]
        extra_nodes = []
        for i in range(swc.shape[0]):
            cnode = swc[i, 2:5]
            pnode = swc[idlist == swc[i, 6], 2:5]
            dvec = pnode - cnode
            dvox = np.floor(np.linalg.norm(dvec))
            if dvox >= 1:
                uvec = dvec / (dvox + 1)
                extra_nodes.extend(
                    [cnode + uvec * i for i in range(1, int(dvox))])

        # Deal with nodes in swc
        for i in range(swc.shape[0]):
            node = [math.floor(n) for n in swc[i, 2:5]]
            skimg[node[0], node[1], node[2]] = 0

        # Deal with the extra nodes
        for ex in extra_nodes:
            node = [math.floor(n) for n in ex[0]]
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


class Patch25DExtractor(object):
    def __init__(self, K=7, radii=(7), nrotate=1):
        super(Patch25DExtractor, self).__init__()
        self._K = K  # Block Radius
        self._kernelsz = 2 * K + 1
        self._radii = radii  # Radii to sample at each location
        self._nrotate = nrotate  # Number of random rotations to perform
        self._init_grids()

    def set_input(self, bimg3d, distmap, labelmap):
        self._bimg3d = bimg3d
        self._distmap = distmap
        self._labelmap = labelmap

    def set_candidates(self, candidates):
        self._candidates = candidates

    def set_batch_bounds(self, start, end):
        self._batch_start = start
        self._batch_end = end

    def get_batch_bounds(self):
        return self._batch_start, self._batch_end

    def _init_blocks(self, nsample):
        self._blocks = np.zeros(shape=(nsample, max(self._nrotate, 1),
                                       len(self._radii), 2 * self._K + 1,
                                       2 * self._K + 1, len(self._grids)))

    def run(self):
        '''
        Extract 2.5D blocks from a 3D Image with the ground truth
        in the distance map at the central voxel
        '''

        # Get all candidate positions
        nsample = self._candidates.shape[0]
        imgvox = self._bimg3d.get_data()

        # Claim the memory for all 2.5D blocks
        self._init_blocks(nsample)
        self._dist, self._label = np.zeros((nsample, 1)), np.zeros((nsample, 1))

        self._standard_grid = (np.arange(imgvox.shape[0]),
                               np.arange(imgvox.shape[1]),
                               np.arange(imgvox.shape[2]))

        # The grid used below this are all flatten for speed
        base_flatten_grid = np.zeros(
            (len(self._grids), 4, self._grids[0][0].size))

        for i in range(len(self._grids)):
            base_flatten_grid[i, :, :] = np.stack(
                (self._grids[i][0].flatten(), self._grids[i][1].flatten(),
                 self._grids[i][2].flatten(), np.ones(self._grids[0][0].size)))

        # Extract the ground truth labels
        for i, (bx, by, bz) in enumerate(self._candidates):
            self._dist[i] = self._distmap.get(bx, by, bz)
            self._label[i] = self._labelmap.get(bx, by, bz)

        for s in range(len(self._radii)):
            # Scale transform
            scale_trns_grid = base_flatten_grid.copy()
            rs = self._make_scale_transform(self._radii[s] / self._K)
            scale_trns_grid = self._apply_transform(scale_trns_grid, rs)

            # Start extracting blocks
            for r in range(max(self._nrotate, 1)):
                # Rotation Transform
                rot_trns_grid = scale_trns_grid.copy()

                # Skip the rotation if nrotate is 0
                if self._nrotate != 0:
                    x_angle = np.random.rand() * 2 * np.pi
                    y_angle = np.random.rand() * 2 * np.pi
                    z_angle = np.random.rand() * 2 * np.pi
                    rx, ry, rz = self._make_rotation_transform(
                        x_angle, y_angle, z_angle)
                    rot_trns_grid = self._apply_transform(rot_trns_grid, rx)
                    rot_trns_grid = self._apply_transform(rot_trns_grid, ry)
                    rot_trns_grid = self._apply_transform(rot_trns_grid, rz)

                for i in range(nsample):
                    # Spatial Transform
                    bx, by, bz = self._candidates[i, :]
                    trans_trns_grid = rot_trns_grid.copy()
                    rt = self._make_translation_transform(bx, by, bz)
                    trans_trns_grid = self._apply_transform(trans_trns_grid,
                                                            rt)

                    # Sample the block with the current grids
                    self._blocks[i, r, s, :, :, :] = self._sample(
                        trans_trns_grid, imgvox)
        print('End of extraction')

    def get_outputs(self):
        return self._blocks, self._dist, self._label

    def get_candidates(self):
        return self._candidates

    def _sample(self, trns, imgvox):
        pts = [trns[i, :3, :].T for i in range(len(self._grids))]
        binterp = RegularGridInterpolator(self._standard_grid, imgvox)
        return np.stack(
            [binterp(p).reshape((self._kernelsz, self._kernelsz)) for p in pts],
            axis=-1)

    def _init_grids(self):
        width = 2 * self._K + 1

        x = np.linspace(-self._K, self._K + 1, width)
        y = np.linspace(-self._K, self._K + 1, width)
        z = np.linspace(-self._K, self._K + 1, width)

        # Make Grid 1 on XY plane
        grid_xy_x, grid_xy_y, grid_xy_z = np.meshgrid(x, y, 0)

        # Make Grid 2 on YZ plane
        grid_yz_x, grid_yz_y, grid_yz_z = np.meshgrid(0, y, z)

        # Make Grid 3 on XZ plane
        grid_xz_x, grid_xz_y, grid_xz_z = np.meshgrid(x, 0, z)

        self._grids_backup = [[grid_xy_x, grid_xy_y, grid_xy_z],
                              [grid_yz_x, grid_yz_y, grid_yz_z],
                              [grid_xz_x, grid_xz_y, grid_xz_z]]
        self._grids = self._grids_backup.copy()

    def plot_grids(self, grids, title):
        '''
        Plot the grids to debug the transformation code
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for g in grids:
            ax.scatter(g[0], g[1], g[2], s=5, c='r')
        plt.title(title)
        plt.show()

    def _apply_transform(self, flatten_grid, trns):
        result_grid = np.zeros((flatten_grid.shape))
        for i in range(result_grid.shape[0]):
            result_grid[i][:] = flatten_grid[i][:].T.dot(trns).T
        return result_grid

    def _make_sample_grid(self, trns):
        gridshape = self._grids[0][0].shape
        for i in range(3):
            self._grids[i][0] = trns[i, 0, :].reshape(gridshape)
            self._grids[i][1] = trns[i, 1, :].reshape(gridshape)
            self._grids[i][2] = trns[i, 2, :].reshape(gridshape)

    def _make_rotation_transform(self, angle_x, angle_y, angle_z):
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
        return rx, ry, rz

    def _make_scale_transform(self, scale_ratio):
        # Scale Matrix
        rs = np.asarray([[scale_ratio, 0, 0, 0], [0, scale_ratio, 0, 0],
                         [0, 0, scale_ratio, 0], [0, 0, 0, 1]])
        return rs

    def _make_translation_transform(self, tx, ty, tz):
        # Scale Matrix
        rt = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                         [tx, ty, tz, 1]])
        return rt


class NOV25DExtractor(Patch25DExtractor):
    '''
    Hex 2.5D Patch Extractor

    Each of TF observation consists of 9X2D patches (3 Sets).
    Each set of 2D patches are pi/2 away from each other 
    '''

    def __init__(self, K=7, radii=(7), nrotate=1):
        super(NOV25DExtractor, self).__init__(K, radii, nrotate)

    # Override _init_grids to make Hex 2.5D patches
    def _init_grids(self):
        width = 2 * self._K + 1

        x = np.linspace(-self._K, self._K + 1, width)
        y = np.linspace(-self._K, self._K + 1, width)
        z = np.linspace(-self._K, self._K + 1, width)

        # Make Horizontal Grid on XY plane
        grid_xy_x, grid_xy_y, grid_xy_z = np.meshgrid(x, y, 0)

        # Make Vertical on YZ plane
        grid_yz_x, grid_yz_y, grid_yz_z = np.meshgrid(0, y, z)

        # Make Vertical on XZ plane
        grid_xz_x, grid_xz_y, grid_xz_z = np.meshgrid(x, 0, z)

        # Rotate XY Grid pi/3 according to X axis
        grid_xy_x1, grid_xy_y1, grid_xy_z1 = self._rotate_2d_grid(
            grid_xy_x, grid_xy_y, grid_xy_z, np.pi / 4, 0., 0.)

        # Rotate XY Grid -pi/3 according to X axis
        grid_xy_x2, grid_xy_y2, grid_xy_z2 = self._rotate_2d_grid(
            grid_xy_x, grid_xy_y, grid_xy_z, -np.pi / 4, 0., 0.)

        # Rotate YZ Grid pi/3 according to Y axis
        grid_yz_x1, grid_yz_y1, grid_yz_z1 = self._rotate_2d_grid(
            grid_yz_x, grid_yz_y, grid_yz_z, 0., np.pi / 4, 0.)

        # Rotate YZ Grid -pi/3 according to Y axis
        grid_yz_x2, grid_yz_y2, grid_yz_z2 = self._rotate_2d_grid(
            grid_yz_x, grid_yz_y, grid_yz_z, 0., -np.pi / 4, 0.)

        # Rotate XZ grid pi/3 according to Z axis
        grid_xz_x1, grid_xz_y1, grid_xz_z1 = self._rotate_2d_grid(
            grid_xz_x, grid_xz_y, grid_xz_z, 0., 0., np.pi / 4)

        # Rotate XZ grid -pi/3 according to Z axis
        grid_xz_x2, grid_xz_y2, grid_xz_z2 = self._rotate_2d_grid(
            grid_xz_x, grid_xz_y, grid_xz_z, 0., 0., -np.pi / 4)

        self._grids_backup = [[grid_xy_x, grid_xy_y, grid_xy_z],
                              [grid_xy_x1, grid_xy_y1, grid_xy_z1],
                              [grid_xy_x2, grid_xy_y2, grid_xy_z2],
                              [grid_yz_x, grid_yz_y, grid_yz_z],
                              [grid_yz_x1, grid_yz_y1, grid_yz_z1],
                              [grid_yz_x2, grid_yz_y2, grid_yz_z2],
                              [grid_xz_x, grid_xz_y, grid_xz_z],
                              [grid_xz_x1, grid_xz_y1, grid_xz_z1],
                              [grid_xz_x2, grid_xz_y2, grid_xz_z2]]
        self._grids = self._grids_backup.copy()

    def _rotate_2d_grid(self, grid_x, grid_y, grid_z, angle_x, angle_y,
                        angle_z):
        rx, ry, rz = self._make_rotation_transform(angle_x, angle_y, angle_z)
        flatten_grid = np.stack((grid_x.flatten(), grid_y.flatten(),
                                 grid_z.flatten(), np.ones((grid_x.size, ))))
        flatten_grid = flatten_grid.T.dot(rx).T
        flatten_grid = flatten_grid.T.dot(ry).T
        flatten_grid = flatten_grid.T.dot(rz).T
        return flatten_grid[0, :].reshape(grid_x.shape), flatten_grid[
            1, :].reshape(grid_y.shape), flatten_grid[2, :].reshape(
                grid_z.shape)


class Patch3DExtractor(Patch25DExtractor):
    '''
    3D Patch Extractor

    Each of observation consists of 1 3D block.
    Each 3D patch
    '''
    def __init__(self, K=7, radii=(7), nrotate=1):
        super(Patch3DExtractor, self).__init__(K, radii, nrotate)

    # Override _init_grids to make Hex 2.5D patches
    def _init_grids(self):
        width = 2 * self._K + 1

        x = np.linspace(-self._K, self._K + 1, width)
        y = np.linspace(-self._K, self._K + 1, width)
        z = np.linspace(-self._K, self._K + 1, width)
        gridx, gridy, gridz = np.meshgrid(x, y, z)

        self._grids_backup = [[gridx, gridy, gridz], ]
        self._grids = self._grids_backup.copy()

    def _init_blocks(self, nsample):
        self._blocks = np.zeros(shape=(nsample, max(self._nrotate, 1),
                                       len(self._radii), 2 * self._K + 1,
                                       2 * self._K + 1, 2 * self._K + 1))

    def _sample(self, trns, imgvox):
        pts = trns[0, :3, :].T
        binterp = RegularGridInterpolator(self._standard_grid, imgvox)
        return binterp(pts).reshape(
            (self._kernelsz, self._kernelsz, self._kernelsz))


class Patch25DB(object):
    '''
    A Simple Database system to store&query the
    2.5 blocks of voxels using h5 files
    '''

    def __init__(self, extract_batch_size=1000, patch_type='25d'):
        self._extract_batch_size = extract_batch_size
        self._sema = mp.Semaphore(1)
        assert(patch_type in ('25d', 'nov', '3d'))
        self._patch_type = patch_type  # patch_type can be '25d'/'nov'/'3d'


    def connect(self, h5file=None, mode='a'):
        print('-- Trying to open h5 file at %s' % h5file)
        self._db = h5py.File(h5file, mode)
        self._h5file = h5file

    def disconnect(self):
        self._db.close()

    def im_extract(self,
                   img_name,
                   img=None,
                   labelmap=None,
                   distmap=None,
                   threshold=0,
                   K=7,
                   radii=[7, 9, 11],
                   nrotate=1,
                   nsample=-1,
                   template_img=None,
                   nthread=1,
                   sema=None):

        # Pad Image
        print('Extracting 2.5D blocks from %s' % img_name)

        candidates = self._get_candidates(img)

        nsample_to_extract = candidates.shape[0] if candidates.shape[
            0] < nsample or nsample < 0 else nsample

        # Create a new group for this image

        print('Creating dataset')
        img_grp = self._db.create_group(img_name)
        meta = img_grp.create_group('meta')

        meta.create_dataset('shape', data=np.asarray(img._data.shape))
        meta.create_dataset('K', data=np.asarray(K).reshape(1, ))
        meta.create_dataset('radii', data=np.asarray(radii))
        meta.create_dataset('nrotate', data=np.asarray(nrotate).reshape(1, ))
        meta.create_dataset(
            'nsample', data=np.asarray(nsample_to_extract).reshape(1, ))
        data_grp = img_grp.create_group('data')

        # Determine the depth of the block
        if self._patch_type == '25d':
            block_depth = 3
        elif self._patch_type == 'nov':
            block_depth = 9
        elif self._patch_type == '3d':
            block_depth = 2 * K + 1

        data_grp.create_dataset('x', (nsample_to_extract, max(nrotate, 1), len(radii),
                                      2 * K + 1, 2 * K + 1, block_depth))
        data_grp.create_dataset('dist', (nsample_to_extract, 1))
        data_grp.create_dataset('label', (nsample_to_extract, 1))
        data_grp.create_dataset('c', (nsample_to_extract, 3))
        self._db.close()  # Close for safe write
        print('Datasets created')

        task_queue = mp.JoinableQueue()

        procs = []
        # Start the Process Workers
        for i in range(nthread):
            # Make the process pool
            print('Starting Process %d' % i)
            p = mp.Process(
                name=str(i),
                target=self._extract_worker,
                args=(img_name, task_queue))
            p.daemon = True
            p.start()
            procs.append(p)

        # Put the extraction tasks into task queue
        batch_start = 0
        while True:
            batch_end = batch_start + self._extract_batch_size
            batch_end = batch_end if batch_end <= nsample_to_extract else nsample_to_extract
            batch_candidates = candidates[batch_start:batch_end, :]

            if self._patch_type == '25d':
                e = Patch25DExtractor(K=K, radii=radii, nrotate=nrotate)
            elif self._patch_type == 'nov':
                e = NOV25DExtractor(K=K, radii=radii, nrotate=nrotate)
            elif self._patch_type == '3d':
                e = Patch3DExtractor(K=K, radii=radii, nrotate=nrotate)

            e.set_input(img, distmap, labelmap)
            e.set_candidates(batch_candidates)
            e.set_batch_bounds(batch_start, batch_end)

            task_queue.put(e)

            batch_start += self._extract_batch_size

            if batch_start >= nsample_to_extract:
                break

        # print('Ready to join the task queue')
        task_queue.join()

        for p in procs:
            task_queue.put(None)

        task_queue.join()

        for p in procs:
            p.join()

    def _extract_worker(self, img_name, task_queue):
        for extractor in iter(task_queue.get, None):
            # Run the extraction
            batch_start, batch_end = extractor.get_batch_bounds()
            print('Working on ', batch_start, batch_end)
            extractor.run()
            print('Finished on ', batch_start, batch_end)
            x, dist, label = extractor.get_outputs()
            c = extractor.get_candidates()

            # Save blocks to h5
            self._sema.acquire()
            self.connect(self._h5file, 'a')
            self._db[img_name]['data']['x'][batch_start:
                                            batch_end, :, :, :, :, :] = x
            self._db[img_name]['data']['dist'][batch_start:batch_end, :] = dist
            self._db[img_name]['data']['label'][batch_start:
                                                batch_end, :] = label
            self._db[img_name]['data']['c'][batch_start:batch_end] = c
            task_queue.task_done()
            self.disconnect()
            self._sema.release()

        task_queue.task_done()

    def _get_candidates(self, img3d):
        bimg = img3d.get_binary()
        bimg = bimg > 0

        for i in range(3):
            bimg = binary_dilation(bimg)

        idx = np.argwhere(bimg)
        return idx

    def get_im_num(self):
        return len([k for k in self._db.keys() if k != 'cache'])

    def get_cached_train(self):
        return self._db['cache']['train_x'][()], self._db['cache']['train_y'][(
        )]

    def cache_train(self, train_x, train_y):
        if 'cache' not in self._db:
            self._db.create_group('cache')

        if 'train_x' in self._db['cache']:
            del self._db['cache/train_x']

        if 'train_y' in self._db['cache']:
            del self._db['cache/train_y']

        self._db['cache']['train_x'] = train_x
        self._db['cache']['train_y'] = train_y

    def select_patches_from(self, idx, nsample_each, binary=True):
        img_names = [k for k in self._db['/'].keys() if k != 'cache']
        x = self._db[img_names[idx]]['data']['x']

        if binary:
            y = self._db[img_names[idx]]['data']['label']
        else:
            y = self._db[img_names[idx]]['data']['dist']

        c = self._db[img_names[idx]]['data']['c']  # Total number of locations
        n, nrotate, nscale, kernelsz, _, _ = x.shape

        # Sample half with zeros
        y_np = np.squeeze(np.array(y))  # Convert y to numpy array
        zero_idx = np.argwhere(y_np == 0)
        nonzero_idx = np.argwhere(y_np > 0)
        np.random.shuffle(zero_idx)
        np.random.shuffle(nonzero_idx)

        if binary:
            nsample_each_cls = np.floor(nsample_each / 2)
            sample_idx = np.concatenate(
                (zero_idx[:nsample_each_cls if zero_idx.size > nsample_each_cls
                          else zero_idx.size],
                 nonzero_idx[:nsample_each_cls if zero_idx.size >
                             nsample_each_cls else zero_idx.size]))
        else:
            nsample_nonzero = np.floor(nsample_each * 1 / 4)
            nsample_zero = np.floor(nsample_each * 3 / 4)
            sample_idx = np.concatenate(
                (zero_idx[:nsample_zero
                          if zero_idx.size > nsample_zero else zero_idx.size],
                 nonzero_idx[:nsample_nonzero if zero_idx.size >
                             nsample_nonzero else zero_idx.size]))

        # np.random.shuffle(sample_idx)
        sample_idx = np.squeeze(sample_idx)
        nsample_each = sample_idx.size
        print('sample_idx', sample_idx.shape)

        # Determine the depth of the block
        if self._patch_type == '25d':
            block_depth = 3
        elif self._patch_type == 'nov':
            block_depth = 9
        elif self._patch_type == '3d':
            block_depth = kernelsz

        # Claim memory for the patches
        patches = np.zeros(
            (nsample_each, nrotate, nscale, kernelsz, kernelsz, block_depth))
        groundtruth = np.zeros((nsample_each, 1))
        coords = np.zeros((nsample_each, 3))

        # for i, idx in enumerate(tqdm(sample_idx)):
        sample_idx = np.sort(sample_idx)
        patch_idx = np.arange(len(sample_idx))
        patches[patch_idx, :, :, :, :] = x[sample_idx, :, :, :, :]
        groundtruth[patch_idx, :] = y[sample_idx, :]
        coords[patch_idx] = c[sample_idx, :]

        print('Nonzeros: %d/%d' %
              (np.count_nonzero(groundtruth), groundtruth.size))

        return patches, groundtruth, coords

    def get_all_patches_from(self, idx, binary=True):
        img_names = [k for k in self._db['/'].keys() if k != 'cache']
        x = self._db[img_names[idx]]['data']['x']
        c = self._db[img_names[idx]]['data']['c']  # Total number of locations

        if binary:
            y = self._db[img_names[idx]]['data']['label']
        else:
            y = self._db[img_names[idx]]['data']['dist']

        return x, y, c

    def get_im_shape(self, idx):
        img_names = [k for k in self._db['/'].keys() if k != 'cache']
        shape = self._db[img_names[idx]]['meta']['shape']
        return shape


def flatten_blocks(x, y=None):
    nsample, nrotate, nscale, kernelsz, _, depth = x.shape
    xnew = np.zeros((nsample * nrotate * nscale, kernelsz, kernelsz, depth))

    if y is not None:
        # Assign value y to each observation
        y = np.tile(y.reshape((y.size, 1)), (1, nrotate * nscale))
        y = y.reshape((nsample * nrotate * nscale, 1))

    for i in range(nsample):
        for j in range(nrotate):
            for z in range(nscale):
                xnew[i * nrotate + j * nscale + z, :, :, :] = x[i, j,
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
        description='Arguments to extract 2.5D/NOV2.5D/3D patches from 3D images.')

    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default=None,
        required=True,
        help='The input file. A image file (*.tif, *.nii, *.mat). ')

    parser.add_argument(
        '--distmap',
        type=str,
        default=None,
        required=False,
        help='The distance map file in .npy. ')

    parser.add_argument(
        '--labelmap',
        type=str,
        default=None,
        required=False,
        help='The label map file in .npy. ')

    parser.add_argument(
        '--patch_type',
        type=str,
        default='25d',
        required=False,
        help='The type of extracted patch. Options are \'25d\', \'nov\' and \'3d\'. Default \'25d\'')

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
        help='The input swc file. Not used if --file is a json file')

    parser.add_argument(
        '-z',
        '--zoom_factor',
        type=float,
        default=1.,
        help='''The factor to zoom the image to speed up the whole thing.
                Default 1.''')

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
              according to the threshold. Default -1''')

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

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="Size of the batch to write h5 file. Default 1000")

    parser.add_argument(
        '--nrotate',
        type=int,
        default=1,
        help="Size of the batch to write h5 file. Default 1")

    parser.add_argument(
        '--radii',
        type=int,
        nargs='+',
        default=[7],
        help="The radii to sample 2.5D patches. Default [5,7,9]")

    parser.add_argument(
        '-k',
        '--kernel_radius',
        type=int,
        default=7,
        help="The radius of the sampled patch. Default 7")

    args = parser.parse_args()

    # Extract 2.5D Blocks
    db = Patch25DB(args.batch_size, args.patch_type)
    db.connect(h5file=args.h5)

    template_img = None
    if args.template:
        template_img = loadimg(args.template)

    print('Loading image file', args.file)
    imgvox = loadimg(args.file)

    if args.swc is not None:
        swc = loadswc(args.swc)

    if args.zoom_factor != 1.:
        imgvox = zoom(imgvox, args.zoom_factor)

        if args.swc is not None:
            swc[:, 2:5] *= args.zoom_factor

    img = Image3D(imgvox)
    img.binarize(args.threshold)
    if template_img is not None:
        print('Normalising intensity...')
        img.gradient_based_normalise(template_img)
    img.pad(max(args.radii) * 3)

    print('Making Distance Transform Map...')
    if args.swc is None:

        if args.distmap is None or args.labelmap is None:
            raise Exception(
                'SWC file not provided, thus both distmap and labelmap should be provided in .npy files'
            )

        distmap = Image3D(np.load(args.distmap))
        labelmap = Image3D(np.load(args.labelmap))
    else:
        distmap = DistanceMap3D(swc, imgvox.shape, binary=False)
        labelmap = DistanceMap3D(swc, imgvox.shape, binary=True)

    distmap.pad(max(args.radii) * 3)
    labelmap.pad(max(args.radii) * 3)

    db.im_extract(
        os.path.split(args.file)[1],
        img,
        distmap,
        labelmap,
        threshold=args.threshold,
        K=args.kernel_radius,
        radii=args.radii,
        nrotate=args.nrotate,
        nsample=args.nsample,
        template_img=template_img,
        nthread=args.thread)
