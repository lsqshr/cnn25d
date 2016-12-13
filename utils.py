
class BlockExtractor(object):

	def __init__(self, img3d, labelmap, augment=True, nsample=4000):
		self.augment = augment
		self.nsample = nsample
		self.img3d = img3d
		self.labelmap = labelmap

	def extract(img3d, labelmap, threshold=0):
	    '''	
        Extract 2.5D blocks from a 3D Image
	    '''

	    include_region = img3d > threshold
	    include_idx = np.argwhere(include_region)
	    nidx = include_idx.shape[0]

	    nsample = nidx if nsample > nidx else nsample

	    # Claim the memory for 2.5D blocks
	    x = np.zeros((nsample, 2 * K + 1, 2 * K + 1, 3))
	    y = np.zeros((nsample, 1))  # Claim the memory for 2.5D blocks

	    for i in range(idx2train.shape[0]):
	        bx, by, bz = idx2train[i, :]
	        x[i, :, :, 0] = img[bx - K:bx + K + 1, by - K:by + K + 1, bz]
	        x[i, :, :, 1] = img[bx - K:bx + K + 1, by, bz - K:bz + K + 1]
	        x[i, :, :, 2] = img[bx, by - K:by + K + 1, bz - K:bz + K + 1]
	        y[i] = dt[bx, by, bz]

	    return x, y

def gradient_based_normalise(input_img, ref_img):
	'''
	Normalise the image intensity with Gradient based Histogram matching.
    F. PICCININI, E. LUCARELLI, A. GHERARDI, A. BEVILACQUA, Multi-image based method to correct vignetting effect in light microscopy images, Journal of Microscopy, 2012, 248, 1, 6
	'''

	p_ag_input, gx = _hist2d(input_img)
	p_ag_ref, _ = _hist2d(ref_img)
	cinput = _cdf(p_ag_input)
	cref = _cdf(p_ag_ref)
    
    # # Histogram Match Table
	bins = gx[:-1] * 255
	res = np.interp(input_img, bins, cinput)
	res = np.interp(res, cref, bins)
    
    return im

def _hist2d(img):
	'''
	Compute the Average Gradient Weighted histogram of the input image
	'''

    # Compute the bivriate histogram of input
    input_grad = morphological_gradient(img, size=(3,3,3))
    input_hist2d, gx, gy = np.histogram2d(img.flatten()/img.max(), input_grad.flatten()/input_grad.max(), bins=256)

    # Compute the AG profile of the input histogram
    p_ohm = input_hist2d.sum(axis=1) # Ordinary Histogram

    # Gradient Weighted
    gy_tiled = np.tile(gy[:-1], (256,1))
    p_gw = (gy_tiled * input_hist2d).sum(axis=1)

    # Average Gradient Weighted
    p_ag = p_gw / (p_ohm+1)

    return p_ag, gx

def _cdf(h):
    c = np.cumsum(h)
    return 255 * c / c[-1]

def hist_match(h1, h2, bins):
    cdf1 = _cdf(h1)
    cdf2 = _cdf(h2)
    
    ctable = {}
    for c1, b1 in zip(cdf1, bins):
        ctable[np.floor(b1)] = np.floor(b1)
        for c2, b2 in zip(cdf2, bins):
            if np.floor(c1) == np.floor(c2):
                ctable[np.floor(b1)] = np.floor(b2)
                break;
    return ctable

