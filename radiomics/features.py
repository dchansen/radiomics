from functools import reduce

import numpy as np
import pywt
import scipy
import scipy.stats
import skimage.measure
from scipy.spatial.distance import pdist


def _energy(img):
    return np.sum(img ** 2)


def _entropy(hist):
    return scipy.stats.entropy(hist)


def _kurtosis(img):
    return scipy.stats.kurtosis(img, fisher=False, axis=None)


def _maximum(img):
    return np.amax(img)


def _minimum(img):
    return np.amin(img)


def _mean(img):
    return np.mean(img)


def _mean_deviation(img):
    return np.mean(np.abs(np.mean(img) - img))


def _median(img):
    return np.median(img)


def _img_range(img):
    return np.amax(img) - np.amin(img)


def _rms(img):
    return np.sqrt(np.mean(img ** 2))


def _skewness(img):
    return scipy.stats.skew(img, axis=None)


def _std(img):
    return np.std(img)


def _uniformity(hist):
    return np.sum(hist ** 2)


def _variance(img):
    return np.var(img)


def group1_features(img):
    bins = np.arange(np.amin(img), np.amax(img), step=25)
    hist = np.histogram(img, bins=bins)[0]

    features = {"energy": _energy(img), "statistics_energy": _energy(hist), "kurtosis": _kurtosis(img),
                "maximum": _maximum(img), "minimum": _minimum(img),
                "mean": _mean(img), "mean_deviation": _mean_deviation(img), "median": _median(img),
                "img_range": _img_range(
                    img),
                "rms": _rms(img), "skewness": _skewness(img), "std": _std(img), "variance": _variance(img),
                "entropy": _entropy(hist), "uniformity": _uniformity(hist)}
    return features


def tumour_features(tumor_array, voxel_size):
    '''

    Args:
        tumor_array: Binary array containing the tumor shape
        voxel_size: Size of the voxels
    Returns: A dict containing the shape based features

    '''

    if np.sum(tumor_array) > 0:
        verts, faces, _, _ = skimage.measure.marching_cubes(tumor_array, 0.5, spacing=voxel_size)
        area = skimage.measure.mesh_surface_area(verts, faces)
        volume = np.sum(tumor_array > 0.1) * voxel_size[0] * voxel_size[1] * voxel_size[2]
        radius = (3.0 / (4.0 * np.pi) * volume) ** (1.0 / 3)

        distance = pdist(verts)

        features = {"compactness1": volume / (np.sqrt(np.pi) * area ** (2.0 / 3)),
                    "compactness2": 36 * np.pi * volume ** 2 / (area ** 3),
                    "maximum_diameter": np.amax(distance),
                    "spherical_disproportion": area / (4 * np.pi * radius ** 2),
                    "sphericity": np.pi ** (1.0 / 3) * (6 * volume) ** (2.0 / 3) / area,
                    "surface_area": area,
                    "surface_to_volume_ratio": area / volume,
                    "volume": volume

                    }

        return features
    else:
        return 0


def wavelet_features(img, mask):
    # pywt.swt(img,wavelet,level=1)
    s = img.shape

    img2 = np.pad(img,
                  pad_width=[(0, s[0] % 2), (0, s[1] % 2), (0, s[2] % 2)],
                  mode="constant", constant_values=0)
    mask2 = np.pad(mask,
                   pad_width=[(0, s[0] % 2), (0, s[1] % 2), (0, s[2] % 2)],
                   mode="constant", constant_values=0)

    wavelet_coefs = pywt.swtn(img2, 'coif1', level=1)[0]

    features = {}

    for key in wavelet_coefs:
        features1 = group1_features(wavelet_coefs[key])
        features31 = gray_level_cooccurrence_features(wavelet_coefs[key], mask2)
        features32 = gray_level_runlength_features(wavelet_coefs[key], mask2)

        for name in features1:
            features[key + "_" + name] = features1[name]
        for name in features31:
            features[key + "_" + name] = features31[name]
        for name in features32:
            features[key + "_" + name] = features32[name]

    return features


def bounding_box(img):
    '''
    Calculates the bounding box of a binary image
    Args:
        img: Binary image

    Returns: xmin,xmax, ymin,ymax,zmin,zmax

    '''

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def clip_to_bounding_box(img, mask):
    rmin, rmax, cmin, cmax, zmin, zmax = bounding_box(mask)

    img2 = img[rmin:rmax + 1, cmin:cmax + 1, zmin:zmax + 1]
    mask2 = mask[rmin:rmax + 1, cmin:cmax + 1, zmin:zmax + 1]
    return img2, mask2


def calculate_all_features(image, tumour_mask):
    '''

    Args:
        image: Input image
        tumour_mask: Binary image containing the GTV

    Returns: dict containing the features for the image

    '''

    img2 = np.array(image)
    img2 -= np.amin(image)
    img2, mask2 = clip_to_bounding_box(img2, tumour_mask)
    img2 *= mask2 > 0

    features = {}
    features.update(group1_features(img2[mask2 > 0]))
    features.update(tumour_features(mask2, [3, 1, 1]))
    features.update(gray_level_runlength_features(img2, mask2))
    features.update(gray_level_cooccurrence_features(img2, mask2))
    #
    features.update(wavelet_features(img2, mask2))

    return features


def super_features(img):
    if (not np.any(img[1, ...])):
        return [np.nan, np.nan, np.nan, np.nan]

    img2 = np.array(img)
    img2[0, ...] -= np.amin(img[0, ...])
    img2 = clip_to_bounding_box(img2)
    img2[0, ...] *= img2[1, ...] > 0

    feature1 = _energy(img2[0, ...])

    feature2 = tumour_features(img2[1, ...], [3, 1, 1])

    feature3 = gray_level_runlength_features(img2[0, ...])
    feature4 = gray_level_runlength_features(wavelet_features(img2[0, ...]))

    return [feature1, feature2, feature3, feature4]


def gray_level_runlength_features(img, mask):
    '''
    Create gral level runlength features
    Args:
        img:
        mask:

    Returns:

    '''
    bins = np.arange(np.nanmin(img), np.nanmax(img), step=25)

    bin_img = np.digitize(img, bins)
    bin_img[np.logical_not(mask)] = 0

    levels = np.arange(0, len(bins) + 1)
    glrl = np.array(_calculate_glrl(levels, bin_img), dtype=np.float64)

    glrl_sum = np.sum(glrl, axis=(0, 1))

    ix = np.arange(1, glrl.shape[0] + 1)[:, np.newaxis, np.newaxis]
    iy = np.arange(1, glrl.shape[1] + 1)[np.newaxis, :, np.newaxis]

    features = {"short_run_emphasis": np.mean(np.sum(glrl / iy ** 2, axis=(0, 1)) / glrl_sum),
                "long_run_emphasis": np.mean(np.sum(glrl * iy ** 2, axis=(0, 1)) / glrl_sum),
                "run_lenght_nonuniformity": np.mean(np.sum(np.sum(glrl, axis=0) ** 2, axis=0) / glrl_sum),
                "run_percentage": np.mean(np.sum(glrl / np.sum(img > 0))),
                "low_gray_level_run_emphasis": np.mean(np.sum(glrl / ix ** 2, axis=(0, 1)) / glrl_sum),
                "high_gray_level_run_emphasis": np.mean(np.sum(glrl * ix ** 2, axis=(0, 1)) / glrl_sum),
                "short_run_gray_level_emphasis": np.mean(np.sum(glrl / (ix ** 2 * iy ** 2), axis=(0, 1)) / glrl_sum),
                "short_run_high_gray_level_emphasis": np.mean(
                    np.sum(iy ** 2 * glrl / (ix ** 2), axis=(0, 1)) / glrl_sum),
                "long_run_low_gray_level_emphasis": np.mean(np.sum(ix ** 2 * glrl / (iy ** 2), axis=(0, 1)) / glrl_sum),
                "long_run_high_gray_level_emphasis": np.mean(np.sum(ix ** 2 * iy ** 2 * glrl, axis=(0, 1)) / glrl_sum)}

    return features

    return GLN


def _glcm_entroy(data, axis):
    dataN = data / np.sum(data, axis)  # normalize

    logdata = np.log2(dataN)
    logdata[dataN == 0] = 0
    return -np.sum(dataN * logdata, axis=axis)


def gray_level_cooccurrence_features(img, mask):
    bins = np.arange(np.amin(img), np.amax(img), step=25)

    bin_img = np.digitize(img, bins)

    matrix_index = np.where(mask > 0)
    # glcm = np.squeeze(calculate_glcm(np.arange(0,len(bins)),bin_img,matrix_index,[1]))
    glcm = _calculate_glcm2(bin_img, mask, bins.size)

    glcm = glcm / np.sum(glcm, axis=(0, 1))

    ix = np.array(np.arange(1, bins.size + 1)[:, np.newaxis, np.newaxis], dtype=np.float64)
    iy = np.array(np.arange(1, bins.size + 1)[np.newaxis, :, np.newaxis], dtype=np.float64)

    px = np.sum(glcm, axis=0)
    px = px[np.newaxis, ...]
    py = np.sum(glcm, axis=1)
    py = py[:, np.newaxis, :]

    ux = np.mean(glcm, axis=0)
    ux = ux[np.newaxis, ...]
    uy = np.mean(glcm, axis=1)
    uy = uy[:, np.newaxis, :]

    sigma_x = np.std(glcm, axis=0)
    sigma_x = sigma_x[np.newaxis, ...]
    sigma_y = np.std(glcm, axis=1)
    sigma_y = sigma_y[:, np.newaxis, :]

    hx = _glcm_entroy(px, (0, 1))
    hy = _glcm_entroy(py, (0, 1))

    h = _glcm_entroy(glcm, (0, 1))

    pxylog = np.log2(px * py)
    pxylog[(px * py) == 0] = 0
    hxy1 = -np.sum(glcm * pxylog, axis=(0, 1))

    hxy2 = _glcm_entroy(px * py, axis=(0, 1))

    pxyp = np.zeros((2 * bins.size, glcm.shape[2]))
    ki = ix + iy
    ki = ki[:, :, 0]

    for angle in range(glcm.shape[2]):
        for k in range(1, 2 * bins.size):
            glcm_view = glcm[..., angle]
            pxyp[k, angle] = np.sum(glcm_view[ki == (k + 1)])

    pxyplog = np.log2(pxyp)
    pxyplog[pxyp == 0] = 0

    pxym = np.zeros((bins.size, glcm.shape[2]))
    ki = np.abs(ix - iy)
    ki = ki[:, :, 0]
    for angle in range(glcm.shape[2]):
        for k in range(0, bins.size):
            glcm_view = glcm[..., angle]
            pxym[k, :] = np.sum(glcm_view[ki == k])

    pxymlog = np.log2(pxym)
    pxymlog[pxym == 0] = 0

    inverse_variance = 0

    for angle in range(glcm.shape[2]):
        glcm_view = glcm[:, :, angle]
        index = ix != iy
        diff = ix - iy
        diff = diff[..., 0]
        index = index[..., 0]
        inverse_variance += np.sum(glcm_view[index] / (diff[index]) ** 2)
    inverse_variance /= glcm.shape[2]

    sum_entropy = -np.sum(pxyp * pxyplog, axis=0)

    features = {"autocorrelation": np.mean(np.sum(ix * iy * glcm, axis=(0, 1))),
                "cluster_prominence": np.mean(np.sum((ix + iy - ux - uy) ** 4 * glcm, axis=(0, 1))),
                "cluster_shade": np.mean(np.sum((ix + iy - ux - uy) ** 3 * glcm, axis=(0, 1))),
                "cluster_tendency": np.mean(np.sum((ix + iy - ux - uy) ** 2 * glcm, axis=(0, 1))),
                "contrast": np.mean(np.sum((ix - iy) ** 2 * glcm, axis=(0, 1))),
                "correlation": np.mean(np.sum((ix * iy * glcm - ux * uy) / (sigma_x * sigma_y + 1e-6), axis=(0, 1))),
                "difference_entropy": np.mean(np.sum(pxym * pxymlog, axis=0)),
                "dissimilarity": np.mean(np.sum(np.abs(ix - iy) * glcm, axis=(0, 1))),
                "glcm_energy": np.mean(np.sum(glcm ** 2, axis=(0, 1))),
                "glcm_entropy": np.mean(h),
                "homogeneity1": np.mean(np.sum(glcm / (1 + np.abs(ix - iy)), axis=(0, 1))),
                "homogeneity1": np.mean(np.sum(glcm / (1 + (ix - iy) ** 2), axis=(0, 1))),
                "imc1": np.mean((h - hxy1) / np.maximum(hx, hy)),
                "imc2": np.mean(np.sqrt(1.0 - np.exp(-2 * (hxy2 - h)))),
                "idmn": np.mean(np.sum(glcm / (1.0 + ((ix - iy) ** 2) / (bins.size ** 2)), axis=(0, 1))),
                "idn": np.mean(np.sum(glcm / (1.0 + np.abs(ix - iy) / bins.size), axis=(0, 1))),
                "inverse_variance": inverse_variance,
                "maximum_probability": np.mean(np.amax(glcm, axis=(0, 1))),
                "sum_average": np.mean(np.sum(pxyp * np.arange(1, 2 * bins.size + 1)[:, np.newaxis], axis=0)),
                "sum_entropy": np.mean(sum_entropy),
                "sum_varianc": np.mean(
                    np.sum((np.arange(1, 2 * bins.size + 1)[:, np.newaxis] - sum_entropy[np.newaxis, :]) ** 2 * pxyp,
                           axis=0)),
                "variance": np.mean(np.sum(glcm * (ix - np.mean(glcm)) ** 2, axis=(0, 1)))
                }

    return features


def _calculate_glrl(grayLevels, matrix):
    # From https://github.com/vnarayan13/Slicer-OpenCAD/tree/master/HeterogeneityCAD
    padVal = 0  # use eps or NaN to pad matrix
    matrixDiagonals = list()

    # i.e.: self.heterogeneityFeatureWidgets = list(itertools.chain.from_iterable(self.featureWidgets.values()))

    # For a single direction or diagonal (aDiags, bDiags...lDiags, mDiags):
    # Generate a 1D array for each valid offset of the diagonal, a, in the range specified by lowBound and highBound
    # Convert each 1D array to a python list ( matrix.diagonal(a,,).tolist() )
    # Join lists using reduce(lamda x,y: x+y, ...) to represent all 1D arrays for the direction/diagonal
    # Use filter(lambda x: np.nonzero(x)[0].size>1, ....) to filter 1D arrays of size < 2 or value == 0 or padValue

    # Should change from nonzero() to filter for the padValue specifically (NaN, eps, etc)

    # (1,0,0), #(-1,0,0),
    aDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(matrix, (1, 2, 0))])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, aDiags))

    # (0,1,0), #(0,-1,0),
    bDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(matrix, (0, 2, 1))])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, bDiags))

    # (0,0,1), #(0,0,-1),
    cDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(matrix, (0, 1, 2))])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, cDiags))

    # (1,1,0),#(-1,-1,0),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[1]

    dDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 0, 1).tolist() for a in range(lowBound, highBound)])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, dDiags))

    # (1,0,1), #(-1,0-1),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[2]

    eDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 0, 2).tolist() for a in range(lowBound, highBound)])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, eDiags))

    # (0,1,1), #(0,-1,-1),
    lowBound = -matrix.shape[1] + 1
    highBound = matrix.shape[2]

    fDiags = reduce(lambda x, y: x + y, [matrix.diagonal(a, 1, 2).tolist() for a in range(lowBound, highBound)])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, fDiags))

    # (1,-1,0), #(-1,1,0),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[1]

    gDiags = reduce(lambda x, y: x + y,
                    [matrix[:, ::-1, :].diagonal(a, 0, 1).tolist() for a in range(lowBound, highBound)])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, gDiags))

    # (-1,0,1), #(1,0,-1),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[2]

    hDiags = reduce(lambda x, y: x + y,
                    [matrix[:, :, ::-1].diagonal(a, 0, 2).tolist() for a in range(lowBound, highBound)])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, hDiags))

    # (0,1,-1), #(0,-1,1),
    lowBound = -matrix.shape[1] + 1
    highBound = matrix.shape[2]

    iDiags = reduce(lambda x, y: x + y,
                    [matrix[:, :, ::-1].diagonal(a, 1, 2).tolist() for a in range(lowBound, highBound)])
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, iDiags))

    # (1,1,1), #(-1,-1,-1)
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[1]

    jDiags = [np.diagonal(h, x, 0, 1).tolist() for h in [matrix.diagonal(a, 0, 1) for a in range(lowBound, highBound)]
              for x in range(-h.shape[0] + 1, h.shape[1])]
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, jDiags))

    # (-1,1,-1), #(1,-1,1),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[1]

    kDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
              [matrix[:, ::-1, :].diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
              range(-h.shape[0] + 1, h.shape[1])]
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, kDiags))

    # (1,1,-1), #(-1,-1,1),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[1]

    lDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
              [matrix[:, :, ::-1].diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
              range(-h.shape[0] + 1, h.shape[1])]
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, lDiags))

    # (-1,1,1), #(1,-1,-1),
    lowBound = -matrix.shape[0] + 1
    highBound = matrix.shape[1]

    mDiags = [np.diagonal(h, x, 0, 1).tolist() for h in
              [matrix[:, ::-1, ::-1].diagonal(a, 0, 1) for a in range(lowBound, highBound)] for x in
              range(-h.shape[0] + 1, h.shape[1])]
    matrixDiagonals.append(filter(lambda x: np.nonzero(x)[0].size > 1, mDiags))

    # [n for n in mDiags if np.nonzero(n)[0].size>1] instead of filter(lambda x: np.nonzero(x)[0].size>1, mDiags)?


    P_out = np.zeros((len(grayLevels), np.max(matrix.shape), 13), dtype=np.int32)
    # Run-Length Encoding (rle) for the 13 list of diagonals (1 list per 3D direction/angle)
    for angle in range(0, len(matrixDiagonals)):
        P = P_out[:, :, angle]
        for diagonal in matrixDiagonals[angle]:
            diagonal = np.array(diagonal, dtype='int')
            pos, = np.where(np.diff(diagonal) != 0)  # can use instead of using map operator._ on np.where tuples
            pos = np.concatenate(([0], pos + 1, [len(diagonal)]))

            # a or pos[:-1] = run start #b or pos[1:] = run stop #diagonal[a] is matrix value
            # adjust condition for pos[:-1] != padVal = 0 to != padVal = eps or NaN or whatever pad value
            rle = zip([n for n in diagonal[pos[:-1]] if n != padVal], pos[1:] - pos[:-1])

            rle = [[np.where(grayLevels == x)[0][0], y - 1] for x, y in
                   rle]  # rle = map(lambda (x,y): [voxelToIndexDict[x],y-1], rle)

            # Increment GLRL matrix counter at coordinates defined by the run-length encoding
            P[list(zip(*rle))] += 1

    return P_out


def _calculate_glcm2(img, mask, nbins):
    out = np.zeros((nbins, nbins, 13))
    offsets = [(1, 0, 0),
               (0, 1, 0),
               (0, 0, 1),
               (1, 1, 0),
               (-1, 1, 0),
               (1, 0, 1),
               (-1, 0, 1),
               (0, 1, 1),
               (0, -1, 1),
               (1, 1, 1),
               (-1, 1, 1),
               (1, -1, 1),
               (1, 1, -1)
               ]
    matrix = np.array(img)
    matrix[mask <= 0] = nbins
    s = matrix.shape

    bins = np.arange(0, nbins + 1)

    for i, offset in enumerate(offsets):
        matrix1 = np.ravel(
            matrix[max(offset[0], 0):s[0] + min(offset[0], 0), max(offset[1], 0):s[1] + min(offset[1], 0),
            max(offset[2], 0):s[2] + min(offset[2], 0)])

        matrix2 = np.ravel(
            matrix[max(-offset[0], 0):s[0] + min(-offset[0], 0), max(-offset[1], 0):s[1] + min(-offset[1], 0),
            max(-offset[2], 0):s[2] + min(-offset[2], 0)])

        out[:, :, i] = np.histogram2d(matrix1, matrix2, bins=bins)[0]
    return out
