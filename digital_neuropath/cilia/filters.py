"""Image filters and array helpers that do not know which marker they are looking at.
Filters on a volume, then statistics on its histogram, then helpers on its grid.
Every scipy and skimage call is direct, so the filter a step runs is the one on the page.
"""
from collections.abc import Sequence

import numpy as np
from scipy import linalg, ndimage
from skimage import feature, filters

__all__ = ["subtract_background", "smooth", "ridge", "robust_background", "local_background",
           "positive_percentile", "um_to_vox", "ellipsoid_footprint", "padded_bbox", "relabel"]


def um_to_vox(distance_um: float, spacing: tuple[float, float, float]
              ) -> tuple[float, float, float]:
    """Convert one physical distance to an unrounded voxel count per axis."""
    sz, sy, sx = spacing
    return (float(distance_um) / float(sz), float(distance_um) / float(sy),
            float(distance_um) / float(sx))


def ellipsoid_footprint(radius_um: float, spacing: tuple[float, float, float]) -> np.ndarray:
    """Return a boolean ellipsoid whose physical radius is the same on every axis."""
    # each semi-axis is at least one voxel: a zero one divides by zero in the inequality
    rz, ry, rx = (max(1, int(round(float(radius_um) / float(s)))) for s in spacing)
    zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return ((zz / rz) ** 2 + (yy / ry) ** 2 + (xx / rx) ** 2) <= 1.0


def padded_bbox(box: Sequence[slice], shape: Sequence[int],
                pad: Sequence[int]) -> tuple[slice, ...]:
    """Widen a box by a per-axis pad, clipped to the array shape."""
    return tuple(slice(max(0, int(axis.start) - int(p)), min(int(dim), int(axis.stop) + int(p)))
                 for axis, dim, p in zip(box, shape, pad, strict=True))


def relabel(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Drop the labels whose keep flag is false and renumber the survivors from 1."""
    flags = np.asarray(keep, dtype=bool).copy()
    flags[0] = False
    remap = np.zeros(flags.size, dtype=np.int32)
    remap[np.nonzero(flags)[0]] = np.arange(1, int(flags.sum()) + 1, dtype=np.int32)
    return remap[np.asarray(labels)]


def subtract_background(volume: np.ndarray, spacing: tuple[float, float, float],
                        sigma_um: float) -> np.ndarray:
    """Subtract a heavy gaussian estimate of the out-of-focus glow, clipped at zero."""
    array = np.asarray(volume)
    background = ndimage.gaussian_filter(array, sigma=um_to_vox(sigma_um, spacing))
    # every later percentile is over the positive part, so estimation error is not an intensity
    return np.clip(array - background, 0.0, None)


def smooth(volume: np.ndarray, spacing: tuple[float, float, float], sigma_um: float) -> np.ndarray:
    """Blur with a gaussian that is isotropic in micrometres."""
    return ndimage.gaussian_filter(np.asarray(volume), sigma=um_to_vox(sigma_um, spacing))


def ridge(volume: np.ndarray, sigma_vox: float | tuple[float, float, float], *, z_chunk: int,
          z_halo: int) -> np.ndarray:
    """Return the Meijering ridge response, computed in z bands with a halo and normalised once."""
    z_total = volume.shape[0]
    if z_chunk >= z_total and np.isscalar(sigma_vox):
        return filters.meijering(volume, sigmas=[sigma_vox],  # pyright: ignore[reportArgumentType]
                                 black_ridges=False)
    # meijering's own per-sigma body, band by band; its max normalisation is global, so it runs
    # once over the assembled volume rather than per band
    image = -np.asarray(volume, dtype=np.float32)
    alpha = 1.0 / (image.ndim + 1)
    mtx = linalg.circulant([1, *[alpha] * (image.ndim - 1)]).astype(image.dtype)
    raw = np.empty(volume.shape, dtype=np.float32)
    start = 0
    while start < z_total:
        end = min(start + z_chunk, z_total)
        pad_lo, pad_hi = min(z_halo, start), min(z_halo, z_total - end)
        band = image[start - pad_lo:end + pad_hi]
        hessian = feature.hessian_matrix(band, sigma_vox,  # pyright: ignore[reportArgumentType]
                                         mode="reflect", cval=0, use_gaussian_derivatives=True)
        vals = np.tensordot(mtx, feature.hessian_matrix_eigvals(hessian), 1)
        vals = np.take_along_axis(vals, np.abs(vals).argmax(0)[None], 0).squeeze(0)
        raw[start:end] = np.maximum(vals, 0)[pad_lo:pad_lo + (end - start)]
        start = end
    max_val = raw.max()
    if max_val > 0:
        raw /= max_val
    return raw


def robust_background(volume: np.ndarray) -> tuple[float, float]:
    """Return the per-image background as median and 1.4826 times the median absolute deviation."""
    array = np.asarray(volume)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    # a zero mad (a constant image, or an integer background on one value) falls back to the std
    sigma = 1.4826 * mad if mad > 0.0 else float(array.std())
    return median, sigma


def local_background(volume: np.ndarray, spacing: tuple[float, float, float], radius_um: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return per-voxel background median and robust σ from blocks of side 2·radius, linearly
    interpolated between block centres."""
    array = np.asarray(volume, dtype=np.float32)
    block = tuple(max(1, int(round(2.0 * float(radius_um) / float(s)))) for s in spacing)
    padded = np.pad(array, [(0, (-n) % b) for n, b in zip(array.shape, block, strict=True)],
                    mode="symmetric")
    (bz, by, bx), (gz, gy, gx) = block, tuple(n // b for n, b in zip(padded.shape, block,
                                                                       strict=True))
    level = np.empty((gz, gy, gx), dtype=np.float32)
    scale = np.empty((gz, gy, gx), dtype=np.float32)
    # one z row of blocks at a time, so memory stays one slab rather than a copy of the frame
    for i in range(gz):
        cells = (padded[i * bz:(i + 1) * bz].reshape(bz, gy, by, gx, bx)
                 .transpose(1, 3, 0, 2, 4).reshape(gy, gx, -1))
        median = np.median(cells, axis=-1)
        mad = np.median(np.abs(cells - median[..., None]), axis=-1)
        level[i] = median
        scale[i] = np.where(mad > 0, 1.4826 * mad, cells.std(axis=-1))
    window = tuple(slice(0, n) for n in array.shape)
    return (np.asarray(ndimage.zoom(level, block, order=1, mode="nearest", grid_mode=True))[window],
            np.asarray(ndimage.zoom(scale, block, order=1, mode="nearest", grid_mode=True))[window])


def positive_percentile(volume: np.ndarray, percentile: float) -> float:
    """Return a percentile of the positive voxels, or infinity when none is positive."""
    array = np.asarray(volume)
    positive = array[array > 0]
    # a blank image must select nothing; a cut-off of 0 would select the whole frame
    return float(np.percentile(positive, percentile)) if positive.size else float("inf")
