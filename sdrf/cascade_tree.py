#!/usr/bin/env python3

from typing import Any, Tuple, Sequence
import itertools
from functools import reduce
import math

import numpy as np
import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk
from scipy.spatial import KDTree
from jax.scipy.ndimage import map_coordinates

from .models import PyTorchBiasInitializer, SineInitializer
from util import get_fan


def gaussian(M, std, sym=True):
    r"""Return a Gaussian window.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    std : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).
    Notes
    -----
    The Gaussian window is defined as
    .. math::  w(n) = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }
    """
    if M == 1:
        return jnp.ones(1, "d")
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = jnp.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = jnp.exp(-(n ** 2) / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


@jax.custom_jvp
def exp_smin(a, b, k=32):
    """
    Order independent smooth union operation

    From https://iquilezles.org/www/articles/smin/smin.htm
    """
    res = jnp.exp2(-k * a) + jnp.exp2(-k * b)

    # return -jnp.log2(res) / k

    # required for numerical stability far from isosurface
    # doesn't matter for our particular case, as the coordinates
    # in each quadtree node are normalized to [0, 1] anyway
    #
    # also vmap(grad(exp_smin)) breaks a batching rule in JAX :/
    return jax.lax.cond(
        ((abs(res) < 1e-10) | (abs(res) > 1e10)).sum(),
        lambda r: jnp.minimum(a, b),
        lambda r: -jnp.log2(r) / k,
        res,
    )


@exp_smin.defjvp
def exp_smin_jvp(primals, tangents):
    """
    Derivative of the above exp_smin
    """
    a, b, k = primals
    da, db, _ = tangents

    res = jnp.exp2(-k * a) + jnp.exp2(-k * b)

    primal_out = -jnp.log2(res) / k

    tangent_exp_fn = lambda a, b: (jnp.exp2(k * b) * da + jnp.exp2(k * a) * db) / (
        jnp.exp2(k * a) + jnp.exp2(k * b)
    )
    tangent_exp = tangent_exp_fn(jnp.clip(a, a_max=3.0), jnp.clip(b, a_max=3.0))

    tangent_min = jax.jvp(jnp.minimum, (a, b), (da, db))[1]

    tangent_out = jnp.where(
        ((abs(res) < 1e-10) | (abs(res) > 1e10) | ~jnp.isfinite(res)),
        tangent_min,
        tangent_exp,
    )
    return (primal_out, tangent_out)


@jax.custom_jvp
def exp_smax(a, b, k=32):
    """
    Order independent smooth intersection operation

    From https://iquilezles.org/www/articles/smin/smin.htm
    """
    res = jnp.exp2(k * a) + jnp.exp2(k * b)

    # return -jnp.log2(res) / k

    # required for numerical stability far from isosurface
    # doesn't matter for our particular case, as the coordinates
    # in each quadtree node are normalized to [0, 1] anyway
    #
    # also vmap(grad(exp_smin)) breaks a batching rule in JAX :/
    return jax.lax.cond(
        ((abs(res) < 1e-10) | (abs(res) > 1e10)).sum(),
        lambda r: jnp.maximum(a, b),
        lambda r: jnp.log2(r) / k,
        res,
    )


@exp_smax.defjvp
def exp_smax_jvp(primals, tangents):
    """
    Derivative of the above exp_smax
    """
    a, b, k = primals
    da, db, _ = tangents

    res = jnp.exp2(k * a) + jnp.exp2(k * b)

    primal_out = jnp.log2(res) / k

    tangent_exp_fn = lambda a, b: (jnp.exp2(k * b) * da + jnp.exp2(k * a) * db) / (
        jnp.exp2(k * a) + jnp.exp2(k * b)
    )
    tangent_exp = tangent_exp_fn(jnp.clip(a, a_max=3.0), jnp.clip(b, a_max=3.0))

    tangent_max = jax.jvp(jnp.maximum, (a, b), (da, db))[1]

    tangent_out = jnp.where(
        ((abs(res) < 1e-10) | (abs(res) > 1e10) | ~jnp.isfinite(res)),
        tangent_max,
        tangent_exp,
    )
    return (primal_out, tangent_out)


def pow_smin(a, b, k=8):
    """
    Also from that article, doesn't work as well
    """
    a_k = jnp.power(a, k)
    b_k = jnp.power(b, k)

    return jnp.power((a_k * b_k) / (a_k + b_k), 1.0 / k)


def sphere_init(grid_min, grid_max, resolution, dtype=jnp.float32):
    dimension = grid_min.shape[0]
    ds = jnp.stack(
        jnp.meshgrid(
            *tuple(
                jnp.linspace(grid_min[di], grid_max[di], resolution, dtype=dtype)
                for di in range(dimension)
            )
        ),
        axis=-1,
    )

    sphere_sdf = lambda pt: jnp.linalg.norm(pt, keepdims=True) - 0.5
    return vmap(sphere_sdf)(ds.reshape(-1, dimension)).reshape(ds.shape[:-1] + (1,))


def variance_init(grid_min, grid_max, resolution, hidden_size, dtype=jnp.float32):
    dimension = grid_min.shape[0]
    ds = jnp.stack(
        jnp.meshgrid(
            *tuple(
                jnp.linspace(grid_min[di], grid_max[di], resolution, dtype=dtype)
                for di in range(dimension)
            )
        ),
        axis=-1,
    )

    w_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")
    b_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")

    # the weight and bias matrix for the first layer of a NeRF
    w_matrix = w_init((dimension, hidden_size), dtype=dtype)
    b_matrix = b_init((hidden_size,), dtype=dtype)

    # the equivalent of running the first layer of the network
    return ((ds.reshape(-1, dimension) @ w_matrix) + b_matrix).reshape(
        ds.shape[:-1] + (hidden_size,)
    )


def siren_init(grid_min, grid_max, resolution, hidden_size, dtype=jnp.float32):
    dimension = grid_min.shape[0]
    ds = jnp.stack(
        jnp.meshgrid(
            *tuple(
                jnp.linspace(grid_min[di], grid_max[di], resolution, dtype=dtype)
                for di in range(dimension)
            )
        ),
        axis=-1,
    )

    w_init = SineInitializer(True)
    b_init = PyTorchBiasInitializer()

    # the weight and bias matrix for the first layer of a NeRF
    w_matrix = w_init((dimension, hidden_size), dtype=dtype)
    b_matrix = b_init((hidden_size,), dtype=dtype)

    # the equivalent of running the first layer of the network
    return ((ds.reshape(-1, dimension) @ w_matrix) + b_matrix).reshape(
        ds.shape[:-1] + (hidden_size,)
    )


class PointCloudSDF(object):
    def __init__(self, point_cloud: np.ndarray, normals: np.ndarray):
        self.point_cloud = point_cloud
        self.normals = normals
        self.kd_tree = KDTree(self.point_cloud)

    def __call__(self, pts: np.ndarray):
        sdf, idx = self.kd_tree.query(pts, k=3)
        avg_normal = np.mean(self.normals[idx], axis=1)
        sdf = np.sum((pts - self.point_cloud[idx][:, 0]) * avg_normal, axis=-1)
        return sdf


class GeometricInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ..., 1]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        assert shape[-1] == 1
        resolution = shape[0]
        return sphere_init(self.grid_min, self.grid_max, resolution, dtype=dtype)


class RadianceInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ..., hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        resolution, hidden_size = shape[0], shape[-1]
        return variance_init(
            self.grid_min, self.grid_max, resolution, hidden_size, dtype=dtype
        )


class ConstantInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ..., hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        return jnp.ones(shape, dtype=dtype)


class ZeroInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ..., hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        return jnp.zeros(shape, dtype=dtype)


class SHInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ..., hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        s = shape[-1]
        assert s == 27
        rows = jnp.concatenate(
            [jnp.concatenate((jnp.ones(1), jnp.zeros(s // 3 - 1))) for _ in range(3)],
            axis=0,
        )
        return jnp.broadcast_to(rows, shape)


class SirenInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ..., hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        resolution, hidden_size = shape[0], shape[-1]
        return siren_init(
            self.grid_min, self.grid_max, resolution, hidden_size, dtype=dtype
        )


def downsample(base_mipmap, scale_factor) -> jnp.ndarray:
    resolution, dims = base_mipmap.shape[0], base_mipmap.ndim - 1

    # round down to the nearest odd kernel size
    kern_length = resolution
    if kern_length % 2 == 0:
        kern_length = kern_length - 1

    kern = gaussian(kern_length, std=scale_factor)
    # kern = gaussian(kern_length, std=1.0)
    kern = kern / kern.sum()
    off = kern_length // 2

    blurred = base_mipmap
    for i in range(dims):
        moved = jnp.moveaxis(blurred, i, -1)
        reshaped = jnp.reshape(moved, (-1, resolution))
        padded = jnp.pad(reshaped, ((0, 0), (off, off)), mode="reflect")

        evaluated = vmap(
            lambda reshaped_i: jnp.convolve(reshaped_i, kern, mode="valid")
        )(padded)

        blurred = jnp.moveaxis(jnp.reshape(evaluated, moved.shape), -1, i)

    return blurred


def n_dimensional_interpolation(cs, alpha):
    if alpha.shape[0] > 1:
        a = n_dimensional_interpolation(cs[0, ...], alpha[1:])
        b = n_dimensional_interpolation(cs[1, ...], alpha[1:])
    else:
        a = cs[0]
        b = cs[1]
    return (1 - alpha[0]) * a + alpha[0] * b


class FeatureGrid(hk.Module):
    def __init__(
        self,
        resolution,
        decoder_fn,
        grid_min=jnp.array([-1.0, -1.0]),
        grid_max=jnp.array([1.0, 1.0]),
        feature_size=128,
        feature_initializer_fn=GeometricInitializer,
        warp_field=None,
    ):
        super(FeatureGrid, self).__init__()
        self.dimensions = grid_min.shape[0]
        self.resolution = resolution
        self.dims = tuple(self.resolution for _ in range(self.dimensions))
        self.sample_locs = jnp.array(
            tuple(itertools.product(range(2), repeat=self.dimensions))
        )
        self.sample_locs = self.sample_locs.reshape((2,) * self.dimensions + (-1,))
        self.decoder_fn = decoder_fn

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.feature_size = feature_size
        self.warp_field = warp_field

        self.base_features = hk.get_parameter(
            "w",
            self.dims + (self.feature_size,),
            dtype=jnp.float32,
            init=feature_initializer_fn(self.grid_min, self.grid_max),
        )

    def sample(self, pt: jnp.ndarray, decoder_args=[]):
        alpha = (pt - self.grid_min) / (self.grid_max - self.grid_min)

        if self.warp_field is not None:
            alpha = alpha + self.warp_field(alpha)

        # lattice point interpolation, not grid
        idx_f = alpha * (jnp.array(self.dims).astype(jnp.float32) - 1)
        idx = idx_f.astype(jnp.int32)
        idx_alpha = jnp.modf(idx_f)[0]

        # TODO: is the clipping even needed anymore?
        xs = jnp.clip(idx + self.sample_locs, a_min=0, a_max=self.resolution)
        cs = vmap(lambda x: self.base_features[tuple(x)])(
            xs.reshape(-1, self.dimensions)
        ).reshape(xs.shape[:-1] + (self.feature_size,))
        pt_feature = n_dimensional_interpolation(cs, idx_alpha)

        """
        # extremely slow alternative that uses LAX-backed scipy function
        pt_feature = vmap(
            lambda pt_i: map_coordinates(
                self.base_features[..., pt_i],
                alpha * (jnp.array(self.dims).astype(jnp.float32) - 1),
                order=1,
                mode="nearest",
            )
        )(jnp.arange(self.feature_size))"""

        return self.decoder_fn(pt_feature, *decoder_args)

    def finite_difference(self):
        # must be scalar field
        assert self.base_features.shape[-1] == 1
        kernel = jnp.array(
            [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
        )

        off = kernel.shape[0] // 2

        def rec_conv(xs: jnp.ndarray):
            """
            xs is an array where the last dimension is the spatial dimension
            that we want to convolve; so we just vmap over all other axes and
            convolve that last one
            """
            if xs.ndim == 1:
                h = (self.grid_max[0] - self.grid_min[0]) / self.resolution
                padded = jnp.pad(xs, ((off, off),), mode="reflect")
                # TODO: fix for non-uniform grid dimensions?
                return jnp.correlate(padded, kernel, mode="valid") / h
            else:
                return vmap(rec_conv)(xs)

        # move axis to the end, then convolve with the kernel, then move back
        df = tuple(
            jnp.moveaxis(rec_conv(jnp.moveaxis(self.base_features, axis, -1)), -1, axis)
            for axis in range(self.base_features.ndim - 1)
        )
        return jnp.stack(df, axis=-1)
