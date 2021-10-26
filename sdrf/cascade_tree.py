#!/usr/bin/env python3

from typing import Any, Tuple, Sequence
import itertools
from functools import reduce
import math

import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk

from util import get_fan
from jax.experimental.host_callback import id_print


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


def sphere_init(grid_min, grid_max, resolution, hidden_size, dtype=jnp.float32):
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

    initializer = GeometricInitializer()((dimension, hidden_size), dtype=dtype)
    return (ds.reshape(-1, dimension) @ initializer).reshape(
        ds.shape[:-1] + (hidden_size,)
    )


class GeometricInitializer(hk.initializers.Initializer):
    def __init__(self, last_layer=False):
        self.last_layer = last_layer

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        fan_in, fan_out = get_fan(shape)

        if self.last_layer:
            mean = jnp.sqrt(np.pi) / jnp.sqrt(fan_in)
            stdv = 1e-5
            return hk.initializers.RandomNormal(stdv, mean)(shape, dtype)
        else:
            mean = 0.0
            stdv = jnp.sqrt(2.0) / jnp.sqrt(fan_out)
            return hk.initializers.RandomNormal(stdv, mean)(shape, dtype)


class LayerSphereInitializer(hk.initializers.Initializer):
    def __init__(self, grid_min, grid_max):
        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        # shape is [dim_x, dim_y, ...,hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        resolution = shape[0]
        return jax.nn.relu(
            sphere_init(
                self.grid_min, self.grid_max, resolution, shape[-1], dtype=dtype
            )
        )


def downsample_static(base_mipmap, scale_factor, kern_length) -> jnp.ndarray:
    shape = base_mipmap.shape
    ndim = len(shape) - 1
    resolution = shape[0]

    kern = gaussian(kern_length, std=scale_factor)
    kern = kern / kern.sum()

    blurred = jnp.array(base_mipmap)
    for dim in range(ndim):
        n_axis = [1 for _ in range(len(shape) - 1)]
        n_axis[dim] = kern_length

        # TODO(kazasrinivas3): not sure if this padding is actually right?
        # seems to get the top off by 1
        padding = [(0, 0) for _ in range(len(shape))]
        padding[dim] = (kern_length, kern_length)

        blurred = jnp.pad(blurred, padding, mode="reflect")

        kern_nd = jnp.resize(kern, (*n_axis, 1))
        """blurred = jax.scipy.signal.convolve(
            blurred, kern_nd, mode="same", method="direct"
        )"""
        conv_general_dilated_separable = lambda a, b: vmap(
            lambda a_: jax.lax.conv_general_dilated(
                a_[jnp.newaxis, jnp.newaxis, ...],
                b[jnp.newaxis, jnp.newaxis, ...],
                tuple(1 for _ in range(ndim)),
                "SAME",
            )
        )(a)
        blurred = jnp.moveaxis(
            conv_general_dilated_separable(
                jnp.moveaxis(blurred, -1, 0),
                kern_nd[..., -1],
            ),
            0,
            -1,
        )[0, 0]

        blurred = jnp.take(
            blurred,
            jnp.arange(kern_length, resolution + kern_length),
            axis=dim,
        )

    return blurred


def n_dimensional_interpolation(cs, alpha):
    if alpha.shape[0] > 1:
        a = n_dimensional_interpolation(cs[0, ...], alpha[1:])
        b = n_dimensional_interpolation(cs[1, ...], alpha[1:])
    else:
        a = cs[0]
        b = cs[1]
    return (1 - alpha[0]) * a + alpha[0] * b


class MipMap(hk.Module):
    def __init__(
        self,
        create_decoder_fn,
        resolution,
        scale_factor: float,
        kern_length: int,
        grid_min=jnp.array([-1.0, -1.0]),
        grid_max=jnp.array([1.0, 1.0]),
        feature_size=128,
        feature_initializer_fn=sphere_init,
    ):
        super(MipMap, self).__init__()
        self.dimensions = grid_min.shape[0]
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.kern_length = kern_length
        self.dims = tuple(self.resolution for _ in range(self.dimensions))
        self.sample_locs = jnp.array(
            tuple(itertools.product(range(2), repeat=self.dimensions))
        )
        self.sample_locs = self.sample_locs.reshape((2,) * self.dimensions + (-1,))
        self.create_decoder_fn = create_decoder_fn

        self.num_levels = int(math.log2(self.resolution))

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.feature_size = feature_size
        self.feature_initializer_fn = feature_initializer_fn

        self.base_features = hk.get_parameter(
            "w",
            self.dims + (self.feature_size,),
            dtype=jnp.float32,
            init=LayerSphereInitializer(self.grid_min, self.grid_max),
        )

    def __call__(
        self,
        pt: jnp.ndarray,
    ):
        alpha = (pt - self.grid_min) / (self.grid_max - self.grid_min)
        decoder_fn = self.create_decoder_fn()

        # does this actually work? I still saw nans unless I selected a minimum sigma
        feature_map = jax.lax.select(
            self.scale_factor > 0.5,
            downsample_static(self.base_features, self.scale_factor, self.kern_length),
            self.base_features,
        )

        # lattice point interpolation, not grid
        idx_f = alpha * (jnp.array(self.dims).astype(jnp.float32) - 1)
        idx = idx_f.astype(jnp.int32)
        idx_alpha = jnp.modf(idx_f)[0]

        # TODO: is the clipping even needed anymore?
        xs = jnp.clip(idx + self.sample_locs, a_min=0, a_max=self.resolution)
        cs = vmap(lambda x: feature_map[tuple(x)])(
            xs.reshape(-1, self.dimensions)
        ).reshape(xs.shape[:-1] + (self.feature_size,))
        pt_feature = n_dimensional_interpolation(cs, idx_alpha)

        return decoder_fn(pt_feature)


class CascadeTree(hk.Module):
    def __init__(self, highest_mipmap, union_fn=exp_smin, ignore_levels=2):
        super(CascadeTree, self).__init__()
        self.highest_mipmap = highest_mipmap
        self.union_fn = union_fn

        self.resolution = highest_mipmap.resolution
        self.mipmaps_res = [
            2 ** i
            for i in range(ignore_levels, math.floor(math.log2(self.resolution)) + 1)
        ]
        # Create downsampled mipmaps
        self.mipmaps = tuple(
            MipMap(
                self.highest_mipmap.create_decoder_fn,
                res,
                self.highest_mipmap.scale_factor * (res / self.resolution),
                self.highest_mipmap.kern_length,
                self.highest_mipmap.grid_min,
                self.highest_mipmap.grid_max,
                self.highest_mipmap.feature_size,
                self.highest_mipmap.feature_initializer_fn,
            )
            for res in self.mipmaps_res
        )

    def __call__(self, pt: jnp.ndarray):
        csg_levels = [mipmap(pt) for mipmap in self.mipmaps]

        samples = reduce(self.union_fn, csg_levels)
        # samples = mip_levels[-1]

        return samples, csg_levels
