#!/usr/bin/env python3

from typing import Any, Tuple, Sequence
import itertools
from functools import reduce

import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk

from util import get_fan


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
        # shape is [dim_x, dim_y, ..., hidden_size]
        assert all(shape[i] == shape[0] for i in range(len(shape) - 1))
        resolution = shape[0]
        return jax.nn.relu(
            sphere_init(
                self.grid_min, self.grid_max, resolution, shape[-1], dtype=dtype
            )
        )


def n_dimensional_interpolation(cs, alpha):
    if alpha.shape[0] > 1:
        a = n_dimensional_interpolation(cs[0, ...], alpha[1:])
        b = n_dimensional_interpolation(cs[1, ...], alpha[1:])
    else:
        a = cs[0]
        b = cs[1]
    return (1 - alpha[0]) * a + alpha[0] * b


class CascadeTree(hk.Module):
    def __init__(
        self,
        create_decoder_fn,
        grid_min=jnp.array([-1.0, -1.0]),
        grid_max=jnp.array([1.0, 1.0]),
        union_fn=exp_smin,
        max_depth=4,
        feature_size=128,
        feature_initializer_fn=sphere_init,
    ):
        super(CascadeTree, self).__init__()
        self.dimensions = grid_min.shape[0]
        self.dims = tuple(
            tuple(2 ** i for _ in range(self.dimensions)) for i in range(max_depth)
        )
        self.sample_locs = jnp.array(
            tuple(itertools.product(range(2), repeat=self.dimensions))
        )
        self.sample_locs = self.sample_locs.reshape((2,) * self.dimensions + (-1,))
        self.create_decoder_fn = create_decoder_fn

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.union_fn = union_fn
        self.max_depth = max_depth
        self.feature_size = feature_size
        self.feature_initializer_fn = feature_initializer_fn

    def __call__(self, pt: jnp.ndarray):
        fetch_grid = lambda s: self.feature_initializer_fn(
            self.grid_min, self.grid_max, s[0], self.feature_size
        )
        features = tuple(
            hk.get_parameter(
                f"w_{i}",
                self.dims[i] + (self.feature_size,),
                dtype=jnp.float32,
                init=LayerSphereInitializer(self.grid_min, self.grid_max),
            )
            for i, dim in enumerate(self.dims)
        )

        alpha = (pt - self.grid_min) / (self.grid_max - self.grid_min)

        decoder_fns = [self.create_decoder_fn() for i in range(self.max_depth)]

        def fetch_miplevel(level):
            idx_f = alpha * jnp.array(self.dims[level]).astype(jnp.float32)

            idx = idx_f.astype(jnp.int64)
            idx_alpha = jnp.modf(idx_f)[0]

            # TODO: Fix this to handle grids with different axis res
            xs = jnp.clip(idx + self.sample_locs, a_min=0, a_max=self.dims[level][0])
            cs = vmap(lambda x: features[level][tuple(x)])(
                xs.reshape(-1, self.dimensions)
            ).reshape(xs.shape[:-1] + (self.feature_size,))
            return n_dimensional_interpolation(cs, idx_alpha)

        depth = self.max_depth
        predecode_fns = [hk.Linear(self.feature_size) for i in range(depth)]

        # pt = pt * 2 - 1
        mip_levels = tuple(
            decoder_fns[i](
                fetch_miplevel(i) * 0.5 + predecode_fns[i](pt) * 0.5,
            )
            for i in range(2, depth)
        )
        samples = reduce(self.union_fn, mip_levels)
        # samples = mip_levels[-1]

        return samples