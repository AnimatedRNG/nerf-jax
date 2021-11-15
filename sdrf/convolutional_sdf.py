#!/usr/bin/env python3

import itertools

import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk

from . import LayerSphereInitializer, n_dimensional_interpolation


class ConvolutionalSDF(hk.Module):
    def __init__(
        self,
        resolution,
        dimension,
        feature_size,
        grid_min=jnp.array([-1.0, -1.0]),
        grid_max=jnp.array([1.0, 1.0]),
    ):
        super(ConvolutionalSDF, self).__init__()
        self.resolution = resolution
        self.dimension = dimension
        self.dims = (resolution,) * self.dimension
        self.feature_size = feature_size

        self.sample_locs = jnp.array(
            tuple(itertools.product(range(2), repeat=self.dimension))
        )
        self.sample_locs = self.sample_locs.reshape((2,) * self.dimension + (-1,))

        self.grid_min = grid_min
        self.grid_max = grid_max

    def __call__(self, pts: jnp.ndarray):
        all_channels = [(2 ** i) * self.feature_size for i in range(4)]
        extra_channels = [(2 ** i) * self.feature_size for i in range(4, 7)]

        x = hk.get_parameter(
            "feature_map",
            self.dims + (self.feature_size,),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(),
        )

        for channels in all_channels:
            x = hk.Conv2D(
                output_channels=channels,
                kernel_shape=4,
                stride=2,
                padding=lambda _: (1, 1),
            )(x)
            x = jax.nn.relu(x)

        for i, channels in enumerate(list(reversed(all_channels)) + extra_channels):
            x = hk.Conv2DTranspose(
                output_channels=channels,
                kernel_shape=4,
                stride=2,
                padding="SAME",
            )(x)
            x = jax.nn.relu(x) if i < len(all_channels) - 1 else x
        out_dims = tuple(x.shape[0] for i in range(self.dimension))
        print(f"out_dims are {out_dims}")

        feature_map = x

        def sample_pt(pt):
            alpha = (pt - self.grid_min) / (self.grid_max - self.grid_min)

            idx_f = alpha * (jnp.array(out_dims).astype(jnp.float32) - 1)
            idx = idx_f.astype(jnp.int32)
            idx_alpha = jnp.modf(idx_f)[0]

            # TODO: is the clipping even needed anymore?
            xs = idx + self.sample_locs
            cs = vmap(lambda x: feature_map[tuple(x)])(
                xs.reshape(-1, self.dimension)
            ).reshape(xs.shape[:-1] + (self.feature_size,))
            return n_dimensional_interpolation(cs, idx_alpha)

        return vmap(sample_pt)(pts), feature_map


'''def eikonal_loss(x: jnp.ndarray):
    """
    :param x is [x_dim, y_dim, z_dim] sdf grid
    """
    kernel = jnp.array(
        [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, -4 / 105, -1 / 280]
    )

    def rec_conv(xs: jnp.ndarray):
        """
        xs is an array where the last dimension is the spatial dimension
        that we want to convolve; so we just vmap over all other axes and
        convolve that last one
        """
        if xs.ndim == 1:
            return jnp.convolve(xs, kernel) / x.shape[0]
        else:
            return vmap(rec_convolve)(xs)

    # move axis to the end, then convolve with the kernel, then move back
    df_di = tuple(
        jnp.moveaxis(rec_conv(jnp.moveaxis(xs, axis, -1)), -1, axis)
        for axis in range(x.ndim)
    )

    return 1 - jnp.sqrt(sum(df_di) ** 2.0).ravel()'''
