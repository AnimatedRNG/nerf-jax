#!/usr/bin/env python3

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import functools

from nerf_helpers import cumprod_exclusive


@functools.partial(jit, static_argnums=(4, 5))
def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    rng,
    radiance_field_noise_std,
    white_background,
):
    """
    First, let's confirm that the jax sigmoid function works as expected

    >>> data = np.random.random((200,))
    >>> a = torch.sigmoid(torch.from_numpy(data))
    >>> b = jax.nn.sigmoid(jnp.array(data))
    >>> np.allclose(a.numpy(), np.array(b))
    True

    Now let's borrow a test from the pytorch implmentation...

    >>> rng = jax.random.PRNGKey(1010)

    >>> raw_np = np.random.uniform(size=(2, 2, 8, 4)).astype(np.float32)
    >>> rays_o_np = np.random.uniform(size=(2, 2, 3)).astype(np.float32)
    >>> z_vals_np = np.random.uniform(size=(8)).astype(np.float32)

    >>> raw_torch = torch.from_numpy(raw_np)
    >>> rays_o_torch = torch.from_numpy(rays_o_np)
    >>> z_vals_torch = torch.from_numpy(z_vals_np)

    >>> raw_jax = jnp.array(raw_np)
    >>> rays_o_jax = jnp.array(rays_o_np)
    >>> z_vals_jax = jnp.array(z_vals_np)

    >>> rgb_torch, disp_torch, acc_torch, weights_torch, depth_torch = volume_render_radiance_field_torch(
    ...     raw_torch, z_vals_torch, rays_o_torch)

    >>> rgb, disp, acc, weights, depth = volume_render_radiance_field(
    ...    raw_jax, z_vals_jax, rays_o_jax, rng, 0.0, False)

    >>> np.allclose(rgb_torch.numpy(), np.array(rgb))
    True
    >>> np.allclose(disp_torch.numpy(), np.array(disp))
    True
    """
    dists = jnp.concatenate(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            jnp.ones(depth_values[..., :1].shape, dtype=depth_values.dtype) * 1e10,
        ),
        axis=-1,
    )
    dists = dists * jnp.linalg.norm(ray_directions[..., None, :], ord=2, axis=-1)
    # dists = dists * jnp.sqrt((ray_directions[..., None, :] ** 2).sum(axis=-1))

    rgb = jax.nn.sigmoid(radiance_field[..., :3])
    if radiance_field_noise_std > 0.0:
        noise = (
            jax.random.normal(rng, radiance_field[..., 3].shape, jnp.float32,)
            * radiance_field_noise_std
        )
    else:
        noise = 0.0

    sigma_a = jax.nn.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - jnp.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(axis=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(axis=-1)

    acc_map = weights.sum(axis=-1)
    disp_map = 1.0 / jnp.maximum(depth_map / acc_map, 1e-10)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


if __name__ == "__main__":
    import doctest
    import torch
    from torch_impl import *

    print(doctest.testmod(exclude_empty=True))
