import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.ops import index_update
import functools

from .nerf_helpers import cumprod_exclusive


@functools.partial(jit, static_argnums=(4, 5))
def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    rng,
    radiance_field_noise_std,
    white_background,
):
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
    # disp_map =  1.0 / jnp.maximum(depth_map / acc_map, 1e-10)
    disp_map = 1.0 / jnp.maximum(depth_map / (jnp.maximum(acc_map, 1e-10)), 1e-10)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
