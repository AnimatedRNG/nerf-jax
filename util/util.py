import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.ops import index, index_update

def get_ray_bundle(height, width, focal_length, tfrom_cam2world):
    ii, jj = jnp.meshgrid(
        jnp.arange(width, dtype=jnp.float32,),
        jnp.arange(height, dtype=jnp.float32,),
        indexing="xy",
    )

    directions = jnp.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -jnp.ones_like(ii),
        ],
        axis=-1,
    )

    ray_directions = jnp.sum(
        directions[..., None, :] * tfrom_cam2world[:3, :3], axis=-1
    )
    ray_origins = jnp.broadcast_to(tfrom_cam2world[:3, -1], ray_directions.shape)
    return ray_origins, ray_directions


# @functools.partial(jit, static_argnums=(1, 2, 3))
def map_batched(tensor, f, chunksize, use_vmap):
    if tensor.shape[0] < chunksize:
        return f(tensor)
    else:
        tensor_diff = -tensor.shape[0] % chunksize
        initial_len = tensor.shape[0]
        tensor_len = tensor.shape[0] + tensor_diff
        tensor = jnp.pad(tensor, ((0, tensor_diff), (0, 0)), "constant")
        tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])

        if use_vmap:
            out = vmap(f)(tensor)  # this unfortunately keeps each batch in memory...
        else:
            out = jax.lax.map(f, tensor)

        out = out.reshape(-1, *out.shape[2:])[:initial_len]
        return out


# @functools.partial(jit, static_argnums=(1, 2, 3))
def map_batched_rng(tensor, f, chunksize, use_vmap, rng):
    if tensor.shape[0] < chunksize:
        key, subkey = jax.random.split(rng)
        return f((tensor, subkey)), key
    else:
        tensor_diff = -tensor.shape[0] % chunksize
        initial_len = tensor.shape[0]
        tensor_len = tensor.shape[0] + tensor_diff
        tensor = jnp.pad(tensor, ((0, tensor_diff), (0, 0)), "constant")
        tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])

        key, *subkey = jax.random.split(rng, tensor_len // chunksize + 1)
        subkey = jnp.stack(subkey)

        if use_vmap:
            out = vmap(f)((tensor, subkey))  # kinda gross imo
        else:
            out = jax.lax.map(f, (tensor, subkey))

        out = out.reshape(-1, *out.shape[2:])[:initial_len]
        return out, key
