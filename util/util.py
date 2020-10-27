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
        return vmap(f)(tensor) if use_vmap else f(tensor)
    else:
        tensor_diff = -tensor.shape[0] % chunksize
        initial_len = tensor.shape[0]
        tensor_len = tensor.shape[0] + tensor_diff
        extra_dims = tuple((0, 0) for _ in range(len(tensor.shape) - 1))
        tensor = jnp.pad(tensor, ((0, tensor_diff), *extra_dims), "constant")
        tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])

        if use_vmap:
            out = jax.lax.map(lambda chunk: vmap(f)(chunk), tensor)
        else:
            out = jax.lax.map(f, tensor)

        # out = out.reshape(-1, *out.shape[2:])[:initial_len]
        fix_shape = lambda output: output.reshape(-1, *output.shape[2:])[:initial_len]
        if isinstance(out, (tuple, list)):
            out = tuple(fix_shape(output) for output in out)
        else:
            out = fix_shape(out)
        return out


# @functools.partial(jit, static_argnums=(1, 2, 3))
def map_batched_rng(tensor, f, chunksize, use_vmap, rng):
    if tensor.shape[0] < chunksize:
        if use_vmap:
            key, *subkey = jax.random.split(rng, tensor.shape[0] + 1)
            return vmap(f)((tensor, subkey)), key
        else:
            key, subkey = jax.random.split(rng)
            return f((tensor, subkey)), key
    else:
        tensor_diff = -tensor.shape[0] % chunksize
        initial_len = tensor.shape[0]
        tensor_len = tensor.shape[0] + tensor_diff
        extra_dims = tuple((0, 0) for _ in range(len(tensor.shape) - 1))
        tensor = jnp.pad(tensor, ((0, tensor_diff), *extra_dims), "constant")
        tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])

        if use_vmap:
            rng, subrng = jax.random.split(rng)
            subrng = jax.random.split(rng, tensor.shape[0] * tensor.shape[1]).reshape(
                *tensor.shape[:2], 2
            )

            out = jax.lax.map(
                lambda chunk: vmap(f)((chunk[0], chunk[1])), (tensor, subrng)
            )
        else:
            rng, *subrng = jax.random.split(rng, tensor_len // chunksize + 1)
            subrng = jnp.stack(subrng)

            out = jax.lax.map(f, (tensor, subrng))

        fix_shape = lambda output: output.reshape(-1, *output.shape[2:])[:initial_len]
        if isinstance(out, (tuple, list)):
            out = tuple(fix_shape(output) for output in out)
        else:
            out = fix_shape(out)
        return out, rng


def normalize(a):
    return a / jnp.linalg.norm(a, ord=2, axis=-1)


def look_at(eye, center, up):
    n = normalize(center - eye)
    u = normalize(up)
    v = normalize(jnp.cross(n, u))
    u = jnp.cross(v, n, axis=-1)

    out = jnp.zeros((4, 4))

    out = index_update(out, jax.ops.index[0, :3], v)
    out = index_update(out, jax.ops.index[1, :3], u)
    out = index_update(out, jax.ops.index[2, :3], -n)

    out = index_update(out, jax.ops.index[0, 3], jnp.dot(-1.0 * v, eye))
    out = index_update(out, jax.ops.index[1, 3], jnp.dot(-1.0 * u, eye))
    out = index_update(out, jax.ops.index[2, 3], jnp.dot(1.0 * n, eye))
    out = index_update(out, jax.ops.index[3, 3], 1.0)

    return out
