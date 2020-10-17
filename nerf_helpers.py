#!/usr/bin/env python3

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.ops import index, index_update
import functools


@functools.partial(jit, static_argnums=(1,))
def positional_encoding(tensor, num_encoding_functions):
    """
    >>> a = positional_encoding_torch(torch.linspace(0.0, 1.0, 3), 6).numpy()
    >>> b = np.array(positional_encoding(jnp.linspace(0.0, 1.0, 3), 6))

    >>> np.allclose(a, b)
    True
    """
    frequency_bands = (
        2.0
        ** jnp.linspace(
            0.0, num_encoding_functions - 1, num_encoding_functions, dtype=jnp.float32,
        )[..., jnp.newaxis]
    )

    encoding = jnp.zeros((num_encoding_functions, 2, tensor.shape[0]))
    ten_arr = tensor[jnp.newaxis, ...]

    ten_freq = frequency_bands @ ten_arr

    encoding = index_update(encoding, index[:, 0, :], jnp.sin(ten_freq))
    encoding = index_update(encoding, index[:, 1, :], jnp.cos(ten_freq))

    if num_encoding_functions == 0:
        return tensor
    else:
        return jnp.concatenate((tensor, encoding.flatten()))


@jit
def cumprod_exclusive(tensor):
    """
    >>> a = cumprod_exclusive_torch(torch.arange(1, 10)).numpy()
    >>> b = np.array(cumprod_exclusive(jnp.arange(1, 10)))
    >>> np.array_equal(a, b)
    True
    """
    prod = jnp.roll(jnp.cumprod(tensor, axis=-1), 1, axis=-1)
    return index_update(prod, index[..., 0], 1.0)


@functools.partial(jit, static_argnums=(2, 3, 4))
def sample_pdf(bins, weights, num_samples, rng, det):
    """
    >>> rng = jax.random.PRNGKey(1010)

    >>> bins_np = np.random.randn(8).reshape(2, 4).astype(np.float32) * 10
    >>> weights_np = np.random.randn(8).reshape(2, 4).astype(np.float32) * 15

    >>> samples_torch = sample_pdf_torch(torch.from_numpy(bins_np),
    ...     torch.from_numpy(weights_np), 10, True).numpy()

    >>> bins = jnp.array(bins_np)
    >>> weights = jnp.array(weights_np)
    >>> samples = np.array(sample_pdf(bins, weights, 10, rng, True))

    >>> np.allclose(samples_torch, samples, 1e-3)
    True
    """
    weights = weights + 1e-5
    pdf = weights / jnp.sum(weights, axis=-1, keepdims=True)
    cdf = jnp.cumsum(pdf, -1)
    cdf = jnp.concatenate((jnp.zeros_like(cdf[..., :1]), cdf), -1)

    if det:
        u = jnp.linspace(0.0, 1.0, num_samples)
        u = jnp.repeat(jnp.expand_dims(u, 0), cdf.shape[:-1], axis=0)
    else:
        u = jax.random.uniform(rng, list(cdf.shape[:-1]) + [num_samples])

    inds = vmap(
        lambda cdf_i, u_i: jnp.searchsorted(cdf_i, u_i, side="right").astype(np.int32)
    )(cdf, u)

    below = jnp.maximum(0, inds - 1)
    above = jnp.minimum(cdf.shape[-1] - 1, inds)
    inds_g = jnp.stack((below, above), axis=-1)

    cdf_g = vmap(lambda cdf_i, inds_gi: cdf_i[inds_gi])(cdf, inds_g)
    bins_g = vmap(lambda bins_i, inds_gi: bins_i[inds_gi])(bins, inds_g)

    # don't know why we have to zero out the outliers?
    clean_inds = lambda arr, cutoff: jnp.where(inds_g < cutoff, arr, 0)
    cdf_g = clean_inds(cdf_g, cdf.shape[-1])
    bins_g = clean_inds(bins_g, bins.shape[-1])

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = jnp.where(denom < 1e-5, 1.0, denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


@functools.partial(jit, static_argnums=(0, 1, 2))
def get_ray_bundle(height, width, focal_length, tfrom_cam2world):
    """
    >>> tfrom_cam2world = np.eye(4, 4, dtype=np.float32)
    >>> tfrom_cam2world[0, 3] = 2.0
    >>> tfrom_cam2world[1, 3] = -3.0
    >>> tfrom_cam2world[2, 3] = 5.0
    >>> bundle_torch = get_ray_bundle_torch(10, 10, 0.3, torch.from_numpy(tfrom_cam2world))
    >>> bundle = get_ray_bundle(10, 10, 0.3, jnp.array(tfrom_cam2world))
    >>> np.allclose(bundle_torch[0].numpy(), np.array(bundle[0]))
    True
    >>> np.allclose(bundle_torch[1].numpy(), np.array(bundle[1]))
    True
    """
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


@functools.partial(jit, static_argnums=(1, 2, 3))
def map_batched(tensor, f, chunksize, use_vmap):
    """
    TODO: Write test case

    Splits a tensor into chunks, applies the function `f` on each chunk, and
    then rejoins the chunks.

    Args:
    tensor (JAX tensor):            Input provided to `f`
    f (JIT-compiled JAX function):  Function which performs the mapping
    chunksize (int):                Size of each batch
    use_vmap (bool):                If True, track history for each batch
                                    independently (faster, but more memory),
                                    otherwise, reuse tensors allocated for `f`
                                    (only uses one batch of memory).
    Returns:
    out                             Result of applying `f` to tensor
    """
    tensor_diff = -tensor.shape[0] % chunksize
    tensor_len = tensor.shape[0] + tensor_diff
    tensor = jnp.pad(tensor, ((0, tensor_diff), (0, 0)), "constant")
    tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])
    if use_vmap:
        out = vmap(f)(tensor)  # this unfortunately keeps each batch in memory...
    else:
        out = jax.lax.map(f, tensor)
    out = out.reshape(-1, *out.shape[2:])[:tensor_len]
    return out


if __name__ == "__main__":
    import doctest
    import torch
    from torch_impl import *

    print(doctest.testmod(exclude_empty=True))
