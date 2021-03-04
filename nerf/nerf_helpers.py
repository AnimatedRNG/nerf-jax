import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.ops import index, index_update


# @functools.partial(jit, static_argnums=(1,))
def positional_encoding(tensor, num_encoding_functions):
    frequency_bands = 2.0 ** jnp.linspace(
        0.0,
        num_encoding_functions - 1,
        #2 ** (num_encoding_functions - 1),
        num_encoding_functions,
        dtype=jnp.float32,
    )[..., jnp.newaxis]

    encoding = jnp.zeros((num_encoding_functions, 2, tensor.shape[0]))
    ten_arr = tensor[jnp.newaxis, ...]

    ten_freq = frequency_bands @ ten_arr

    encoding = index_update(encoding, index[:, 0, :], jnp.sin(ten_freq))
    encoding = index_update(encoding, index[:, 1, :], jnp.cos(ten_freq))

    if num_encoding_functions == 0:
        return tensor
    else:
        return jnp.concatenate((tensor, encoding.flatten()))
        #return encoding.flatten()


# @jit
def cumprod_exclusive(tensor):
    prod = jnp.roll(jnp.cumprod(tensor, axis=-1), 1, axis=-1)
    return index_update(prod, index[..., 0], 1.0)


# @functools.partial(jit, static_argnums=(2, 4))
def sample_pdf(bins, weights, num_samples, rng, det):
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
