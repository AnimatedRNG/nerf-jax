import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

import torch

from reference import (
    positional_encoding_torch,
    cumprod_exclusive_torch,
    sample_pdf_torch,
    get_ray_bundle_torch,
)
from nerf import (
    positional_encoding,
    cumprod_exclusive,
    sample_pdf,
    get_ray_bundle,
    map_batched,
)


def test_positional_encoding():
    a = positional_encoding_torch(torch.linspace(0.0, 1.0, 3), 6).numpy()
    b = np.array(positional_encoding(jnp.linspace(0.0, 1.0, 3), 6))

    assert np.allclose(a, b)


def test_cumprod_exclusive():
    a = cumprod_exclusive_torch(torch.arange(1, 10)).numpy()
    b = np.array(cumprod_exclusive(jnp.arange(1, 10)))

    assert np.array_equal(a, b)


def test_sample_pdf():
    rng = jax.random.PRNGKey(1010)

    bins_np = np.random.randn(8).reshape(2, 4).astype(np.float32) * 10
    weights_np = np.random.randn(8).reshape(2, 4).astype(np.float32) * 15

    samples_torch = sample_pdf_torch(
        torch.from_numpy(bins_np), torch.from_numpy(weights_np), 10, True
    ).numpy()

    bins = jnp.array(bins_np)
    weights = jnp.array(weights_np)
    samples = np.array(sample_pdf(bins, weights, 10, rng, True))

    assert np.allclose(samples_torch, samples, 1e-3)


def test_get_ray_bundle():
    tfrom_cam2world = np.eye(4, 4, dtype=np.float32)
    tfrom_cam2world[0, 3] = 2.0
    tfrom_cam2world[1, 3] = -3.0
    tfrom_cam2world[2, 3] = 5.0
    bundle_torch = get_ray_bundle_torch(10, 10, 0.3, torch.from_numpy(tfrom_cam2world))
    bundle = get_ray_bundle(10, 10, 0.3, jnp.array(tfrom_cam2world))

    assert np.allclose(bundle_torch[0].numpy(), np.array(bundle[0]))
    assert np.allclose(bundle_torch[1].numpy(), np.array(bundle[1]))


def test_map_batched():
    # TODO: Fix map batched to properly handle 1D arrays...
    # also why is this test so slow...
    rng = jax.random.PRNGKey(1010)

    chunk_size = 3
    for arr_len in range(2, 7):
        for vmap_val in [False, True]:
            arr = jax.random.uniform(rng, (arr_len, 3))

            sqrt_test = map_batched(
                arr, lambda a: jnp.sqrt(a), chunk_size, vmap_val
            )
            assert jnp.allclose(sqrt_test, jnp.sqrt(arr))
