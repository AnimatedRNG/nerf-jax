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
from test_helpers import run_and_grad


def test_positional_encoding():
    inp = np.linspace(0.0, 1.0, 3)

    jf = lambda x: positional_encoding(x, 6)
    tf = lambda x: positional_encoding_torch(x, 6)

    jo, to, djos, dtos = run_and_grad(jf, tf, (0,), inp)

    assert np.allclose(jo, to)
    assert all(np.allclose(djo, dto) for djo, dto in zip(djos, dtos))


def test_cumprod_exclusive():
    inp = np.arange(1, 10).astype(np.float32)

    jo, to, djos, dtos = run_and_grad(
        cumprod_exclusive, cumprod_exclusive_torch, (0,), inp
    )

    assert np.allclose(jo, to)
    assert all(np.allclose(djo, dto) for djo, dto in zip(djos, dtos))


def test_sample_pdf():
    rng = jax.random.PRNGKey(1010)

    bins_np = np.random.randn(8).reshape(2, 4).astype(np.float32) * 10
    weights_np = np.random.randn(8).reshape(2, 4).astype(np.float32) * 15

    jax_fn = lambda bins, weights: sample_pdf(bins, weights, 10, rng, True)
    torch_fn = lambda bins, weights: sample_pdf_torch(bins, weights, 10, True)

    jo, to, djos, dtos = run_and_grad(jax_fn, torch_fn, (0, 1), bins_np, weights_np)

    assert np.allclose(jo, to, rtol=1e-3, atol=1e-5)
    assert all(
        np.allclose(djo, dto, rtol=1e-3, atol=1e-5) for djo, dto in zip(djos, dtos)
    )


def test_get_ray_bundle():
    tfrom_cam2world = np.eye(4, 4, dtype=np.float32)
    tfrom_cam2world[0, 3] = 2.0
    tfrom_cam2world[1, 3] = -3.0
    tfrom_cam2world[2, 3] = 5.0

    for i in range(0, 2):
        jax_fn = lambda x: get_ray_bundle(10, 10, 0.3, x)[i]
        torch_fn = lambda x: get_ray_bundle_torch(10, 10, 0.3, x)[i]

        jo, to, djos, dtos = run_and_grad(jax_fn, torch_fn, (0,), tfrom_cam2world)

        assert np.allclose(jo, to, rtol=1e-3, atol=1e-5)
        assert all(
            np.allclose(djo, dto, rtol=1e-3, atol=1e-5) for djo, dto in zip(djos, dtos)
        )


def test_map_batched():
    # TODO: Fix map batched to properly handle 1D arrays...
    # also why is this test so slow...
    rng = jax.random.PRNGKey(1010)

    chunk_size = 3
    for arr_len in range(2, 7):
        for vmap_val in [False, True]:
            arr = jax.random.uniform(rng, (arr_len, 3))

            sqrt_test = map_batched(arr, lambda a: jnp.sqrt(a), chunk_size, vmap_val)
            assert jnp.allclose(sqrt_test, jnp.sqrt(arr))
