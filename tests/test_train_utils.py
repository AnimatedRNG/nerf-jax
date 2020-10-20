import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import haiku as hk
import torch

from reference import (
    FlexibleNeRFModelTorch,
    positional_encoding_torch,
    run_network_torch,
    torch_to_jax,
)
from nerf import FlexibleNeRFModel, positional_encoding, run_network


def test_run_network():
    x = np.random.randn(250, 66).astype(np.float32)

    net_torch = FlexibleNeRFModelTorch()
    net_jax = hk.without_apply_rng(
        hk.transform(jax.jit(lambda x: FlexibleNeRFModel()(x)))
    )

    jax_params = torch_to_jax(
        dict(net_torch.named_parameters()), "flexible_ne_rf_model"
    )

    jax_out = net_jax.apply(jax_params, jnp.array(x))
    torch_out = net_torch(torch.from_numpy(x))

    pts_np = np.random.random((256, 128, 3)).astype(np.float32)
    ray_batch_np = np.random.random((256, 11)).astype(np.float32)

    pts_torch = torch.from_numpy(pts_np)
    ray_batch_torch = torch.from_numpy(ray_batch_np)

    pts_jax = jnp.array(pts_np)
    ray_batch_jax = jnp.array(ray_batch_np)

    torch_result = run_network_torch(
        net_torch,
        pts_torch,
        ray_batch_torch,
        32,
        lambda p: positional_encoding_torch(p, 6),
        lambda p: positional_encoding_torch(p, 4),
    )
    jax_result = run_network(
        functools.partial(net_jax.apply, jax_params), pts_jax, ray_batch_jax, 32, 6, 4
    )

    assert np.allclose(np.array(jax_result), torch_result.detach().numpy(), atol=1e-7)

    '''jax_fn = (
        lambda pt, rb, p: run_network(
            functools.partial(net_jax.apply, p), pt, rb, 32, 6, 4
        )
        .flatten()
        .sum()
    )

    dx = jit(grad(jax_fn, argnums=(0, 1, 2)))(pts_jax, ray_batch_jax, jax_params)
    print(dx)'''
