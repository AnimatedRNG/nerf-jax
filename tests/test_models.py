import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import torch

from reference import FlexibleNeRFModelTorch, torch_to_jax
from nerf import FlexibleNeRFModel


def test_flexible_nerf_model():
    x = np.ones((250, 66), dtype=np.float32)

    l = np.random.random(1)[0]

    key = hk.PRNGSequence(42)
    net = hk.transform(lambda x: FlexibleNeRFModel()(x))
    params = net.init(next(key), jnp.array(x))

    params = jax.tree_util.tree_map(lambda v: jnp.ones_like(v) * l, params)

    net_torch = FlexibleNeRFModelTorch()
    with torch.no_grad():
        _ = [v.fill_(l) for v in net_torch.parameters()]

    torch_out = net_torch(torch.from_numpy(x))
    jnp_out = net.apply(params, next(key), jnp.array(x))
    assert np.allclose(torch_out.detach().numpy(), np.array(jnp_out))


def test_torch_to_jax():
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

    assert np.allclose(torch_out.detach().numpy(), np.array(jax_out), atol=1e-7)
