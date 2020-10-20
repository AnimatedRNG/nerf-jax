import numpy as np
import jax
from jax import jit, grad
import jax.numpy as jnp
import haiku as hk
import torch
from torch.autograd import grad as torch_grad

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

    # now let's verify that the gradients are correct
    jax_fn = lambda x, p: net_jax.apply(p, x).flatten().sum()

    jax_params_grad = jit(grad(jax_fn, argnums=(1,)))(jnp.array(x), jax_params)[0]

    torch_loss = torch_out.flatten().sum()
    torch_loss.backward()

    torch_grads = torch_to_jax(
        {k: v.grad for k, v in net_torch.named_parameters()}, "flexible_ne_rf_model"
    )

    def recursive_compare(d1, d2):
        assert(d1.keys() == d2.keys())
        for key in d1.keys():
            if isinstance(d1[key], dict):
                assert isinstance(d2[key], dict)
                recursive_compare(d1[key], d2[key])
            else:
                assert np.allclose(d1[key], d2[key], rtol=1e-3, atol=1e-7)

    recursive_compare(jax_params_grad, torch_grads)
