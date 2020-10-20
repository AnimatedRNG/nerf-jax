import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad

import torch
from torch.autograd import grad as torch_grad

def sum_loss_torch(ten):
    return ten.flatten().sum()


def sum_loss(f):
    return lambda *args, **kwargs: f(*args, **kwargs).flatten().sum()


def run_and_grad(jax_fn, torch_fn, differentiable_inputs, *args):
    inps = list(args)

    jax_inps = [jnp.array(inp) for inp in inps]
    torch_inps = [torch.from_numpy(inp).requires_grad_(True) for inp in inps]
    differentiable_torch_inputs = [
        inp for idx, inp in enumerate(torch_inps) if idx in differentiable_inputs
    ]

    jax_outs = jit(jax_fn)(*jax_inps)
    torch_outs = torch_fn(*torch_inps)

    jax_grad_outs = jit(grad(sum_loss(jax_fn), argnums=differentiable_inputs),)(
        *jax_inps
    )
    torch_grad_outs = torch_grad(
        sum_loss_torch(torch_outs), differentiable_torch_inputs
    )

    return (
        np.array(jax_outs),
        torch_outs.detach().numpy(),
        tuple(np.array(jo) for jo in jax_grad_outs),
        (to.detach().numpy() for to in torch_grad_outs),
    )
