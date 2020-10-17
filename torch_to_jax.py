#!/usr/bin/env python3

import numpy
import torch
import jax
import jax.numpy as jnp

from models import FlexibleNeRFModel


def torch_to_jax(torch_params, model_name):
    """
    >>> x = np.random.randn(250, 66).astype(np.float32)

    >>> net_torch = FlexibleNeRFModelTorch()
    >>> net_jax = hk.without_apply_rng(hk.transform(jax.jit(lambda x: FlexibleNeRFModel()(x))))

    >>> jax_params = torch_to_jax(dict(net_torch.named_parameters()), 'flexible_ne_rf_model')

    >>> jax_out = net_jax.apply(jax_params, jnp.array(x))
    >>> torch_out = net_torch(torch.from_numpy(x))

    >>> np.allclose(torch_out.detach().numpy(), np.array(jax_out), atol=1e-7)
    True
    """
    jax_params = {}
    replacements = {"weight": "w", "bias": "b"}

    for name, param in torch_params.items():
        delimiters = tuple(name.split("."))

        attrib = delimiters[0]
        attrib = replacements.get(attrib, attrib)

        prefix = f"{model_name}/{attrib}"

        if len(delimiters) > 1:
            try:
                prefix += f"__{int(delimiters[1])}"
                delimiters = delimiters[1:]
            except ValueError:
                pass

        if len(delimiters) > 1:
            existing_value = jax_params.setdefault(prefix, {})
            existing_value[".".join(delimiters[1:])] = param
        else:
            jax_params[attrib] = param

    out = {}
    for jax_name, jax_nest in jax_params.items():
        if isinstance(jax_nest, torch.Tensor):
            # find cleaner solution than transpose :)
            out[jax_name] = jnp.array(jax_nest.detach().cpu().numpy().T)
        else:
            out[jax_name] = torch_to_jax(jax_nest, jax_name)
    return out


if __name__ == "__main__":
    import doctest
    import torch
    import haiku as hk
    from torch_impl import *

    print(doctest.testmod(exclude_empty=True))
