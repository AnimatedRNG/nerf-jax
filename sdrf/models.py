from typing import Sequence, Union
import functools

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk


def get_fan(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        raise Exception("invalid shape for SIREN!")

    return fan_in, fan_out


class SineInitializer(hk.initializers.Initializer):
    def __init__(self, first_layer=False):
        self.first_layer = first_layer

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        fan_in, fan_out = get_fan(shape)

        if self.first_layer:
            return jax.random.uniform(
                hk.next_rng_key(), shape, dtype, -1.0 / fan_in, 1.0 / fan_in,
            )
        else:
            return jax.random.uniform(
                hk.next_rng_key(),
                shape,
                dtype,
                -jnp.sqrt(6.0 / fan_in) / 30.0,
                jnp.sqrt(6.0 / fan_in) / 30.0,
            )


class PyTorchBiasInitializer(hk.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        fan_in, fan_out = get_fan(shape)

        stdv = 1.0 / jnp.sqrt(fan_in)
        return hk.initializers.RandomUniform(-stdv, stdv)(shape, dtype)


class GeometricInitializer(hk.initializers.Initializer):
    def __init__(self, last_layer=False):
        self.last_layer = last_layer

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        fan_in, fan_out = get_fan(shape)

        if self.last_layer:
            mean = jnp.sqrt(np.pi) / jnp.sqrt(fan_in)
            stdv = 1e-5
            return hk.initializers.RandomNormal(stdv, mean)(shape, dtype)
        else:
            mean = 0.0
            stdv = jnp.sqrt(2.0) / jnp.sqrt(fan_out)
            return hk.initializers.RandomNormal(stdv, mean)(shape, dtype)


class IGR(hk.Module):
    def __init__(self, depths, skip_in=(4,), radius_init=1, beta=100, name=None):
        super().__init__(name=name)
        self.depths = depths
        self.skip_in = skip_in
        self.radius_init = radius_init
        self.beta = beta

    def __call__(self, coords):
        embedding_size = coords.shape[-1]

        w_init_n = GeometricInitializer()
        w_init_last = GeometricInitializer(True)
        b_init_n = hk.initializers.Constant(-self.radius_init)
        b_init_last = hk.initializers.Constant(0.0)

        if self.beta > 0:
            activation = lambda x: jax.nn.softplus(self.beta * x) / self.beta
        else:
            activation = lambda x: jnp.maximum(x, 0.0)

        emb = coords
        x = jnp.array(emb)

        depths = [embedding_size] + self.depths + [1]

        for layer_id in range(0, len(self.depths) - 1):
            if layer_id + 1 in self.skip_in:
                out_dim = self.depths[layer_id + 1] - embedding_size
            else:
                out_dim = self.depths[layer_id + 1]

            if layer_id in self.skip_in:
                x = jnp.concatenate((x, emb), axis=-1) / jnp.sqrt(2)

            w_init = w_init_last if layer_id == len(self.depths) - 2 else w_init_n
            b_init = b_init_last if layer_id == len(self.depths) - 2 else b_init_n
            x = hk.Linear(
                out_dim, w_init=w_init, b_init=b_init, name=f"layer_{layer_id}",
            )(x)

            if layer_id < len(self.depths) - 2:
                x = activation(x)
        return x


class Siren(hk.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=True,
        name=None,
    ):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear

    def __call__(self, coords):
        sine_layer = lambda x: jnp.sin(30 * x)

        w_init_0 = SineInitializer(True)
        b_init = PyTorchBiasInitializer()

        w_init_n = SineInitializer(False)

        x = sine_layer(
            hk.Linear(
                self.hidden_features, w_init=w_init_0, b_init=b_init, name="layer_0",
            )(coords)
        )

        # turn this into a jax.lax.scan?
        for i in range(self.num_hidden_layers):
            x = sine_layer(
                hk.Linear(
                    self.hidden_features,
                    w_init=w_init_n,
                    b_init=b_init,
                    name=f"layer_{i + 1}",
                )(x)
            )

        x = hk.Linear(
            self.out_features,
            w_init=w_init_n,
            b_init=b_init,
            name=f"layer_{self.num_hidden_layers + 1}",
        )(x)

        if self.outermost_linear:
            x = sine_layer(x)

        return x
