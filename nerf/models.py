import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import functools


class FlexibleNeRFModel(hk.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        name=None,
    ):
        super(FlexibleNeRFModel, self).__init__(name=name)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.skip_connect_every = skip_connect_every
        self.num_encoding_fn_xyz = num_encoding_fn_xyz
        self.num_encoding_fn_dir = num_encoding_fn_dir
        self.include_input_xyz = include_input_xyz
        self.include_input_dir = include_input_dir
        self.use_viewdirs = use_viewdirs

    def __call__(self, x):
        include_input_xyz = 3 if self.include_input_xyz else 0
        include_input_dir = 3 if self.include_input_dir else 0
        dim_xyz = include_input_xyz + 2 * 3 * self.num_encoding_fn_xyz
        dim_dir = include_input_dir + 2 * 3 * self.num_encoding_fn_dir
        if not self.use_viewdirs:
            dim_dir = 0
            xyz = x[..., : self.dim_xyz]
        else:
            xyz, view = x[..., :dim_xyz], x[..., dim_xyz:]

        x = hk.Linear(self.hidden_size, name="layer1")(xyz)

        for i in range(self.num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != self.num_layers - 1:
                x = jnp.concatenate((x, xyz), axis=-1)

            x = jax.nn.relu(
                hk.Linear(self.hidden_size, name="layers_xyz__{}".format(i))(x)
            )
        if self.use_viewdirs:
            feat = jax.nn.relu(hk.Linear(self.hidden_size, name="fc_feat")(x))
            alpha = hk.Linear(1, name="fc_alpha")(x)
            x = jnp.concatenate((feat, view), axis=-1)
            x = jax.nn.relu(
                hk.Linear(self.hidden_size // 2, name="layers_dir__{}".format(0))(x)
            )
            rgb = hk.Linear(3, name="fc_rgb")(x)
            return jnp.concatenate((rgb, alpha), axis=-1)
        else:
            return hk.Linear(4, name="fc_out")(x)
