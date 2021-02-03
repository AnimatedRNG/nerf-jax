import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import functools


 def compute_embedding_size(
    include_input_xyz, include_input_dir, num_encoding_fn_xyz, num_encoding_fn_dir
):
    include_input_xyz = 3 if include_input_xyz else 0
    include_input_dir = 3 if include_input_dir else 0
    dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
    dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

    return dim_xyz, dim_dir


def linear(size, name, w_init, b_init):
    return hk.Linear(
        size,
        name=name,
        w_init=w_init,
        b_init=b_init
        # w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
        # b_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
    )


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
        w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
        b_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
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
        self.w_init = w_init
        self.b_init = b_init

    def __call__(self, xyz, view):
        dim_xyz, dim_dir = compute_embedding_size(
            self.include_input_xyz,
            self.include_input_dir,
            self.num_encoding_fn_xyz,
            self.num_encoding_fn_dir,
        )

        x = linear(
            self.hidden_size, name="layer1", w_init=self.w_init, b_init=self.b_init
        )(xyz)

        for i in range(self.num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != self.num_layers - 1:
                x = jnp.concatenate((x, xyz), axis=-1)

            x = jax.nn.relu(
                linear(
                    self.hidden_size,
                    name="layers_xyz__{}".format(i),
                    w_init=self.w_init,
                    b_init=self.b_init,
                )(x)
            )
        if self.use_viewdirs:
            feat = jax.nn.relu(
                linear(
                    self.hidden_size,
                    name="fc_feat",
                    w_init=self.w_init,
                    b_init=self.b_init,
                )(x)
            )
            alpha = linear(1, name="fc_alpha", w_init=self.w_init, b_init=self.b_init)(
                x
            )
            x = jnp.concatenate((feat, view), axis=-1)
            x = jax.nn.relu(
                linear(
                    self.hidden_size // 2,
                    name="layers_dir__{}".format(0),
                    w_init=self.w_init,
                    b_init=self.b_init,
                )(x)
            )
            rgb = linear(3, name="fc_rgb", w_init=self.w_init, b_init=self.b_init)(x)
            return (rgb, alpha)
        else:
            return (
                linear(3, name="fc_out", w_init=self.w_init, b_init=self.b_init)(x),
                linear(1, name="fc_out", w_init=self.w_init, b_init=self.b_init)(x),
            )
