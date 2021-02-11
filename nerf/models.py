import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import functools
import enum


def compute_embedding_size(
    include_input_xyz, include_input_dir, num_encoding_fn_xyz, num_encoding_fn_dir
):
    include_input_xyz = 3 if include_input_xyz else 0
    include_input_dir = 3 if include_input_dir else 0
    dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
    dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

    return dim_xyz, dim_dir


class NeRFModelMode(enum.IntEnum):
    BOTH = 0
    GEOMETRY = 1
    APPEARANCE = 2


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
        geometric_init=False,
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
        self.geometric_init = geometric_init

    def linear(self, size, name, l=None, skip=False):
        if l is None or not self.geometric_init:
            w_init, b_init = (
                hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            )
        else:
            if l == self.num_layers - 1:
                pass
            elif l == 0:
                pass
            elif skip:
                pass
            else:
                pass

        return hk.Linear(
            size,
            name=name,
            w_init=w_init,
            b_init=b_init,
        )

    def __call__(self, xyz, view, mode=NeRFModelMode.BOTH):
        dim_xyz, dim_dir = compute_embedding_size(
            self.include_input_xyz,
            self.include_input_dir,
            self.num_encoding_fn_xyz,
            self.num_encoding_fn_dir,
        )

        x = self.linear(self.hidden_size, name="layer1", l=0)(xyz)

        for i in range(self.num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != self.num_layers - 1:
                skip = True
                x = jnp.concatenate((x, xyz), axis=-1)
                if self.geometric_init:
                    x /= jnp.sqrt(2.0)
            else:
                skip = False

            x = jax.nn.relu(
                self.linear(
                    self.hidden_size,
                    name="layers_xyz__{}".format(i),
                    l=i + 1,
                    skip=skip,
                )(x)
            )
        if self.use_viewdirs:
            feat = jax.nn.relu(self.linear(self.hidden_size, name="fc_feat")(x))
            alpha = self.linear(1, name="fc_alpha", l=self.num_layers)(x)

            if mode == NeRFModelMode.GEOMETRY:
                return alpha

            x = jnp.concatenate((feat, view), axis=-1)
            x = jax.nn.relu(
                self.linear(
                    self.hidden_size // 2,
                    name="layers_dir__{}".format(0),
                )(x)
            )
            rgb = self.linear(3, name="fc_rgb")(x)

            if mode == NeRFModelMode.BOTH:
                return (rgb, alpha)
            else:
                return alpha
        else:
            geo, appearance = (
                self.linear(1, name="fc_out", l=self.num_layers + 1)(x),
                self.linear(3, name="fc_out")(x),
            )
            if mode == NeRFModelMode.GEOMETRY:
                return geo
            elif mode == NeRFModelMode.APPEARANCE:
                return appearance
            else:
                return (appearance, geo)
