import os
import sys
import functools

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import pywavefront
import matplotlib.pyplot as plt
from drawnow import figure

from sdrf import IGR, CascadeTree, exp_smin
from util import plot_iso, plot_heatmap, create_mrc
from cascade_tree_fit_base import fit


def get_model(npoints=1000):
    vec = np.random.randn(npoints, 2)
    vec /= np.linalg.norm(vec, axis=0)
    return jnp.array(vec)


def visualization_hook(
    scene_fn,
    params,
    grid_min=jnp.array([-1.0, -1.0]),
    grid_max=jnp.array([1.0, 1.0]),
):
    plt.subplot(1, 2, 1)
    plot_iso(
        lambda pt: scene_fn(params, jnp.array([pt[0], pt[2]])),
        jnp.array([-1.0, -1.0]),
        jnp.array([1.0, 1.0]),
        256,
    )

    plt.subplot(1, 2, 2)
    plot_heatmap(
        lambda pt: scene_fn(params, jnp.array([pt[0], pt[2]])).repeat(3, axis=-1),
        jnp.array([-1.0, -1.0]),
        jnp.array([1.0, 1.0]),
        256,
    )


def main():
    rng = jax.random.PRNGKey(1024)

    create_decoder_fn = lambda: IGR([32, 32], beta=0)

    subrng = jax.random.split(rng, 2)

    feature_size = 16
    max_depth = 5

    grid_min = jnp.array([-1.0, -1.0])
    grid_max = jnp.array([1.0, 1.0])

    scene = hk.transform(
        lambda p: CascadeTree(
            create_decoder_fn,
            grid_min=grid_min,
            grid_max=grid_max,
            union_fn=lambda a, b: exp_smin(a, b, 32),
            max_depth=max_depth,
            feature_size=feature_size,
        )(p)
    )

    fit(
        scene,
        get_model(),
        subrng[1],
        visualization_hook=visualization_hook,
    )


if __name__ == "__main__":
    main()
