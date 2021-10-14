import os
import sys
import functools

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import haiku as hk
import jax
import jax.numpy as jnp
import pywavefront
import matplotlib.pyplot as plt
from drawnow import figure

from sdrf import IGR, CascadeTree, MipMap, exp_smin
from util import plot_iso, plot_heatmap, create_mrc
from cascade_tree_fit_base import fit, get_normals


def get_model(filename):
    model_scene = pywavefront.Wavefront(filename, collect_faces=True)
    assert model_scene.materials["default0"].vertex_format == "V3F"
    model_vertices = jnp.array(model_scene.materials["default0"].vertices).reshape(
        -1, 3
    )

    # normalize vertices
    model_min, model_max = jnp.min(model_vertices, axis=0), jnp.max(
        model_vertices, axis=0
    )
    scale_factor = jnp.max(model_max - model_min, axis=0)
    model_vertices = (model_vertices - model_min) / (scale_factor)
    model_vertices = model_vertices * 2 - 1

    faces = jnp.array(model_scene.mesh_list[0].faces, dtype=jnp.int32)
    model_normals = get_normals(model_vertices, faces)

    return model_vertices, model_normals


def main():
    rng = jax.random.PRNGKey(1024)

    create_decoder_fn = lambda: IGR([32, 32, 32], skip_in=(1,), beta=100.0)

    subrng = jax.random.split(rng, 2)

    feature_size = 16
    max_depth = 5

    grid_min = jnp.array([-1.0, -1.0, -1.0])
    grid_max = jnp.array([1.0, 1.0, 1.0])

    scene = hk.transform(
        lambda p, scale_factor, kern_length: MipMap(
            create_decoder_fn,
            resolution=32,
            scale_factor=scale_factor,
            kern_length=kern_length,
            grid_min=grid_min,
            grid_max=grid_max,
            feature_size=feature_size,
        )(p)
    )

    fit(
        scene,
        subrng[1],
        30.0,
        *get_model("../data/stanford-bunny.obj"),
        visualization_hook=visualization_hook,
        batch_size=2 ** 10,
        lr=1e-3
    )

def visualization_hook(
    scene_fn,
    points,
    params,
    grid_min=jnp.array([-1.0, -1.0, -1.0]),
    grid_max=jnp.array([1.0, 1.0, 1.0]),
):
    create_mrc("test.mrc", functools.partial(scene_fn, params), grid_min, grid_max, 256)


if __name__ == "__main__":
    main()
