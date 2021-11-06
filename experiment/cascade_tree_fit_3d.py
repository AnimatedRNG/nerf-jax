import os
import sys
import functools

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pywavefront
import matplotlib.pyplot as plt
from drawnow import figure

from sdrf import IGR, CascadeTree, MipMap, exp_smin, exp_smax
from util import plot_iso, plot_heatmap, grid_sample, create_mrc
from cascade_tree_fit_base import fit, get_normals


def normalize_vertices(model_vertices):
    model_min, model_max = jnp.min(model_vertices, axis=0), jnp.max(
        model_vertices, axis=0
    )
    scale_factor = jnp.max(model_max - model_min, axis=0)
    model_vertices = (model_vertices - model_min) / (scale_factor)
    model_vertices = model_vertices * 1.8 - 0.9

    return model_vertices


def get_model(filename):
    if filename.endswith(".obj"):
        model_scene = pywavefront.Wavefront(filename, collect_faces=True)
        assert model_scene.materials["default0"].vertex_format == "V3F"
        model_vertices = jnp.array(model_scene.materials["default0"].vertices).reshape(
            -1, 3
        )

        # normalize vertices
        model_vertices = normalize_vertices(model_vertices)

        faces = jnp.array(model_scene.mesh_list[0].faces, dtype=jnp.int32)
        model_normals = get_normals(model_vertices, faces)

        return model_vertices, model_normals
    else:
        pointcloud = np.genfromtxt(filename)
        v = pointcloud[:, :3]
        n = pointcloud[:, 3:]

        n = n / (np.linalg.norm(n, axis=-1)[:, None])

        # normalize vertices
        v = normalize_vertices(v)

        return v, n


def main():
    rng = jax.random.PRNGKey(1024)

    # create_decoder_fn = lambda: IGR([32, 32, 32, 32], skip_in=(1,), beta=100.0)
    # create_decoder_fn = lambda: IGR([16, 16, 16, 16], skip_in=(1,), beta=100.0)
    create_decoder_fn = lambda: IGR([16], skip_in=tuple(), beta=100.0)

    subrng = jax.random.split(rng, 2)

    feature_size = 16
    max_depth = 5

    grid_min = jnp.array([-1.0, -1.0, -1.0])
    grid_max = jnp.array([1.0, 1.0, 1.0])

    scene = hk.transform(
        lambda p, scale_factor: CascadeTree(
            MipMap(
                create_decoder_fn,
                resolution=64,
                scale_factor=scale_factor,
                grid_min=grid_min,
                grid_max=grid_max,
                feature_size=feature_size,
            ),
            # union_fn=lambda a, b: exp_smax(a, -b),
            # union_fn=lambda a, b: jnp.maximum(a, -b),
            ignore_levels=6,
        )(p)
    )

    fit(
        scene,
        subrng[1],
        3.0,
        # *get_model("../data/stanford-bunny.obj"),
        *get_model("../data/bunny_watertight.xyz"),
        visualization_hook=visualization_hook,
        batch_size=2 ** 16,
        lr=1e-4,
        num_epochs=10001,
    )


def visualization_hook(
    scene_fn,
    points,
    normals,
    params,
    grid_min=jnp.array([-1.0, -1.0, -1.0]),
    grid_max=jnp.array([1.0, 1.0, 1.0]),
):
    num_scene_fns = len(scene_fn(params, jnp.zeros((3,)))[1])
    for level in range(num_scene_fns):
        print("num_scene_fns", num_scene_fns)
        scene_fn_level = lambda pt: scene_fn(params, pt)[1][level]
        create_mrc(
            f"test_{level}.mrc",
            scene_fn_level,
            grid_min,
            grid_max,
            64,
        )

    create_mrc(
        "test.mrc",
        lambda pt: scene_fn(params, pt)[0][0],
        grid_min,
        grid_max,
        64,
    )
    """lt.clf()

    points, normals = np.array(points), np.array(normals)
    ax = plt.subplot(projection="3d")
    ax.quiver(
        points[::512, 0],
        points[::512, 1],
        points[::512, 2],
        points[::512, 0],
        points[::512, 1],
        points[::512, 2],
        length=0.1,
        normalize=True,
    )"""


if __name__ == "__main__":
    main()
