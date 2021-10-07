import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sdrf import IGR, MipMap, CascadeTree, exp_smin
from util import plot_iso, plot_heatmap
from cascade_tree_fit_base import fit


def get_model(npoints=1000, pointcloud_path="./bunny2d.ply", keep_aspect_ratio=True):

    point_cloud = np.genfromtxt(pointcloud_path)
    coords = point_cloud[:, :2]
    # normals = point_cloud[:, 3:]

    # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
    # sample efficiency)
    coords -= np.mean(coords, axis=0, keepdims=True)
    if keep_aspect_ratio:
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
    else:
        coord_max = np.amax(coords, axis=0, keepdims=True)
        coord_min = np.amin(coords, axis=0, keepdims=True)

    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= 2.0
    vec = coords

    # vec = np.random.randn(npoints, 2)
    # vec /= np.linalg.norm(vec, axis=0)
    return jnp.array(vec)


def visualization_hook(
    scene_fn,
    points,
    params,
    grid_min=jnp.array([-1.0, -1.0]),
    grid_max=jnp.array([1.0, 1.0]),
):
    plt.clf()
    plt.subplot(1, 2, 1)
    plot_iso(
        lambda pt: scene_fn(params, jnp.array([pt[0], pt[2]])),
        jnp.array([-1.0, -1.0]),
        jnp.array([1.0, 1.0]),
        256,
    )

    plt.plot(points[:, 0], points[:, 1], ".")

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
    max_depth = 3

    grid_min = jnp.array([-1.0, -1.0])
    grid_max = jnp.array([1.0, 1.0])

    scene = hk.transform(
        lambda p: MipMap(
            create_decoder_fn,
            resolution=16,
            grid_min=grid_min,
            grid_max=grid_max,
            feature_size=feature_size,
        )(p)
    )

    # get a better 2D model
    # model = get_model()
    # plt.plot(model[:, 0], model[:, 1], '.')
    # plt.show()

    fit(
        scene,
        get_model(),
        subrng[1],
        visualization_hook=visualization_hook,
    )


if __name__ == "__main__":
    main()
