#!/usr/bin/env python3

import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam, sgd
from jax import grad, vmap
import numpy as np
from drawnow import drawnow, figure

from sdrf import ConvolutionalSDF, eikonal_loss
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
    normals,
    params,
    grid_min=jnp.array([-1.0, -1.0]),
    grid_max=jnp.array([1.0, 1.0]),
):
    plt.clf()
    plt.subplot(1, 2, 1)
    plot_iso(
        lambda pt, _sf, _kl: scene_fn(params, jnp.array([pt[0], pt[2]])),
        jnp.array([-1.0, -1.0]),
        jnp.array([1.0, 1.0]),
        256,
    )

    plt.plot(points[:, 0], points[:, 1], ".")

    plt.subplot(1, 2, 2)
    plot_heatmap(
        lambda pt: scene_fn(params, jnp.array([pt[0], pt[2]])),
        jnp.array([-1.0, -1.0]),
        jnp.array([1.0, 1.0]),
        256,
    )


def main():
    rng = jax.random.PRNGKey(1024)

    scene = hk.transform(lambda x, _sf, _kl: ConvolutionalSDF(32, 2, 32)(x))

    eikonal_fn = lambda scene_fn: eikonal_loss(scene_fn(jnp.zeros([1, 2]))[1])

    fit(
        scene,
        rng,
        10.0,
        get_model(),
        map_fn=lambda x: x[1],
        eikonal_fn=eikonal_fn,
        visualization_hook=visualization_hook,
    )


if __name__ == "__main__":
    main()
