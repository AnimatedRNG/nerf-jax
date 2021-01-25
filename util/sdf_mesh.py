#!/usr/bin/env python3

import tempfile
import urllib.request

import numpy as np
import jax
from jax import vmap, jit
import jax.numpy as jnp
import pywavefront

from util import map_batched

bunny = None


def get_bunny():
    global bunny
    if bunny is None:
        # url = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
        url = "https://cs.uwaterloo.ca/~c2batty/bunny_watertight.obj"
        response = urllib.request.urlopen(url)
        data = response.read()
        text = data.decode("utf-8")
        with tempfile.NamedTemporaryFile(mode="w", dir="/tmp", delete=False) as tmpfile:
            tmpfile.write(text)
            bunny = np.array(
                pywavefront.Wavefront(tmpfile.name, collect_faces=True)
                .materials["default0"]
                .vertices
            ).reshape(-1, 3, 3)
        vs = bunny.reshape(-1, 3)
        min_vertex = vs.min(axis=0)
        max_vertex = vs.max(axis=0)
        span = max_vertex - min_vertex
        span = jnp.array(span.max(), span.max(), span.max())
        bunny = (bunny - min_vertex) / span
    return bunny


def sdf_mesh(pt, mesh):
    # computes sdf of mesh (V, 3, 3) to pt (3,)
    # return jnp.min(vmap(lambda tri: sdf_tri(pt, tri))(mesh), axis=0)
    dists = vmap(lambda tri: sdf_tri(pt, tri))(mesh)
    # return jnp.min(dists, axis=0)
    return dists[jnp.argmin(jnp.abs(dists), axis=0)]


def sdf_tri(pt, tri):
    v1, v2, v3 = tri[0, :], tri[1, :], tri[2, :]

    v21, p1 = v2 - v1, pt - v1
    v32, p2 = v3 - v2, pt - v2
    v13, p3 = v1 - v3, pt - v3

    nor = jnp.cross(v21, v13)

    # inside/outside test
    sign_test = (
        jnp.sign(jnp.dot(jnp.cross(v21, nor), p1))
        + jnp.sign(jnp.dot(jnp.cross(v32, nor), p2))
        + jnp.sign(jnp.dot(jnp.cross(v13, nor), p3))
    )

    dot2 = lambda q: jnp.dot(q, q)

    three_edge_case = jax.lax.min(
        jax.lax.min(
            dot2(v21 * jnp.clip(jnp.dot(v21, p1) / dot2(v21), 0.0, 1.0) - p1),
            dot2(v32 * jnp.clip(jnp.dot(v32, p2) / dot2(v32), 0.0, 1.0) - p2),
        ),
        dot2(v13 * jnp.clip(jnp.dot(v13, p3) / dot2(v13), 0.0, 1.0) - p3),
    )

    one_face_case = jnp.dot(nor, p1) * jnp.dot(nor, p1) / dot2(nor)

    # multiply by +- 1 to handle inside or out
    signed_sqrt = lambda a: jnp.sqrt(a) * -jnp.sign(jnp.dot(nor, p1))
    # signed_sqrt = lambda a: jnp.sqrt(a)
    return jax.lax.cond(
        sign_test < 2.0, three_edge_case, signed_sqrt, one_face_case, signed_sqrt
    )


def sdf_mesh_to_grid(sdf, min_range, max_range, resolution):
    xv, yv, zv = jnp.meshgrid(
        jnp.linspace(min_range[0], max_range[0], resolution[0]),
        jnp.linspace(min_range[1], max_range[1], resolution[1]),
        jnp.linspace(min_range[2], max_range[2], resolution[2]),
    )
    grid = jnp.stack((xv, yv, zv), axis=-1)
    grid_sh = grid.shape

    #sdfs = vmap(lambda grid_pt: sdf_mesh(grid_pt, sdf))(grid.reshape(-1, 3))
    sdfs = map_batched(grid.reshape(-1, 3), lambda grid_pt: sdf_mesh(grid_pt, sdf), 512, True)
    sdfs = sdfs.reshape(resolution[0], resolution[1], resolution[2])

    return sdfs


if __name__ == "__main__":
    import cv2

    sdfs = np.array(
        jit(sdf_mesh_to_grid, static_argnums=(3,))(
            get_bunny(), (-2.0, -2.0, -1.0), (2.0, 2.0, 1.0), (32, 32, 32)
        )
    )
    print(sdfs)
    #sdfs = (sdfs - sdfs.min()) / (sdfs.max() - sdfs.min())
    for i in range(32):
        cv2.imshow(
            "sdf",
            cv2.resize(
                sdfs[:, :, i] / 2.0,
                dsize=None,
                fx=16,
                fy=16,
                interpolation=cv2.INTER_NEAREST,
            ),
        )
        cv2.waitKey(0)
