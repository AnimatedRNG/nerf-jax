import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from sdrf import (
    sphere_trace_naive,
    sphere_trace,
    importance_sample_render,
    render_img,
    gaussian_pdf,
)
from util import get_ray_bundle, look_at


def create_sphere(pt, origin=jnp.array([0.0, 0.0, 0.0]), radius=2.0):
    return jnp.linalg.norm(pt - origin, ord=2) - radius


def test_render_sphere():
    view_matrix = jnp.array(
        np.linalg.inv(
            np.array(
                look_at(
                    jnp.array([-4.0, 0.0, 0.0]),
                    jnp.array([0.0, 0.0, 0.0]),
                    jnp.array([0.0, 1.0, 0.0]),
                )
            )
        )
    )

    # height, width, chunk_size = 32, 32, 8
    height, width, chunk_size = 256, 256, 32
    ro, rd = get_ray_bundle(height, width, 100.0, view_matrix)

    rng = jax.random.PRNGKey(42)

    origin = jnp.array([0.0, 0.0, 0.0])
    radius = jnp.array([3.0])

    # sigma used for importance sampling
    importance_sigma = 1e-3
    phi_sigma = 1e-3

    num_samples = 8

    geometry = lambda x, params: create_sphere(x, *params)

    # surface is solid white
    # appearance = lambda pt, rd: jnp.array([1.0, 1.0, 1.0])

    # Some Lambertian lighting
    light_pos = jnp.array([-8.0, -4.0, 0.0])
    normalize = lambda vec: vec / jnp.linalg.norm(vec, ord=2)
    normals = lambda pt: grad(lambda pt, params: geometry(pt, params)[0], argnums=(0,))(
        pt, (origin, radius)
    )[0]
    distance = lambda pt: jnp.square(jnp.linalg.norm(light_pos - pt, ord=2))
    light_dir = lambda pt: normalize(light_pos - pt)
    diffuse_power = 20.0
    specular_power = 40.0
    specular_hardness = 16
    diffuse = lambda pt: jnp.broadcast_to(
        jnp.clip(
            jnp.dot(light_dir(pt), normals(pt)) * diffuse_power / distance(pt), 0.0, 1.0
        ),
        (3,),
    )
    h = lambda pt, rd: normalize(light_dir(pt) - rd)
    ndoth = lambda pt, rd: jnp.clip(jnp.dot(normals(pt), h(pt, rd)), 0.0, 1.0)
    specular = (
        lambda pt, rd: jnp.power(ndoth(pt, rd), specular_hardness)
        * specular_power
        / distance(pt)
    )
    appearance = lambda pt, rd: diffuse(pt) + specular(pt, rd)

    """phi = lambda dist: gaussian_pdf(
        jnp.maximum(dist, jnp.zeros_like(dist)), 0.0, phi_sigma
    )"""
    phi = lambda dist: gaussian_pdf(dist, 0.0, phi_sigma)

    render_fn = lambda ro, rd, rng: importance_sample_render(
        geometry,
        appearance,
        ro,
        rd,
        (origin, radius),
        rng,
        phi,
        importance_sigma,
        num_samples,
    )

    # with jax.disable_jit():
    (rgb, depth), rng = jit(render_img, static_argnums=(0, 3))(
        render_fn, rng, (ro, rd), chunk_size
    )

    import cv2

    cv2.imshow("rgb", np.array(rgb))
    cv2.imshow("depth", np.array(depth) / 10.0)
    cv2.waitKey(0)
