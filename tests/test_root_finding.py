import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from sdrf import sphere_trace_naive, sphere_trace


def create_sphere(pt, origin=jnp.array([0.0, 0.0, 0.0]), radius=2.0):
    return jnp.linalg.norm(pt - origin, ord=2) - radius


def perform_radius_test(fn, initial_radius, target_radius):
    params = [jnp.array([0.0, 0.0, 0.0]), jnp.array(initial_radius)]

    radius_fwd = lambda origin, radius: jnp.linalg.norm(fn(origin, radius), ord=2)

    lr = jnp.array(1e-1)

    grad_program = jit(
        grad(
            lambda origin, radius: (target_radius - radius_fwd(origin, radius)) ** 2,
            argnums=(1,),
        )
    )

    for i in range(100):
        params[1] = params[1] - grad_program(params[0], params[1])[0] * lr

    assert abs(radius_fwd(*params) - target_radius) < 1e-3


def test_sphere_trace_naive():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    truncation_dist = 1.0

    fn = lambda origin, radius: sphere_trace_naive(
        create_sphere, ro, rd, 20, truncation_dist, origin, radius
    )

    perform_radius_test(fn, 2.0, 3.0)


def test_sphere_trace():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    truncation_dist = 1.0

    fn = lambda origin, radius: sphere_trace(
        create_sphere, ro, rd, 0.0, truncation_dist, origin, radius
    )

    perform_radius_test(fn, 2.0, 3.0)


def test_sphere_trace_iso():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    truncation_dist = 1.0

    fn = lambda iso: lambda origin, radius: sphere_trace(
        create_sphere, ro, rd, iso, truncation_dist, origin, radius
    )

    # will re-jit, but that's because of the nested call
    # in general, won't re-jit on different isosurface values
    for i in range(5):
        perform_radius_test(fn(i / 5.0), 2.0, 3.0)


def test_depth_accumulation():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    truncation_dist = 100.0

    def inner(origin, radius):
        num_iters = 8

        xs = jnp.linspace(-1e-2, 1e-2, num_iters)
        pts = vmap(
            lambda iso: sphere_trace(
                create_sphere,
                ro,
                rd,
                iso,
                truncation_dist,
                origin,
                radius,
            )
        )(xs)
        valid = vmap(lambda pt: jnp.abs(create_sphere(pt, origin, radius)) < 1e-2)(pts)
        num_valid = jnp.sum(valid)
        return jnp.sum(
            vmap(lambda pt, valid: valid * pt)(pts, valid), axis=0
        ) / jnp.clip(num_valid, 1)

    perform_radius_test(inner, 2.0, 3.0)
