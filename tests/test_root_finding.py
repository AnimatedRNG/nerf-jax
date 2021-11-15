import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from sdrf import (
    sphere_trace_naive,
    sphere_trace,
    sphere_trace_batched,
)


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
        print(params)

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
        lambda pt: create_sphere(pt, origin, radius),
        ro,
        rd,
        0.0,
        truncation_dist,
    )

    perform_radius_test(fn, 2.0, 3.0)


def test_sphere_trace_iso():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    truncation_dist = 1.0

    fn = lambda iso: lambda origin, radius: sphere_trace(
        lambda pt: create_sphere(pt, origin, radius), ro, rd, iso, truncation_dist
    )

    # will re-jit, but that's because of the nested call
    # in general, won't re-jit on different isosurface values
    for i in range(5):
        perform_radius_test(fn(i / 5.0), 2.0, 3.0)


def test_sphere_trace_batched():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])
    truncation_dist = 1.0

    iso, origin, radius = (
        jnp.array(0.0),
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array(2.0),
    )

    create_permutations = lambda a, n, add_noise=True: jnp.repeat(
        jnp.expand_dims(a, axis=0), n, axis=0
    ) + (
        np.random.random((n,) + a.shape) * 1e-2
        if add_noise
        else np.zeros((n,) + a.shape)
    )

    sphere_trace_fn = lambda ro_, rd_, iso_: sphere_trace(
        create_sphere, ro_, rd_, iso_, truncation_dist, origin, radius
    )

    sphere_trace_batched_fn = lambda ro_, rd_, iso_: sphere_trace_batched(
        create_sphere,
        ro_,
        rd_,
        iso_,
        truncation_dist,
        origin,
        radius,
    )

    ro_perm, rd_perm, iso_perm = (
        create_permutations(ro, 10, False),
        create_permutations(rd, 10, False),
        create_permutations(iso, 10, True),
    )
    assert jnp.allclose(
        vmap(sphere_trace_fn)(ro_perm, rd_perm, iso_perm),
        sphere_trace_batched_fn(ro_perm, rd_perm, iso_perm),
    )

    loss_fn_trace = lambda ro, rd, iso: jnp.sum(
        vmap(
            lambda ro_i, rd_i, iso_i: jnp.linalg.norm(
                sphere_trace_fn(ro_i, rd_i, iso_i)
            )
        )(ro, rd, iso)
    )

    loss_fn_batched = lambda ro, rd, iso: jnp.sum(
        vmap(lambda a_i: jnp.linalg.norm(a_i))(sphere_trace_batched_fn(ro, rd, iso))
    )

    assert jnp.allclose(
        grad(loss_fn_trace)(ro_perm, rd_perm, iso_perm),
        grad(loss_fn_batched)(ro_perm, rd_perm, iso_perm),
    )


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
