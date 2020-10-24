import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from sdrf import sphere_trace_naive, sphere_trace


def create_sphere(pt, origin=jnp.array([0.0, 0.0, 0.0]), radius=2.0):
    return jnp.linalg.norm(pt - origin, ord=2) - radius


def test_sphere_trace_naive():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    params = [jnp.array([0.0, 0.0, 0.0]), jnp.array(2.0)]

    pt = sphere_trace_naive(create_sphere, ro, rd, 5, *params)

    assert abs(jnp.linalg.norm(pt, ord=2) - 2.0) < 1e-3

    radius_fwd = lambda origin, radius: jnp.linalg.norm(
        sphere_trace_naive(create_sphere, ro, rd, 20, origin, radius), ord=2
    )

    target_radius = 3
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


def test_sphere_trace():
    ro = jnp.array([-4.0, 0.0, -1.0])
    rd = jnp.array([1.0, 0.0, 0.0])

    params = [jnp.array([0.0, 0.0, 0.0]), jnp.array(2.0)]

    pt = sphere_trace(create_sphere, ro, rd, *params)

    assert abs(jnp.linalg.norm(pt, ord=2) - 2.0) < 1e-3

    # test as above

    radius_fwd = lambda origin, radius: jnp.linalg.norm(
        sphere_trace(create_sphere, ro, rd, origin, radius), ord=2
    )

    target_radius = 3
    lr = jnp.array(1e-1)

    grad_program = jit(
        grad(
            lambda origin, radius: (target_radius - radius_fwd(origin, radius)) ** 2,
            argnums=(1,),
        )
    )

    for i in range(100):
        gradient = grad_program(params[0], params[1])[0]
        params[1] = params[1] - gradient * lr
        #print(f"gradient on iteration {i}: {gradient}. Radius: {params[1]}")

    assert abs(radius_fwd(*params) - target_radius) < 1e-3
