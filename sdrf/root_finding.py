#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, vjp, defvjp
from jax.tree_util import tree_map
from jax.lax import while_loop


@functools.partial(jit, static_argnums=(0, 3))
def sphere_trace_naive(sdf, ro, rd, iterations, *params):
    p = ro
    for _ in range(iterations):
        dist = sdf(p, *params)
        p = p + dist * rd
    return p


def sphere_trace(sdf, ro, rd, *params):
    return sphere_trace_depth(sdf, ro, rd, *params) * rd + ro


# @functools.partial(jit, static_argnums=(0,))
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def sphere_trace_depth(sdf, ro, rd, *params):
    def cond_fun(carry):
        dist, _, iteration, _ = carry
        abs_dist = jnp.abs(dist)
        return (iteration < 30) & (abs_dist < 1e10) & (abs_dist > 1e-3)

    def body_fun(carry):
        old_dist, old_depth, iteration, old_pt = carry
        depth = old_depth + old_dist
        pt = depth * rd + ro
        dist = sdf(pt, *params)
        return (dist, depth, iteration + 1, pt)

    termination_dist, depth, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, (sdf(ro), 0.0, 0, ro)
    )

    return depth


def sphere_trace_depth_fwd(sdf, ro, rd, *params):
    depth = sphere_trace_depth(sdf, ro, rd, *params)

    return depth, (depth, *params)


def sphere_trace_depth_rev(sdf, ro, rd, res, g):
    depth, *params = res

    pt = depth * rd + ro

    vjp_p = grad(sdf, argnums=(0,))

    dp = vjp_p(pt, *params)[0]

    u = (-1.0 / (dp @ rd.T)) * g

    _, vjp_params = vjp(functools.partial(sdf, pt), *params)
    return vjp_params(u)


sphere_trace_depth.defvjp(sphere_trace_depth_fwd, sphere_trace_depth_rev)


if __name__ == "__main__":
    pass
