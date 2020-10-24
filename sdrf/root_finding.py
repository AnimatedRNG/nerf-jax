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


# @functools.partial(jit, static_argnums=(0,))
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def sphere_trace(sdf, ro, rd, *params):
    def cond_fun(carry):
        dist, iteration, _ = carry
        return (iteration < 30) & (dist < 1e10) & (dist > 1e-3)

    def body_fun(carry):
        old_dist, iteration, old_pt = carry
        pt = old_pt + old_dist * rd
        dist = sdf(pt, *params)
        return (dist, iteration + 1, pt)

    _, _, pt = jax.lax.while_loop(cond_fun, body_fun, (sdf(ro), 0, ro))

    return pt


def sphere_trace_fwd(sdf, ro, rd, *params):
    pt = sphere_trace(sdf, ro, rd, *params)

    return pt, (pt, *params)


def sphere_trace_rev(sdf, ro, rd, res, g):
    pt, *params = res

    # dumb hack to get original dL/dd rather than dL/dp
    dL_dp = (g / rd)[0]

    _, vjp_f = vjp(sdf, pt, *params)

    dp, *dtheta = vjp_f(jnp.ones(()))

    u = (-1.0 / (dp @ rd.T)) * dL_dp

    return tuple(
        tree_map(lambda dparam: rd * u * dparam, dparam_tree) for dparam_tree in dtheta
    )


sphere_trace.defvjp(sphere_trace_fwd, sphere_trace_rev)


if __name__ == "__main__":
    pass
