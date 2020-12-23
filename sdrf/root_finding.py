#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, vjp, defvjp
from jax.tree_util import tree_map
from jax.lax import while_loop


@functools.partial(jit, static_argnums=(0, 3))
def sphere_trace_naive(sdf, ro, rd, iterations, truncation, *params):
    p = ro
    for _ in range(iterations):
        dist = sdf(p, *params)
        dist = jnp.minimum(jnp.abs(dist), truncation) * jnp.sign(dist)
        p = p + dist * rd
    return p


def sphere_trace(sdf, ro, rd, iso, truncation, *params):
    return sphere_trace_depth(sdf, ro, rd, iso, truncation, *params) * rd + ro


# @functools.partial(jit, static_argnums=(0,))
@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def sphere_trace_depth(sdf, ro, rd, iso, truncation, *params):
    scalarize = lambda x: x.sum()

    def cond_fun(carry):
        dist, _, iteration, _ = carry
        abs_dist = scalarize(jnp.abs(dist - iso))
        return (iteration < 30) & (abs_dist < 1e10) & (abs_dist > 1e-3)

    def body_fun(carry):
        old_dist, old_depth, iteration, old_pt = carry
        depth = scalarize((old_depth + old_dist))
        pt = depth * rd + ro
        dist = sdf(pt, *params) - iso
        dist = jnp.minimum(jnp.abs(dist), truncation) * jnp.sign(dist)
        return (dist, depth, iteration + 1, pt)

    _, depth, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, (sdf(ro, *params), 0.0, 0, ro)
    )

    return depth


def sphere_trace_depth_fwd(sdf, ro, rd, iso, truncation, *params):
    # why doesn't this just set the isosurface?
    depth = sphere_trace_depth(sdf, ro, rd, iso, truncation, *params)

    # return depth, (depth, *params)
    return depth, (ro, rd, iso, truncation, depth, *params)


# as described in (Niemeyer, et al. 2020)
def sphere_trace_depth_rev_paper(sdf, res, g):
    ro, rd, iso, truncation, depth, *params = res

    pt = depth * rd + ro

    vjp_p = grad(sdf, argnums=(0,))

    dp = vjp_p(pt, *params)[0]

    u = (-1.0 / (dp @ rd.T)) * g

    validity = jnp.abs(sdf(pt, *params) - iso) < 1e-3

    _, vjp_params = vjp(functools.partial(sdf, pt), *params)

    # mask output by validity?
    out_vjp_params = [
        tree_map(
            lambda param: jax.lax.select(validity, param, jnp.zeros_like(param)),
            vjp_param,
        )
        for vjp_param in vjp_params(u)
    ]
    # return (None, None, None, None, *vjp_params(u))
    return (None, None, None, None, *out_vjp_params)


# TODO: rephrase as a jvp at some point?
# not really benefitting from the `vjp_p` call, since adjoints are
# different...
def sphere_trace_depth_rev_single(sdf, res, g):
    ro, rd, iso, truncation, depth, *params = res

    pt = depth * rd + ro

    _, vjp_p = vjp(sdf, pt, *params)

    dp, *dtheta = vjp_p(jnp.ones(()))

    u = (-1.0 / (dp @ rd.T)) * g

    return (
        None,
        None,
        None,
        *tuple(tree_map(lambda x: u * x, dparam_tree) for dparam_tree in dtheta),
    )


sphere_trace_depth.defvjp(sphere_trace_depth_fwd, sphere_trace_depth_rev_paper)


if __name__ == "__main__":
    pass
