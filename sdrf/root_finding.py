#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, vjp, defvjp
from jax.tree_util import tree_map
from jax.lax import while_loop

from util import dvmap_while


@functools.partial(jit, static_argnums=(0, 3))
def sphere_trace_naive(sdf, ro, rd, iterations, truncation, *params):
    p = ro
    for _ in range(iterations):
        dist = sdf(p, *params)
        dist = jnp.minimum(jnp.abs(dist), truncation) * jnp.sign(dist)
        p = p + dist * rd
    return p


def sphere_trace(sdf, ro, rd, iso, truncation, *params):
    depth_output = sphere_trace_depth(sdf, ro, rd, iso, truncation, *params)
    return depth_output * rd + ro


def sphere_trace_batched(sdf, ro, rd, iso, truncation, *params):
    depth_output = sphere_trace_depth_batched(sdf, ro, rd, iso, truncation, *params)
    return vmap(lambda d_i, rd_i, ro_i: d_i * rd_i + ro_i)(
        depth_output,
        rd,
        ro,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def sphere_trace_depth_batched(sdf, ro, rd, iso, truncation, *params):
    scalarize = lambda x: x.sum()

    def cond_fun(carry):
        dist, _, _, _, _, iso_i = carry
        abs_dist = scalarize(jnp.abs(dist - iso_i))
        return (abs_dist < 5.0) & (abs_dist > 1e-3)

    def body_fun(carry):
        old_dist, old_depth, old_pt, rd_i, ro_i, iso_i = carry
        depth = scalarize((old_depth + old_dist))
        pt = depth * rd_i + ro_i
        dist = sdf(pt, *params) - iso_i
        #dist = jnp.minimum(jnp.abs(dist), truncation) * jnp.sign(dist)
        return (dist, depth, pt, rd_i, ro_i, iso_i)

    _, depth, _, _, _, _ = dvmap_while(
        cond_fun,
        body_fun,
        (
            vmap(lambda ro_i, iso_i: sdf(ro_i, *params) - iso_i)(ro, iso),
            jnp.zeros(
                ro.shape[0],
            ),
            ro,
            rd,
            ro,
            iso,
        ),
        max_iters=30,
        num_segments=10,
        use_dvmap=True
    )

    return depth


def sphere_trace_depth_batched_fwd(sdf, ro, rd, iso, truncation, *params):
    depth = sphere_trace_depth_batched(sdf, ro, rd, iso, truncation, *params)

    return depth, (ro, rd, iso, truncation, depth, *params)


def sphere_trace_depth_batched_rev(sdf, res, g):
    # TODO: replace with dvmap masked version for even better performance?
    ro, rd, iso, truncation, depth, *params = res
    # for some reason this doesn't work?
    # rev_fn = lambda ro_i, rd_i, iso_i, depth_i, g_i: sphere_trace_depth_rev_paper(
    #    sdf, (ro_i, rd_i, iso_i, truncation, depth_i, *params), g_i
    # )
    # return vmap(rev_fn)(ro, rd, iso, depth, g)

    pts = vmap(lambda depth_i, rd_i, ro_i: depth_i * rd_i + ro_i)(depth, rd, ro)

    vjp_p = vmap(lambda pt: grad(sdf, argnums=(0,))(pt, *params))

    dps = vjp_p(pts)[0]

    u = vmap(lambda dp_i, rd_i, g_i: (-1.0 / (dp_i @ rd_i.T)) * g_i)(dps, rd, g)

    validity = vmap(lambda pt, iso_i: jnp.abs(sdf(pt, *params)) - iso_i < 1e-3)(
        pts, iso
    )

    # batched_sdf = lambda pts, *params_: vmap(
    #    lambda pt, v: jax.lax.select(v, sdf(pt, *params), 1e10)
    # )(pts, validity)
    batched_sdf = lambda pts, *params_: vmap(lambda pt: sdf(pt, *params))(pts)
    _, vjp_params = vjp(functools.partial(batched_sdf, pts), *params)

    return (None, None, None, None, *vjp_params(u))


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
        cond_fun, body_fun, (sdf(ro, *params) - iso, 0.0, 0, ro)
    )

    return depth


def sphere_trace_depth_fwd(sdf, ro, rd, iso, truncation, *params):
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
    """out_vjp_params = [
        tree_map(
            lambda param: jax.lax.select(validity, param, jnp.zeros_like(param)),
            vjp_param,
        )
        for vjp_param in vjp_params(u)
    ]"""
    return (None, None, None, None, *vjp_params(u))
    # return (None, None, None, None, *out_vjp_params)


sphere_trace_depth.defvjp(sphere_trace_depth_fwd, sphere_trace_depth_rev_paper)
sphere_trace_depth_batched.defvjp(
    sphere_trace_depth_batched_fwd, sphere_trace_depth_batched_rev
)


if __name__ == "__main__":
    pass
