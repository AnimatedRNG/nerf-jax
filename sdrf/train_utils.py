#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.experimental.host_callback import id_tap, id_print

from .rendering import (
    LinearSampler,
    StratifiedSampler,
    GaussianSampler,
    ExponentialSampler,
    gaussian_pdf,
    render,
    find_intersections,
    find_intersections_batched,
)
from util import map_batched_tuple, map_batched_rng


def eikonal_loss(sdf, pts, sdf_params):
    # TODO: Rewrite this, this is clunky
    sdf_grad = lambda pt: grad(lambda pt, paras: sdf(pt, paras).sum(), argnums=(0,))(
        pt, sdf_params
    )
    return jnp.mean(
        vmap(lambda pt: (1.0 - jnp.linalg.norm(sdf_grad(pt))) ** 2.0)(pts),
        axis=0,
    )


def manifold_loss(sdf, pts, sdf_params):
    return jnp.mean(
        vmap(lambda pt: jnp.exp(-1e2 * jnp.abs(sdf(pt, sdf_params))))(pts), axis=0
    )


def run_one_iter_of_sdrf(
    model, params, uv, ray_origins, ray_directions, iteration, options, rng
):
    # reshape ro/rd
    ro = ray_origins.reshape((-1, 3))
    rd = ray_directions.reshape((-1, 3))

    if options.sampler.kind == "linear":
        # sampler = LinearSampler(options.sampler.linear.support)
        sampler = LinearSampler()
    elif options.sampler.kind == "stratified":
        # sampler = StratifiedSampler(options.sampler.stratified.support)
        sampler = StratifiedSampler()
    elif options.sampler.kind == "exponential":
        sampler = ExponentialSampler()
    elif options.sampler.kind == "gaussian":
        # sampler = GaussianSampler(options.sampler.gaussian.sigma)
        sampler = GaussianSampler()
    else:
        raise Exception("Invalid sampler type")

    num_decay_steps = options.render.phi.lr_decay * 1000
    sigma = options.render.phi.initial_sigma * options.render.phi.lr_decay_factor ** (
        iteration / num_decay_steps
    )
    phi = lambda dist, s: gaussian_pdf(jnp.maximum(dist, jnp.zeros_like(dist)), 0.0, s)
    # phi = lambda dist, s: gaussian_pdf(dist, 0.0, s)

    render_fn = lambda uv, ro, rd, xs, depths: render(
        model.geometry,
        model.appearance,
        uv,
        ro,
        rd,
        xs,
        depths,
        params,
        phi,
        sigma,
        options.render,
    )

    intersections, _ = map_batched_tuple(
        (ro, rd),
        lambda ro_, rd_, subrng_: find_intersections(
            sampler,
            model.geometry,
            ro_,
            rd_,
            params,
            subrng_,
            sigma,
            options.render,
        ),
        options.render.chunksize,
        True,
        rng,
    )
    '''intersections, _ = map_batched_tuple(
        (ro, rd),
        lambda ro_, rd_, subrng_: find_intersections_batched(
            sampler,
            model.geometry,
            ro_,
            rd_,
            params,
            subrng_,
            sigma,
            options.render,
        ),
        options.render.chunksize,
        False,
        rng,
    )'''
    xs, depths = intersections

    outputs = map_batched_tuple(
        (uv, ro, rd, xs, depths),
        render_fn,
        options.render.chunksize,
        True,
    )

    return outputs
