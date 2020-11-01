#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

SDRF = namedtuple("SDRF", ["geometry", "appearance"])
register_pytree_node(SDRF, lambda xs: (tuple(xs), None), lambda _, xs: SDRF(*xs))


def run_one_iter_of_sdrf(model, params, ray_origins, ray_directions, options, rng):
    # reshape ro/rd
    ro = ray_origins.reshape((-1, 3))
    rd = ray_directions.reshape((-1, 3))

    if options.sdrf.sampler.kind == "linear":
        sampler = LinearSampler(options.sdrf.sampler.linear.support)
    elif options.sdrf.sampler.kind == "stratified":
        sampler = StratifiedSampler(options.sdrf.sampler.stratified.support)
    elif options.sdrf.sampler.kind == "exponential":
        sampler = ExponentialSampler()
    elif options.sdrf.sampler.kind == "gaussian":
        sampler = GaussianSampler(options.sdrf.sampler.gaussian.sigma)
    else:
        raise Exception("Invalid sampler type")

    phi = lambda dist: gaussian_pdf(
        jnp.maximum(dist, jnp.zeros_like(dist)), 0.0, options.sdrf.render.phi.sigma
    )

    render_fn = lambda ro, rd, rng: render(
        sampler,
        model.geometry,
        model.appearance,
        ro,
        rd,
        params.geometry,
        rng,
        phi,
        options.sdrf.render.num_samples,
        options.sdrf.render.additive,
    )

    (rgb, depth, rng) = map_batched_rng(
        jnp.stack((ro, rd), axis=-1),
        lambda chunk_rng: render_fn(
            chunk_rng[0][:, 0], chunk_rng[0][:, 1], chunk_rng[1]
        ),
        options.train.chunksize,
        rng,
    )

    return (rgb, depth)
