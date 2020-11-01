#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap


def run_one_iter_of_sdrf(
    model, params, ray_origins, ray_directions, iteration, options, rng
):
    # reshape ro/rd
    ro = ray_origins.reshape((-1, 3))
    rd = ray_directions.reshape((-1, 3))

    if options.sampler.kind == "linear":
        sampler = LinearSampler(options.sampler.linear.support)
    elif options.sampler.kind == "stratified":
        sampler = StratifiedSampler(options.sampler.stratified.support)
    elif options.sampler.kind == "exponential":
        sampler = ExponentialSampler()
    elif options.sampler.kind == "gaussian":
        sampler = GaussianSampler(options.sampler.gaussian.sigma)
    else:
        raise Exception("Invalid sampler type")

    sigma = (
        options.render.phi.initial_sigma
        * options.render.phi.decay_factor
        ** (iteration / options.render.phi.num_decay_steps)
    )
    phi = lambda dist: gaussian_pdf(
        jnp.maximum(dist, jnp.zeros_like(dist)), 0.0, sigma
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
        options.render,
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
