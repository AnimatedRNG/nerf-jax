#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

from .root_finding import sphere_trace


def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
        (-1 / 2) * jnp.square((x - mu) / (sigma))
    )


def stratified_render(sdf, appearance, ro, rd, params, rng, sigma, num_samples):
    # 1) Create discrete Gaussian kernel, with `num_samples` bins
    # 2) Sample from each bin (uniform distribution)
    # 3) Weighted sum of samples with previously-mentioned Gaussian kernel
    pass


def importance_sample_render(
    sdf, appearance, ro, rd, params, rng, phi, sigma, num_samples
):
    # 1) \phi(d) is sdf-to-density function (almost certainly a Gaussian)
    # 2) We would want to compute the expectation of \phi(d) on a uniform
    #    distribution with many samples. This is slow, so instead we
    #    importance sample \phi(d) on a distribution q(d) that is very similar
    #    to it
    xs = jax.random.normal(rng, num_samples) * sigma
    intensity = lambda x: appearance(x, rd)
    intersect = lambda iso: sphere_trace(sdf, ro, rd, iso, params)
    return (
        jnp.sum(
            vmap(
                lambda x: intensity(x) * (phi(intersect(x)) / gaussian_pdf(x, sigma)),
                xs,
            )
        )
        / num_samples
    )
