#!/usr/bin/env python3

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

from util import map_batched_rng
from .root_finding import sphere_trace


def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
        (-1 / 2) * jnp.square((x - mu) / (sigma))
    )


class GaussianSampler(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, rng, num_samples):
        return jax.random.normal(rng, (num_samples,)) * self.sigma

    def pdf(self, x):
        return gaussian_pdf(x, 0.0, self.sigma)


class LinearSampler(object):
    def __init__(self, support):
        self.support = support

    def sample(self, rng, num_samples):
        return jnp.linspace(-self.support, self.support, num_samples)

    def pdf(self, x):
        return 1.0 / (self.support * 2)


class StratifiedSampler(object):
    def __init__(self, support):
        self.support = support

    def sample(self, rng, num_samples):
        partition_size = (2 * self.support) / num_samples
        return jnp.linspace(
            -self.support, self.support, num_samples
        ) + jax.random.uniform(
            rng, (num_samples,), minval=-partition_size / 2, maxval=partition_size / 2
        )

    def pdf(self, x):
        return 1.0 / (self.support * 2)

def additive_render(
    sampler, sdf, appearance, ro, rd, params, rng, phi, num_samples
):
    # 1) \phi(d) is sdf-to-density function (almost certainly a Gaussian)
    # 2) We would want to compute the expectation of \phi(d) on a uniform
    #    distribution with many samples. This is slow, so instead we
    #    importance sample \phi(d) on a distribution q(d) that is very similar
    #    to it
    xs = sampler.sample(rng, num_samples)

    intensity = lambda pt: appearance(pt, rd)
    depth = lambda pt: jnp.linalg.norm(pt - ro, ord=2, axis=-1, keepdims=True)
    attribs = (intensity, depth)
    intersect = lambda iso: sphere_trace(sdf, ro, rd, iso, params)

    # mask out the isosurfaces that don't intersect with the ray
    pts = vmap(intersect)(xs)
    error = vmap(lambda pt, x: jnp.abs(sdf(pt, params) - x))(pts, xs)
    valid_mask = error < 1e-2
    num_valid_samples = valid_mask.sum()

    return tuple(
        jax.lax.select(
            num_valid_samples != 0,
            jnp.sum(attrib, axis=-2) / num_valid_samples,
            jnp.zeros(attrib.shape[-1]),
        )
        for attrib in vmap(
            lambda x, pt, valid: tuple(
                # should we just use x here rather than resampling?
                valid * attrib(pt) * (phi(x) / sampler.pdf(x))
                for attrib in attribs
            )
        )(xs, pts, valid_mask)
    )


def render_img(render_fn, rng, ray_bundle, chunksize):
    ro, rd = ray_bundle
    ro_flat, rd_flat = ro.reshape(-1, *ro.shape[2:]), rd.reshape(-1, *rd.shape[2:])
    bundle = jnp.stack((ro_flat, rd_flat), axis=-1)

    (rgb, depth), rng = map_batched_rng(
        bundle,
        lambda chunk_rng: render_fn(
            chunk_rng[0][:, 0], chunk_rng[0][:, 1], chunk_rng[1]
        ),
        chunksize,
        True,
        rng,
    )

    return (rgb.reshape(*ro.shape[:2], -1), depth.reshape(*ro.shape[:2], -1)), rng
