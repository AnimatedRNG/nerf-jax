#!/usr/bin/env python3

import functools
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.ops import index_update, index_add, index
from jax.tree_util import register_pytree_node

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


class ExponentialSampler(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, rng, num_samples):
        return jnp.concatenate(
            (
                -jnp.exp(jnp.linspace(-10.0, 0.0, num_samples / 2)),
                jnp.exp(jnp.linspace(-10.0, 0.0, num_samples / 2)),
            )
        )

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


def additive_integrator(samples, valid_mask):
    return jnp.sum(samples, axis=-2) / (valid_mask.sum() + 1e-9)


# def render(sampler, sdf, appearance, ro, rd, params, rng, phi, num_samples, additive):
def render(sampler, sdf, appearance, ro, rd, params, rng, phi, options):
    # 1) \phi(d) is sdf-to-density function (almost certainly a Gaussian)
    # 2) We would want to compute the expectation of \phi(d) on a uniform
    #    distribution with many samples. This is slow, so instead we
    #    importance sample \phi(d) on a distribution q(d) that is very similar
    #    to it
    xs = sampler.sample(rng, options.num_samples)

    intensity = lambda pt: appearance(pt, rd, params.appearance)
    depth = lambda pt: jnp.linalg.norm(pt - ro, ord=2, axis=-1, keepdims=True)
    attribs = (intensity, depth)
    intersect = lambda iso: sphere_trace(
        sdf, ro, rd, iso, options.truncation_distance, params.geometry
    )

    # mask out the isosurfaces that don't intersect with the ray
    pts = vmap(intersect)(xs)
    error = vmap(lambda pt, x: jnp.abs(sdf(pt, params.geometry) - x))(pts, xs)
    valid_mask = error < 1e-2
    num_valid_samples = valid_mask.sum()

    if options.additive:
        return tuple(
            jax.lax.select(
                num_valid_samples != 0,
                additive_integrator(attrib, valid_mask),
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
    else:
        depths = vmap(depth)(pts)
        # depths = index_update(depths, valid_mask, -1.0)
        depths = vmap(
            lambda depth, valid: jax.lax.select(valid, depth, -jnp.ones_like(depth))
        )(depths, valid_mask)
        inds = jnp.argsort(depths, axis=-2)

        sorted_xs = jnp.take_along_axis(xs, inds[:, 0], axis=0)
        sorted_valids = jnp.take_along_axis(valid_mask, inds, axis=0)
        sorted_depths = jnp.take_along_axis(depths, inds, axis=-2)
        sorted_pts = jnp.take_along_axis(pts, inds, axis=-2)
        hs = -sorted_depths
        hs = index_add(hs, index[:-1], sorted_depths[1:])
        hs = index_update(hs, index[-1], hs[-2])
        # hs = jnp.ones(options.num_samples) * (1.0 / num_valid_samples)
        os = vmap(lambda x, h, valid: valid * phi(x) * h)(sorted_xs, hs, sorted_valids)
        os = jnp.cumsum(os, axis=0)

        vs = vmap(
            lambda x, h, pt, o, valid: tuple(
                valid * phi(x) * attrib(pt) * jnp.exp(-o) * h for attrib in attribs
            )
        )(sorted_xs, hs, sorted_pts, os, sorted_valids)

        return tuple(
            jax.lax.select(
                num_valid_samples != 0,
                jnp.sum(attrib, axis=-2),
                jnp.zeros(attrib.shape[-1]),
            )
            for attrib in vs
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


SDRFParams = namedtuple("SDRFParams", ["geometry", "appearance"])
register_pytree_node(
    SDRFParams, lambda xs: (tuple(xs), None), lambda _, xs: SDRFParams(*xs)
)

SDRF = namedtuple("SDRF", ["geometry", "appearance"])
register_pytree_node(SDRF, lambda xs: (tuple(xs), None), lambda _, xs: SDRF(*xs))
