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
from .root_finding import sphere_trace, sphere_trace_depth


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


def additive_integrator(samples, valid_mask, normalization):
    if normalization is not None:
        integrated = jnp.sum(samples, axis=-2) / (normalization + 1e-9)
    else:
        integrated = jnp.sum(samples, axis=-2)

    num_valid_samples = valid_mask.sum()

    return jax.lax.select(
        num_valid_samples != 0, integrated, jnp.zeros(samples.shape[-1]),
    )


def find_intersections(sampler, sdf, ro, rd, params, rng, options):
    xs = sampler.sample(rng, options.num_samples)
    intersect = lambda iso: sphere_trace_depth(
        sdf, ro, rd, iso, options.truncation_distance, params.geometry
    )

    depths = vmap(intersect)(xs)
    depths = jnp.reshape(depths, (-1, 1))

    return xs, depths


def integrate(sdf, ro, rd, depths, xs, attribs, phi, params, options):
    # Convert the ray depths from earlier into points
    pts = vmap(lambda depth: ro + rd * depth)(depths)

    # Mask that determines if a given sphere tracing attempt had
    # suceeded or failed
    error = vmap(lambda pt, x: jnp.abs(sdf(pt, params.geometry) - x))(pts, xs)
    valid_mask = error < 1e-2
    valid_mask = jnp.reshape(valid_mask, (-1, 1))

    # Now sort the depths along the ray. The invalid samples are included
    # here, but we can remove them using the valid mask. Also, they'll
    # probably be large given that the root finding didn't converge.
    inds = jnp.argsort(depths, axis=0)

    # Re-order all of the inputs to this function based on depth
    sorted_xs = jnp.take_along_axis(xs, inds[:, 0], axis=0)
    sorted_valids = jnp.take_along_axis(valid_mask, inds, axis=0)
    sorted_depths = jnp.take_along_axis(depths, inds, axis=0)
    sorted_pts = jnp.take_along_axis(pts, inds, axis=0)

    # Next, we compute the step size from these depths --
    # h_i = depths_{i + 1} - depths{i}
    # We can do this in-place with the index_add op
    hs = -sorted_depths
    hs = index_add(hs, index[:-1], sorted_depths[1:])
    hs = hs[:-1]

    # A step should be marked as invalid if either endpoint of that segment
    # is invalid
    valid_steps = jnp.logical_and(sorted_valids[:-1], sorted_valids[1:])

    # Now we compute the inner integral, masking out any segments that are
    # invalid. `os` is like an opacity function -- it is like (1 - Transmittance)
    # at any point on the ray. One thing that's awesome is that we don't need
    # to differentiate through `phi(x)`
    os = vmap(lambda x, h, valid: valid * phi(x) * h)(sorted_xs[:-1], hs, valid_steps)
    os = jnp.cumsum(os, axis=0)

    # Finally we compute the rendering integral. `attrib` is usually
    # [rgb, depth, etc].
    vs = vmap(
        lambda x, h, depth, o, valid: tuple(
            valid * phi(x) * attrib(depth) * jnp.exp(-o) * h
            for attrib in attribs
        )
    )(sorted_xs[:-1], hs, sorted_depths[:-1], os, valid_steps)
    rendered_attribs = tuple(jnp.sum(attrib, axis=0) for attrib in vs)

    return rendered_attribs


def render(sampler, sdf, appearance, ro, rd, params, rng, phi, options):
    xs, depths = find_intersections(sampler, sdf, ro, rd, params, rng, options)

    intensity = lambda depth: appearance(ro + rd * depth, rd, params.appearance)
    depth = lambda depth: depth

    attribs = (intensity, depth)

    # fetch the sdf values if requested
    pts = vmap(lambda depth: ro + rd * depth)(depths)
    debug_attribs = (
        (vmap(lambda pt: jnp.abs(sdf(pt, params.geometry)))(pts),)
        if options.debug
        else tuple()
    )

    return (
        integrate(sdf, ro, rd, depths, xs, attribs, phi, params, options)
        + debug_attribs
    )

def render_img(render_fn, rng, ray_bundle, chunksize):
    ro, rd = ray_bundle
    ro_flat, rd_flat = ro.reshape(-1, *ro.shape[2:]), rd.reshape(-1, *rd.shape[2:])
    bundle = jnp.stack((ro_flat, rd_flat), axis=-1)

    attribs, rng = map_batched_rng(
        bundle,
        lambda chunk_rng: render_fn(
            chunk_rng[0][:, 0], chunk_rng[0][:, 1], chunk_rng[1]
        ),
        chunksize,
        True,
        rng,
    )

    # return (rgb.reshape(*ro.shape[:2], -1), depth.reshape(*ro.shape[:2], -1)), rng
    return tuple(attrib.reshape(*ro.shape[:2], -1) for attrib in attribs), rng


SDRFParams = namedtuple("SDRFParams", ["geometry", "appearance"])
register_pytree_node(
    SDRFParams, lambda xs: (tuple(xs), None), lambda _, xs: SDRFParams(*xs)
)

SDRF = namedtuple("SDRF", ["geometry", "appearance"])
register_pytree_node(SDRF, lambda xs: (tuple(xs), None), lambda _, xs: SDRF(*xs))
