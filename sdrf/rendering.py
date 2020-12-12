#!/usr/bin/env python3

import functools
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
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


def find_intersections(sampler, sdf, ro, rd, params, rng, options):
    xs = sampler.sample(rng, options.num_samples)
    intersect = lambda iso: sphere_trace_depth(
        sdf, ro, rd, iso, options.truncation_distance, params.geometry
    )

    depths = vmap(intersect)(xs)
    depths = jnp.reshape(depths, (-1, 1))

    return xs, depths


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def integrate(sdf, ro, rd, depths, xs, attribs, phi_x, params, options):
    return integrate_fwd(sdf, ro, rd, depths, xs, attribs, phi_x, params, options)[0]


def integrate_fwd(sdf, ro, rd, depths, xs, attribs, phi_x, params, options):
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
    sorted_attribs = tuple(
        jnp.take_along_axis(attrib, inds, axis=0) for attrib in attribs
    )
    sorted_phi_x = jnp.take_along_axis(phi_x, inds, axis=0)
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
    # at any point on the ray
    os = vmap(lambda phi_x, h, valid: valid * phi_x * h)(
        sorted_phi_x[:-1], hs, valid_steps
    )
    os = jnp.cumsum(os, axis=0)

    # Finally we compute the rendering integral. `attrib` is usually
    # [rgb, depth, etc].
    vs = tuple(
        vmap(
            lambda phi_x, h, attrib, depth, o, valid: valid
            * phi_x
            * attrib
            * jnp.exp(-o)
            * h
        )(
            sorted_phi_x[:-1],
            hs,
            sorted_attrib[:-1],
            sorted_depths[:-1],
            os,
            valid_steps,
        )
        for sorted_attrib in sorted_attribs
    )
    rendered_attribs = tuple(jnp.sum(attrib, axis=0) for attrib in vs)

    return (
        rendered_attribs,
        (
            ro,
            rd,
            inds,
            valid_steps,
            sorted_depths,
            sorted_attribs,
            sorted_phi_x,
            hs,
            os,
            vs,
        ),
    )


def integrate_rev(sdf, res, rendered_attrib_g):
    # JAX's autodiff seems to run into the NaN issue with the above code,
    # so here we just write out the derivative by hand
    (
        ro,
        rd,
        inds,
        valid_steps,
        sorted_depths,
        sorted_attribs,
        sorted_phi_x,
        hs,
        os,
        vs,
    ) = res

    sorted_pts = vmap(lambda sorted_depth: ro + rd * sorted_depth)(sorted_depths)

    grad_sorted_depths = jnp.zeros_like(sorted_depths)
    grad_sorted_attribs = tuple(
        jnp.zeros_like(sorted_attrib) for sorted_attrib in sorted_attribs
    )
    grad_phi_x = jnp.zeros_like(sorted_phi_x)

    exp_os = vmap(lambda o: jnp.exp(-1 * o))(os)

    tuple_diff_vjp = lambda inners, adjoints: (
        vmap(lambda inner: adjoint[:-1].T @ inner)(inners) for adjoint in adjoints
    )

    # TODO: parameterize over sorted_attribs
    dvsddepth_i = tuple_diff_vjp(
        -1 * sorted_attribs[:-1] * sorted_phi_x[:-1] * valid_steps * exp_os,
        rendered_attrib_g,
    )
    dvsddepth_i1 = -1 * dvsddepth_i

    dvsdattr = tuple_diff_vjp(
        sorted_phi_x[:-1] * valid_steps * hs * exp_os, rendered_attrib_g,
    )

    dvsdphi = tuple_diff_vjp(
        sorted_attribs[:-1] * valid_steps * hs * exp_os, rendered_attrib_g
    )

    # dvsdos = diff_vjp(
    #    -1 * sorted_attribs[:-1] * sorted_phi_x[:-1] * valid_steps * hs * exp_os,
    #    rendered_attrib_g,
    # )
    dvsdos = -vs

    # gradient updates for dvs/ddepth_i
    # index_add(grad_sorted_depths, index[:-1], dvsddepth_i)
    grad_sorted_depths = sum(
        index_add(grad_sorted_depths, index[0], dvsddepth_ia)
        for dvsddepth_ia in dvsddepth_i
    )

    # gradient updates for dvs/ddepth_i1
    # index_add(grad_sorted_depths, index[1:], dvsddepth_i1)
    grad_sorted_depths = sum(
        index_add(grad_sorted_depths, index[-1], dvsddepth_i1a)
        for dvsddepth_i1a in dvsddepth_i1
    )

    # gradient updates for dvs/dsorted_attribs
    grad_sorted_attribs = sum(
        index_add(grad_sorted_attribs, index[:-1], dvsdattr_a)
        for dvsdattr_a in dvsdattr
    )

    # gradient update for dvs/dphi_x
    grad_phi_x = sum(
        index_add(grad_phi_x, index[:-1], dvsdphi_xa) for dvsdphi_xa in dvsdphi_x
    )

    # now let's consider the derivatives of os
    # do reverse cumsum for incoming_adjoints_os
    incoming_adjoints_os = tuple(
        jnp.cumsum(dvsdos_a[::-1], axis=0)[::-1] for dvsdos_a in dvsdos
    )

    # then compute dos/ddepth_i and dos/ddepth_i1
    dosdepth_i = tuple(
        vmap(lambda inner, v: v.T @ inner)(
            -1.0 * sorted_phi_x * valid_steps, incoming_adjoints_os_a
        )
        for incoming_adjoints_os_a in incoming_adjoints_os
    )
    dosdepth_i1 = -dosdepth_i

    # then compute dos/dphi
    dosdphi = tuple(
        vmap(lambda inner, v: v.T @ inner)(-1.0 * hs, incoming_adjoints_os)
        for incoming_adjoints_os_a in incoming_adjoints_os
    )

    # now add the contributions for os
    grad_sorted_depths = sum(
        index_add(grad_sorted_depths, index[0], dosddepth_i)
        for dosddepth_ia in dosddepth_i
    )
    grad_sorted_depths = sum(
        index_add(grad_sorted_depths, index[-1], dosddepth_i1)
        for dosddepth_i1a in dosddepth_i1
    )
    grad_phi_x = sum(
        index_add(grad_phi_x, index[:-1], dosdphi) for dosdphi_a in dosdphi
    )

    # unsort
    grad_depths = jnp.take_along_axis(grad_sorted_depths, inds, axis=0)
    grad_attribs = jnp.take_along_axis(grad_sorted_attribs, inds, axis=0)
    grad_phi_x = jnp.take_along_axis(grad_phi_x, inds, axis=0)

    return (None, None, grad_depths, None, grad_attribs, grad_phi_x, None, None)


def render(sampler, sdf, appearance, ro, rd, params, rng, phi, options):
    xs, depths = find_intersections(sampler, sdf, ro, rd, params, rng, options)

    pts = vmap(lambda depth: ro + rd * depth)(depths)

    intensity = lambda pt: appearance(pt, rd, params.appearance)
    depth = lambda depth: depth

    phi_x = vmap(lambda pt: phi(sdf(pt, params.geometry)))(pts)
    phi_x = phi_x.reshape(-1, 1)

    attribs = (vmap(intensity)(pts), vmap(depth)(depths))

    # fetch the sdf values if requested
    debug_attribs = (depths,) if options.debug else tuple()

    return (
        integrate(sdf, ro, rd, depths, xs, attribs, phi_x, params, options)
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


integrate.defvjp(integrate_fwd, integrate_rev)

SDRFParams = namedtuple("SDRFParams", ["geometry", "appearance"])
register_pytree_node(
    SDRFParams, lambda xs: (tuple(xs), None), lambda _, xs: SDRFParams(*xs)
)

SDRF = namedtuple("SDRF", ["geometry", "appearance"])
register_pytree_node(SDRF, lambda xs: (tuple(xs), None), lambda _, xs: SDRF(*xs))
