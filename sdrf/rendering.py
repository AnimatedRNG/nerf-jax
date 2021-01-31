#!/usr/bin/env python3

import functools
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, vjp
from jax.ops import index_update, index_add, index
from jax.tree_util import register_pytree_node, tree_map
from jax.experimental.host_callback import id_tap, id_print

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
def integrate(sdf, uv, ro, rd, valid_mask, depths, xs, attribs, phi_x, params, options):
    return integrate_fwd(
        sdf, uv, ro, rd, valid_mask, depths, xs, attribs, phi_x, params, options
    )[0]


def integrate_fwd(
    sdf, uv, ro, rd, valid_mask, depths, xs, attribs, phi_x, params, options
):
    # Convert the ray depths from earlier into points
    pts = vmap(lambda depth: ro + rd * depth)(depths)

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
            uv,
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
            options,
        ),
    )


def integrate_rev(sdf, res, rendered_attrib_g):
    # JAX's autodiff seems to run into the NaN issue with the above code,
    # so here we just write out the derivative by hand
    (
        uv,
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
        options,
    ) = res

    if options.oryx_debug:
        import oryx
        import oryx.core as core

    trace = (
        lambda name, attrib, tag: core.sow(attrib, name=name, tag=tag, mode="append")
        if options.oryx_debug
        else attrib
    )

    # trace gradient in
    rendered_attrib_g = tuple(
        trace(f"rendered_attrib_g_{idx}", rendered_attrib, "vjp")
        for idx, rendered_attrib in enumerate(rendered_attrib_g)
    )

    sorted_pts = trace(
        "sorted_pts",
        vmap(lambda sorted_depth: ro + rd * sorted_depth)(sorted_depths),
        "vjp",
    )

    grad_sorted_depths = jnp.zeros_like(sorted_depths)
    grad_sorted_attribs = list(
        jnp.zeros_like(sorted_attrib) for sorted_attrib in sorted_attribs
    )
    grad_phi_x = jnp.zeros_like(sorted_phi_x)

    exp_os = vmap(lambda o: jnp.exp(-1 * o))(os)

    def diff_vjp(inners, adjoint):
        if isinstance(adjoint, (tuple, list)):
            return vmap(
                lambda inner: adjoint[sorted_attrib_index].reshape(1, -1) @ inner
            )(inners)
        else:
            return vmap(lambda a, inner: a.reshape(1, -1) @ inner)(adjoint, inners)

    for sorted_attrib_index, sorted_attrib in enumerate(sorted_attribs):
        trace_i = lambda name, attrib: trace(
            f"{name}_{sorted_attrib_index}", attrib, "vjp"
        )

        dvsddepth_i = trace_i(
            "dvsddepth_i",
            diff_vjp(
                -1 * sorted_attrib[:-1] * sorted_phi_x[:-1] * valid_steps * exp_os,
                rendered_attrib_g,
            ),
        )
        dvsddepth_i1 = trace_i("dvsddepth_i1", -1 * dvsddepth_i)

        dvsdattr_shape = (hs.shape[0], sorted_attribs[sorted_attrib_index].shape[-1])
        dvsdattr = trace_i(
            "dvsdattr",
            diff_vjp(
                vmap(
                    lambda diag: jnp.eye(sorted_attribs[sorted_attrib_index].shape[-1])
                    * diag
                )(sorted_phi_x[:-1] * valid_steps * hs * exp_os),
                rendered_attrib_g,
            )[:, 0, :],
        )

        dvsdphi = trace_i(
            "dvsdphi",
            diff_vjp(
                sorted_attribs[sorted_attrib_index][:-1] * valid_steps * hs * exp_os,
                rendered_attrib_g,
            ),
        )

        dvsdos = trace_i(
            "dvsdos",
            diff_vjp(-vs[sorted_attrib_index], rendered_attrib_g),
        )

        grad_sorted_depths = index_add(grad_sorted_depths, index[0], dvsddepth_i[0])
        grad_sorted_depths = index_add(grad_sorted_depths, index[-1], dvsddepth_i1[-1])
        grad_sorted_attribs[sorted_attrib_index] = index_add(
            grad_sorted_attribs[sorted_attrib_index], index[:-1], dvsdattr
        )
        grad_phi_x = index_add(grad_phi_x, index[:-1], dvsdphi)

        incoming_adjoints_os = trace_i(
            "incoming_adjoints_os", jnp.cumsum(dvsdos[::-1], axis=0)[::-1]
        )

        dosddepth_i = trace_i(
            "dosddepth_i",
            diff_vjp(-1.0 * sorted_phi_x[:-1] * valid_steps, incoming_adjoints_os),
        )
        dosddepth_i1 = trace_i("dosddepth_i1", -dosddepth_i)

        dosdphi = trace_i("dosdphi", diff_vjp(-1.0 * hs, incoming_adjoints_os))

        grad_sorted_depths = index_add(grad_sorted_depths, index[0], dosddepth_i[0])
        grad_sorted_depths = index_add(grad_sorted_depths, index[-1], dosddepth_i1[-1])

        grad_phi_x = index_add(grad_phi_x, index[:-1], dosdphi)

    # unsort
    grad_depths = jnp.take_along_axis(grad_sorted_depths, inds, axis=0)
    grad_attribs = tuple(
        jnp.take_along_axis(grad_sorted_attrib, inds, axis=0)
        for grad_sorted_attrib in grad_sorted_attribs
    )
    grad_phi_x = jnp.take_along_axis(grad_phi_x, inds, axis=0)

    return (
        None,
        None,
        None,
        None,
        grad_depths,
        None,
        grad_attribs,
        grad_phi_x,
        None,
        None,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def masked_sdf(sdf, valid, pt, params):
    return masked_sdf_fwd(sdf, valid, pt, params)[0]


def masked_sdf_fwd(sdf, valid, pt, params):
    return (
        valid * sdf(pt, params.geometry),
        (valid, pt, params),
    )


def masked_sdf_rev(sdf, res, g):
    valid, pt, params = res

    _, vjp_fun = vjp(lambda p, ps: sdf(p, ps.geometry), pt, params)

    grads_input = vjp_fun(g)

    grads_input = tuple(
        tree_map(
            lambda param: jax.lax.select(valid, param, jnp.zeros_like(param)),
            grad_input,
        )
        for grad_input in grads_input
    )

    return (None, *grads_input)


def render(sampler, sdf, appearance, uv, ro, rd, params, rng, phi, options):
    if options.oryx_debug:
        import oryx
        import oryx.core as core

    xs, depths = find_intersections(sampler, sdf, ro, rd, params, rng, options)

    pts = vmap(lambda depth: ro + rd * depth)(depths)

    intensity = lambda pt: appearance(pt, rd, params.appearance)
    depth = lambda depth: depth

    # Mask that determines if a given sphere tracing attempt had
    # suceeded or failed
    error = vmap(lambda pt, x: jnp.abs(sdf(pt, params.geometry) - x))(pts, xs)
    valid_mask = error < 1e-2
    valid_mask = jnp.reshape(valid_mask, (-1, 1))

    phi_x = vmap(lambda pt, valid: phi(masked_sdf(sdf, valid, pt, params)))(
        pts, valid_mask[:, 0]
    )
    phi_x = phi_x.reshape(-1, 1)

    attribs = (
        vmap(
            lambda pt, valid: jax.lax.cond(
                valid,
                pt,
                intensity,
                pt,
                lambda pt: jnp.zeros_like(intensity(pt)),
            )
        )(pts, valid_mask[:, 0]),
        vmap(
            lambda depths_i, valid: jax.lax.cond(
                valid,
                depths_i,
                depth,
                depths_i,
                lambda depths_i: jnp.zeros_like(depth(depths_i)),
            )
        )(depths, valid_mask[:, 0]),
    )
    # attribs = (vmap(intensity)(pts), vmap(depth)(depths))

    # fetch the sdf values if requested
    debug_attribs = [depths] if options.isosurfaces_debug else []
    rd = core.sow(rd, name="rd", tag="vjp", mode="append")
    uv = core.sow(uv, name="uv", tag="vjp", mode="append")

    output = (
        list(
            integrate(
                sdf, uv, ro, rd, valid_mask, depths, xs, attribs, phi_x, params, options
            )
        )
        + debug_attribs
    )

    # clamp rgb
    output[0] = jnp.clip(output[0], 0.0, 1.0)
    return output


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


def extract_debug(uv, reaped, height, width):
    trace_length = reaped.shape[-1]
    uvs, debug = (
        uv[..., :2].astype(np.int64).reshape(-1, 2),
        reaped.reshape(-1),
    )
    uvs = vmap(
        lambda uv: jnp.concatenate(
            (
                jnp.repeat(uv[..., jnp.newaxis], trace_length, axis=-1),
                jnp.arange(trace_length)[jnp.newaxis, :],
            ),
            axis=-2,
        )
    )(uvs)
    uvs = jnp.transpose(uvs, (0, 2, 1)).reshape(-1, 3)
    debug_reshaped = jnp.zeros((height * width * trace_length))
    debug_reshaped = jax.ops.index_update(
        debug_reshaped,
        jnp.ravel_multi_index(uvs.T, (height, width, trace_length)),
        debug,
    ).reshape(height, width, trace_length)

    return debug_reshaped


integrate.defvjp(integrate_fwd, integrate_rev)
masked_sdf.defvjp(masked_sdf_fwd, masked_sdf_rev)

SDRFParams = namedtuple("SDRFParams", ["geometry", "appearance"])
register_pytree_node(
    SDRFParams, lambda xs: (tuple(xs), None), lambda _, xs: SDRFParams(*xs)
)

SDRF = namedtuple("SDRF", ["geometry", "appearance"])
register_pytree_node(SDRF, lambda xs: (tuple(xs), None), lambda _, xs: SDRF(*xs))
