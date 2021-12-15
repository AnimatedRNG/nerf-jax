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
    SDRF,
    SDRFGrid,
    gaussian_pdf,
    render,
    sphere_trace_depth,
    find_intersections,
    find_intersections_batched,
)
from util import map_batched_tuple, map_batched_rng
from nerf import run_one_iter_of_nerf, volume_render_radiance_field


def geometry_bound(projection_options, sdf):
    # return lambda p: jnp.where(
    #    jnp.linalg.norm(p) > projection_options.far, jnp.zeros_like(sdf(p)), sdf(p)
    # )
    return lambda p: jnp.minimum(
        sdf(p), projection_options.far - jnp.linalg.norm(p, keepdims=True)
    )


def eikonal_loss(sdf, pts):
    # TODO: Rewrite this, this is clunky
    sdf_grad = lambda pt: grad(lambda pt: sdf(pt).sum(), argnums=(0,))(pt)
    grad_samples = vmap(sdf_grad)(pts)[0]
    grad_samples = jnp.where(jnp.abs(grad_samples) < 1e-6, 1e-6, grad_samples)
    return jnp.mean(
        vmap(lambda grad_sample: (1.0 - jnp.linalg.norm(grad_sample)) ** 2.0)(
            grad_samples
        ),
        axis=0,
    )


def manifold_loss(sdf, pts):
    return jnp.mean(
        vmap(lambda pt: jnp.exp(-1e2 * jnp.abs(sdf(pt).sum())))(pts), axis=0
    )


def run_one_iter_of_sdrf(
    sdrf,
    ray_origins,
    ray_directions,
    iteration,
    intrinsics,
    nerf_options,
    sdrf_options,
    projection_options,
    rng,
    validation=False,
):
    rng, *subrng = jax.random.split(rng, 3)

    ro = ray_origins.reshape((-1, 3))
    rd = ray_directions.reshape((-1, 3))

    # the beta term from volsdf
    render_options = sdrf_options.render
    sigma = render_options.phi.initial_sigma * (
        render_options.phi.lr_decay_factor
        ** (iteration / (render_options.phi.lr_decay * 1000))
    )
    sigma = jnp.clip(sigma, a_min=5e-3)

    # stratified sampling
    noise_fn = lambda x: x + jax.random.uniform(
        subrng[0], x.shape, minval=-sigma / 2, maxval=sigma / 2
    )

    # geometry_fn = geometry_bound(projection_options, sdrf.geometry)
    geometry_fn = sdrf.geometry

    def intersect(ro, rd):
        # returns the root for a given isosurface
        def intersect_inner(iso):
            return sphere_trace_depth(
                geometry_fn, ro, rd, noise_fn(iso), render_options.truncation_distance
            )

        # isosurface bins to sample over
        xs = jnp.linspace(-sigma, sigma, sdrf_options.render.num_samples)
        return vmap(intersect_inner)(xs)

    # z_vals -> (num_rays, N_samples)
    z_vals = vmap(intersect)(ro, rd)
    z_vals = jax.lax.sort(z_vals, dimension=-1)
    z_vals = jax.lax.stop_gradient(z_vals)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., None]
    mask = vmap(vmap(sdrf.geometry))(pts) < 1.0

    # radiance model
    def model(pt, view):
        # sigma_pt = sigma + sdrf.ddf(pt)
        sigma_pt = sigma
        volsdf_psi = lambda dist: jax.lax.cond(
            (dist <= 0.0)[0],
            dist,
            lambda x: 0.5 * jnp.exp(x / sigma_pt),
            dist,
            lambda x: 1 - 0.5 * jnp.exp(-x / sigma_pt),
        )
        volsdf_phi = lambda dist: (sigma_pt ** -1) * volsdf_psi(-dist)

        alpha = volsdf_phi(geometry_fn(pt))
        rgb = sdrf.appearance(pt, view)
        return jnp.concatenate((rgb, alpha), axis=-1)

    # get RGB and opacity samples over all the rays, mask out the outliers
    radiance_field = vmap(
        lambda pts_ray, mask_ray, view: vmap(
            lambda pt, m: jnp.where(m, model(pt, view), jnp.zeros_like(model(pt, view)))
        )(pts_ray, mask_ray)
    )(pts, mask, rd)
    rgb, disp, acc, _, _, = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        subrng[1],
        nerf_options.radiance_field_noise_std,
        nerf_options.white_background,
    )

    imgs = (rgb, z_vals)

    if validation:
        imgs = tuple(img.reshape(*ray_directions.shape[:-1], -1) for img in imgs)

    return imgs


def run_one_iter_of_sdrf_nerflike(
    sdrf,
    ray_origins,
    ray_directions,
    iteration,
    intrinsics,
    nerf_options,
    sdrf_options,
    projection_options,
    rng,
    validation=False,
):
    render_options = sdrf_options.render

    def model(pt, view):
        sigma = render_options.phi.initial_sigma * (
            render_options.phi.lr_decay_factor
            ** (iteration / (render_options.phi.lr_decay * 1000))
        )
        # sigma = sigma + sdrf.ddf(pt)
        sigma = sigma
        volsdf_psi = lambda dist: jax.lax.cond(
            (dist <= 0.0)[0],
            dist,
            lambda x: 0.5 * jnp.exp(x / sigma),
            dist,
            lambda x: 1 - 0.5 * jnp.exp(-x / sigma),
        )
        volsdf_phi = lambda dist: (sigma ** -1) * volsdf_psi(-dist)

        alpha = volsdf_phi(sdrf.geometry(pt))
        rgb = sdrf.appearance(pt, view)
        return (rgb, alpha)

    H, W, focal = (
        intrinsics["train"].height,
        intrinsics["train"].width,
        intrinsics["train"].focal_length,
    )

    _, rendered_images = run_one_iter_of_nerf(
        H,
        W,
        focal,
        model,
        model,
        ray_origins,
        ray_directions,
        nerf_options.validation if validation else nerf_options.train,
        nerf_options.model,
        projection_options,
        rng,
        validation,
    )

    return rendered_images


def run_one_iter_of_sdrf_old(
    model,
    params,
    uv,
    ray_origins,
    ray_directions,
    scale_factor,
    iteration,
    options,
    rng,
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

    sdrf_model = model
    rng, subrng = jax.random.split(rng, 2)

    render_fn = lambda uv, ro, rd, xs, depths: render(
        sdrf_model.geometry,
        sdrf_model.appearance,
        uv,
        ro,
        rd,
        xs,
        depths,
        phi,
        sigma,
        options.render,
    )

    intersections, _ = map_batched_tuple(
        (ro, rd),
        lambda ro_, rd_, subrng_: find_intersections(
            sampler,
            sdrf_model.geometry,
            ro_,
            rd_,
            subrng_,
            sigma,
            options.render,
        ),
        options.render.chunksize,
        True,
        rng,
    )
    xs, depths = intersections

    outputs = map_batched_tuple(
        (uv, ro, rd, xs, depths),
        render_fn,
        options.render.chunksize,
        True,
    )

    return outputs
