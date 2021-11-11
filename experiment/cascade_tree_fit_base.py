import os
import sys
import functools
import math

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam, sgd
from jax import grad, vjp, vmap
import numpy as np
from drawnow import drawnow, figure
import matplotlib.pyplot as plt

from sdrf import IGR, PointCloudSDF, exp_smin
from util import plot_iso, plot_heatmap, create_mrc
from jax.experimental.host_callback import id_print


def make_contour_plot(array_2d, mode="log", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    else:
        fig = plt.gcf()

    if mode == "log":
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1.0 * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0.0, 1.0, num=num_levels * 2 + 1))
    elif mode == "lin":
        num_levels = 10
        levels = np.linspace(-0.5, 0.5, num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0.0, 1.0, num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    fig.colorbar(CS, ax=ax)

    ax.contour(sample, levels=levels, colors="k", linewidths=0.1)
    ax.contour(sample, levels=[0], colors="k", linewidths=0.3)
    ax.axis("off")
    return fig


def get_normals(model_vertices, faces):
    tris = model_vertices[faces]

    normalize = lambda pt: pt / jnp.linalg.norm(pt)
    n = jnp.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = jax.vmap(normalize)(n)

    model_normals = jnp.zeros(model_vertices.shape)
    model_normals = jax.ops.index_add(model_normals, faces[:, 0], n)
    model_normals = jax.ops.index_add(model_normals, faces[:, 1], n)
    model_normals = jax.ops.index_add(model_normals, faces[:, 2], n)
    model_normals = jax.vmap(normalize)(model_normals)

    return model_vertices


def cosine_similarity(a, b, eps=1e-8):
    return jnp.dot(a, b) / jnp.maximum((jnp.linalg.norm(a) * jnp.linalg.norm(b)), eps)


def fit(
    feature_grid,
    rng,
    initial_scale_factor,
    model_vertices,
    model_normals=None,
    eikonal_fn=None,
    visualization_hook=None,
    lr=1e-3,
    batch_size=2 ** 13,
    num_epochs=100000,
    visualization_epochs=200,
):
    params = feature_grid.init(rng, jnp.ones([model_vertices.shape[-1]]), 1.0)
    scene = hk.without_apply_rng(feature_grid)

    init_adam, update, get_params = adam(lambda _: lr)
    optimizer_state = init_adam(params)

    def loss_fn(pts, off_surface_pts, sdfs, params, scale_factor, rng):
        scene_fn = lambda params, pts: scene.apply(params, pts, scale_factor)

        def reconstruction_loss_fn(pt, sdf, params):
            model_output = scene_fn(params, pt)

            return jnp.sum((model_output - sdf) ** 2)

        def eikonal_loss_fn(pt, params):
            samples, grad_fn = vjp(lambda pts: scene_fn(params, pts), pts)
            grad_sample = grad_fn(jnp.ones_like(samples))[0]

            grad_sample = jnp.where(jnp.abs(grad_sample) < 1e-6, 1e-6, grad_sample)

            return (1.0 - (jnp.linalg.norm(grad_sample))) ** 2.0

        def laplacian_loss_fn(pt, params):
            model_output = scene_fn(params, pt)
            return (1 - model_output) * jnp.exp(-1e2 * jnp.abs(model_output))

        reconstruction_losses = reconstruction_loss_fn(pts, sdfs, params).sum()

        eikonal_losses = eikonal_loss_fn(pts, params).sum()

        laplacian_losses = laplacian_loss_fn(pts, params).sum()

        # losses = (reconstruction_losses, normal_losses)
        losses = (reconstruction_losses, eikonal_losses, laplacian_losses)

        # weights = (3e3, 1e2, 1e2, 5e1)
        weights = (3e3, 1e2, 5e1)
        # weights = (1e-9, 1e-9, 5e1)
        return (
            sum(loss_level * weight for loss_level, weight in zip(losses, weights)),
            losses,
        )

    value_loss_fn_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(3,), has_aux=True))
    # value_loss_fn_jit = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)

    # figure(figsize=(7, 7 / 2))

    pts_all = np.array(model_vertices)
    # off_surface_pts_all = pts_all + np.random.laplace(scale=2e-1, size=pts_all.shape)
    off_surface_pts_all = np.random.uniform(low=-1.0, high=1.0, size=pts_all.shape)
    sdfs_all = jnp.zeros(pts_all.shape[:-1])

    step_size, decay_rate, decay_steps = initial_scale_factor, 0.9, 3000
    for epoch in range(num_epochs):
        rng, subrng_0, subrng_1 = jax.random.split(rng, 3)
        params = get_params(optimizer_state)

        indices = jax.random.randint(
            subrng_0, (batch_size,), minval=0, maxval=pts_all.shape[0]
        )
        pts = pts_all[indices]
        off_surface_pts = off_surface_pts_all[indices]
        sdfs = sdfs_all[indices]

        scale_factor = step_size * decay_rate ** (epoch / decay_steps)

        if epoch % visualization_epochs == 0:
            scene_fn = lambda params, pt: scene.apply(params, pt, scale_factor)
            drawnow(
                lambda: visualization_hook(
                    scene_fn, model_vertices, model_normals, params
                )
            )

        (loss, losses), gradient = value_loss_fn_jit(
            jnp.array(pts),
            off_surface_pts,
            jnp.array(sdfs),
            params,
            scale_factor,
            subrng_1,
        )
        print(
            f"epoch {epoch}; scale_factor: {scale_factor}; loss {loss}, losses: {losses}"
        )

        # gradient = clip_grads(gradient[0], 1.0)
        gradient = gradient[0]

        optimizer_state = update(epoch, gradient, optimizer_state)
