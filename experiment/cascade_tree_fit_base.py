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
from jax import grad, vmap
import numpy as np
from drawnow import drawnow, figure
import matplotlib.pyplot as plt

from sdrf import IGR, CascadeTree, PointCloudSDF, exp_smin
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
    scene,
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
    params = scene.init(rng, jnp.ones([model_vertices.shape[-1]]), 1.0)
    scene = hk.without_apply_rng(scene)

    scene_fn_multires = lambda params, pt, scale_factor: scene.apply(
        params, pt, scale_factor
    )

    init_adam, update, get_params = adam(lambda _: lr)
    optimizer_state = init_adam(params)

    gt_sdf = PointCloudSDF(np.array(model_vertices), np.array(model_normals))

    def loss_fn(pts, sdfs, params, scale_factor, rng):
        scene_fn = lambda params, pt: scene_fn_multires(params, pt, scale_factor)[0]

        def reconstruction_loss_fn(pt, sdf, params):
            model_outputs = scene_fn(params, pt)
            print("num model outputs", len(model_outputs))

            return sum(
                jnp.sum((model_output - sdf) ** 2) for model_output in model_outputs
            )

        reconstruction_losses = vmap(
            lambda pt, sdf: reconstruction_loss_fn(pt, sdf, params)
        )(pts, sdfs).sum()

        """normal_losses = (
            vmap(lambda pt, normal: normal_loss_fn(pt, normal, params))(
                on_surface_pts, model_normals
            ).sum()
            if model_normals is not None
            else 1e-9
        )"""

        # losses = (reconstruction_losses, normal_losses)
        losses = (reconstruction_losses,)

        # weights = (3e3, 1e2, 1e2, 5e1)
        weights = (3e3,)
        return (
            sum(loss_level * weight for loss_level, weight in zip(losses, weights)),
            losses,
        )

    value_loss_fn_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(2,), has_aux=True))
    # value_loss_fn_jit = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)

    # figure(figsize=(7, 7 / 2))

    pts_all = np.array(model_vertices)
    pts_all = np.resize(pts_all, (2 ** 20, 3))
    pts_all = pts_all + np.random.laplace(scale=2e-1, size=pts_all.shape)
    print("sampled random points")
    sdfs_all = gt_sdf(pts_all)

    """test vis
    N = 128
    x = np.linspace(-1, 1, N)
    coords = np.stack([arr.flatten() for arr in np.meshgrid(x, x, x)], axis=-1)
    vis_sdf = gt_sdf(coords).reshape(N, N, N)
    make_contour_plot(vis_sdf[:, :, N // 2], mode="log", ax=None)
    plt.show()

    end test vis"""

    sdfs_all = jnp.array(sdfs_all)

    step_size, decay_rate, decay_steps = initial_scale_factor, 0.5, 1000
    for epoch in range(num_epochs):
        rng, subrng_0, subrng_1 = jax.random.split(rng, 3)
        params = get_params(optimizer_state)

        indices = jax.random.randint(
            subrng_0, (batch_size,), minval=0, maxval=pts_all.shape[0]
        )
        pts = pts_all[indices]
        sdfs = sdfs_all[indices]

        scale_factor = step_size * decay_rate ** (epoch / decay_steps)

        if epoch % visualization_epochs == 0:
            scene_fn = lambda params, pt: scene_fn_multires(params, pt, scale_factor)
            drawnow(
                lambda: visualization_hook(
                    scene_fn, model_vertices, model_normals, params
                )
            )

        (loss, losses), gradient = value_loss_fn_jit(
            jnp.array(pts), jnp.array(sdfs), params, scale_factor, subrng_1
        )
        print(
            f"epoch {epoch}; scale_factor: {scale_factor}; loss {loss}, losses: {losses}"
        )

        # gradient = clip_grads(gradient[0], 1.0)
        gradient = gradient[0]

        optimizer_state = update(epoch, gradient, optimizer_state)
