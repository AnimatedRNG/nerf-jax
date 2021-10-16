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
from drawnow import drawnow, figure

from sdrf import IGR, CascadeTree, exp_smin
from util import plot_iso, plot_heatmap, create_mrc
from jax.experimental.host_callback import id_print


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
    visualization_hook=None,
    lr=1e-3,
    batch_size=2 ** 13,
    num_epochs=100000,
    visualization_epochs=200,
):
    params = scene.init(rng, jnp.ones([model_vertices.shape[-1]]), 1.0, 16)
    scene = hk.without_apply_rng(scene)

    scene_fn_multires = lambda params, pt, scale_factor, kern_length: scene.apply(
        params, pt, scale_factor, kern_length
    )[0]

    init_adam, update, get_params = adam(lambda _: lr)
    optimizer_state = init_adam(params)

    def loss_fn(params, scale_factor, kern_length, rng):
        scene_fn = lambda params, pt: scene_fn_multires(
            params, pt, scale_factor, kern_length
        )
        grad_sample_fn = grad(scene_fn, argnums=(1,))

        on_surface_pts = model_vertices[:, :]

        off_surface_pts = jax.random.uniform(
            rng, (batch_size // 2, model_vertices.shape[-1]), minval=-1.0, maxval=1.0
        )

        total_pts = jnp.concatenate((on_surface_pts, off_surface_pts), axis=0)

        def reconstruction_loss_fn(pt, params):
            model_output = scene_fn(params, pt)

            return (model_output ** 2).sum()

        def eikonal_loss_fn(pt, params):
            grad_sample = grad_sample_fn(params, pt)
            # grad_sample = clip_grads(grad_sample[0], 1.0)
            grad_sample = grad_sample[0]

            # grad_sample = jnp.maximum(grad_sample, jnp.ones_like(grad_sample) * 1e-6)
            grad_sample = jnp.where(jnp.abs(grad_sample) < 1e-6, 1e-6, grad_sample)

            return (1.0 - (jnp.linalg.norm(grad_sample))) ** 2.0

        def normal_loss_fn(pt, normal, params):
            sdf = scene_fn(params, pt)
            grad_sample = grad_sample_fn(params, pt)[0]
            grad_sample = jnp.maximum(grad_sample, jnp.ones_like(grad_sample) * 1e-6)

            return ((sdf * (1 - cosine_similarity(grad_sample, normal))) ** 2.0).sum()

        def inter_loss_fn(pt, params):
            sdf = scene_fn(params, pt)
            return (1 - sdf) * jnp.exp(-1e2 * jnp.abs(sdf)).sum()

        reconstruction_losses = vmap(lambda pt: reconstruction_loss_fn(pt, params))(
            on_surface_pts
        ).sum()

        eikonal_losses = vmap(lambda pt: eikonal_loss_fn(pt, params))(total_pts).sum()

        normal_losses = (
            vmap(lambda pt, normal: normal_loss_fn(pt, normal, params))(
                on_surface_pts, model_normals
            ).sum()
            if model_normals is not None
            else 1e-9
        )

        inter_losses = vmap(lambda pt: inter_loss_fn(pt, params))(off_surface_pts).sum()

        # losses = (reconstruction_losses, eikonal_losses, inter_losses)
        # losses = (reconstruction_losses, eikonal_losses, inter_losses)
        losses = (reconstruction_losses, eikonal_losses, normal_losses, inter_losses)

        #weights = (3e3, 1e2, 1e2, 5e1)
        weights = (3e3, 1e2, 1e2, 5e1)
        return (
            sum(loss_level * weight for loss_level, weight in zip(losses, weights)),
            losses,
        )

    value_loss_fn_jit = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True), static_argnums=(2,)
    )
    # value_loss_fn_jit = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)

    figure(figsize=(7, 7 / 2))

    step_size, decay_rate, decay_steps = initial_scale_factor, 0.5, 1000
    for epoch in range(num_epochs):
        rng, subrng = jax.random.split(rng)
        params = get_params(optimizer_state)

        scale_factor = step_size * decay_rate ** (epoch / decay_steps)
        scale_factor = jnp.maximum(scale_factor, 2.0)
        kern_length = int(math.ceil(abs(math.log2(scale_factor)))) + 1
        # kern_length = 32

        if epoch % visualization_epochs == 0:
            scene_fn = lambda params, pt: scene_fn_multires(
                params, pt, scale_factor, kern_length
            )
            drawnow(
                lambda: visualization_hook(
                    scene_fn, model_vertices, model_normals, params
                )
            )

        (loss, losses), gradient = value_loss_fn_jit(
            params, scale_factor, kern_length, subrng
        )
        print(
            f"epoch {epoch}; scale_factor: {scale_factor}; loss {loss}, losses: {losses}"
        )

        # gradient = clip_grads(gradient[0], 1.0)
        gradient = gradient[0]

        optimizer_state = update(epoch, gradient, optimizer_state)
