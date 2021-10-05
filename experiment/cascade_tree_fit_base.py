import os
import sys
import functools

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import grad, vmap
from drawnow import drawnow, figure

from sdrf import IGR, CascadeTree, exp_smin
from util import plot_iso, plot_heatmap, create_mrc
from jax.experimental.host_callback import id_print


def fit(
    scene,
    model_vertices,
    rng,
    model_normals=None,
    visualization_hook=None,
    lr=1e-3,
    batch_size=2**4, # 2 ** 13,
    num_epochs=100000,
    visualization_epochs=10,
):
    params = scene.init(
        rng,
        jnp.ones(
            [
                model_vertices.shape[-1],
            ]
        ),
    )
    scene = hk.without_apply_rng(scene)
    scene_fn = lambda params, pt: scene.apply(params, pt)[0]

    grad_sample_fn = grad(scene_fn, argnums=(1,))

    init_adam, update, get_params = adam(lambda _: lr)
    optimizer_state = init_adam(params)

    def loss_fn(params, rng):
        on_surface_pts = model_vertices[::100, :]

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

            grad_sample = jnp.maximum(grad_sample, jnp.ones_like(grad_sample)*1e-6)

            return (1.0 - (jnp.linalg.norm(grad_sample))) ** 2.0

        def inter_loss_fn(pt, params):
            sdf = scene_fn(params, pt)
            return (1 - sdf) * jnp.exp(-1e2 * jnp.abs(sdf)).sum()

        reconstruction_losses = vmap(lambda pt: reconstruction_loss_fn(pt, params))(
            on_surface_pts
        ).sum()

        eikonal_losses = vmap(lambda pt: eikonal_loss_fn(pt, params))(total_pts).sum()

        inter_losses = vmap(lambda pt: inter_loss_fn(pt, params))(off_surface_pts).sum()

        # for on_surface_pt in on_surface_pts:
        #     reconstruction_losses = reconstruction_loss_fn(on_surface_pt, params).sum()

        # for total_pt in total_pts:
        #     eikonal_losses = eikonal_loss_fn(total_pt, params).sum()

        # for off_surface_pt in off_surface_pts:
        #     inter_losses = inter_loss_fn(off_surface_pt, params).sum()

        # losses = (reconstruction_losses, eikonal_losses, inter_losses)
        # losses = (reconstruction_losses, eikonal_losses, inter_losses)
        losses = (reconstruction_losses, eikonal_losses, inter_losses)

        weights = (3e3, 1e2, 5e1)
        return (
            sum(loss_level * weight for loss_level, weight in zip(losses, weights)),
            losses,
        )

    # value_loss_fn_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True))
    value_loss_fn_jit = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)

    figure(figsize=(7, 7 / 2))

    for epoch in range(num_epochs):
        rng, subrng = jax.random.split(rng)
        params = get_params(optimizer_state)

        if epoch % visualization_epochs == 0:
            drawnow(lambda: visualization_hook(scene_fn, model_vertices, params))

        (loss, losses), gradient = value_loss_fn_jit(params, subrng)
        print(f"epoch {epoch}; loss {loss}, losses: {losses}")

        # gradient = clip_grads(gradient[0], 1.0)
        gradient = gradient[0]

        optimizer_state = update(epoch, gradient, optimizer_state)
