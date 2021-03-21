#!/usr/bin/env python3

import argparse
import pickle
from collections import namedtuple

import numpy as np
import yaml
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.experimental.optimizers import adam, clip_grads
from box import Box
import haiku as hk
import seaborn as sns
import matplotlib.pyplot as plt
from drawnow import drawnow, figure

from sdrf import Siren, IGR
from nerf import FlexibleNeRFModel, NeRFModelMode, positional_encoding


def create_sphere(pt, origin=jnp.array([0.0, 0.0, 0.0]), radius=0.5):
    return jnp.linalg.norm(pt - origin, ord=2) - radius


def get_on_surface_points(num_pts, radius=0.5):
    phi = jnp.linspace(0, jnp.pi, num_pts)
    theta = jnp.linspace(0, 2 * jnp.pi, num_pts)
    return jnp.stack(
        (
            radius * jnp.sin(phi) * jnp.cos(theta),
            radius * jnp.sin(phi) * jnp.sin(theta),
            radius * jnp.cos(phi),
        ),
        axis=-1,
    )


def generate_initialization(config, batch_size, num_epochs, validation_skip, lr, rng):
    model_fn = hk.transform(
        lambda x: FlexibleNeRFModel(
            num_layers=config.network.num_layers,
            hidden_size=config.network.hidden_size,
            skip_connect_every=config.network.skip_connect_every,
            geometric_init=False,
        )(
            positional_encoding(x, config.network.num_encoding_fn_xyz),
            positional_encoding(jnp.zeros_like(x), config.network.num_encoding_fn_dir),
            mode=NeRFModelMode.BOTH,
        )[
            1
        ]
    )
    params = model_fn.init(
        rng,
        jnp.ones(
            [
                3,
            ]
        ),
    )

    model_fn = hk.without_apply_rng(model_fn)

    optimizer = init_adam, update, get_params = adam(lambda _: lr)

    optimizer_state = init_adam((params))

    model_jit_fn = jit(
        lambda params, pts: vmap(lambda pt: model_fn.apply(params, pt)[0])(pts)
    )
    grad_model_fn = grad(lambda params, pt: model_fn.apply(params, pt)[0], argnums=(1,))

    def compute_loss(params, pts):
        def loss_fn(pt):
            dist = create_sphere(pt)
            model_output = model_fn.apply(params, pt)

            grad_output = grad_model_fn(params, pt)

            reconstruction_loss = ((model_output - dist) ** 2).sum()
            eikonal_loss = (1.0 - jnp.linalg.norm(grad_output)) ** 2.0
            inter_loss = jnp.exp(-1e2 * jnp.abs(model_output)).sum()

            return jnp.array([reconstruction_loss, eikonal_loss, inter_loss])

        losses = jnp.mean(vmap(loss_fn)(pts), axis=0)
        # return losses[0] * 3e3 + losses[1] * 5e1 + losses[2] * 1e2, losses
        return losses[0] * 3e3 + losses[1] * 5e1 + losses[2] * 1e1, losses

    value_fn = jit(compute_loss)
    value_and_grad_loss_fn = jit(grad(compute_loss, argnums=(0,), has_aux=True))

    def generate_samples(rng, num_pts=batch_size):
        off_surface_pts = jax.random.uniform(
            rng, (batch_size // 2, 3), minval=-1.0, maxval=1.0
        )
        on_surface_pts = get_on_surface_points(batch_size // 2)
        pts = jnp.concatenate((on_surface_pts, off_surface_pts), axis=0)
        return pts

    # contour_fig, contour_ax = plt.subplots()
    heat_fig, heat_ax = plt.subplots()

    def make_fig():
        pts = jnp.stack(
            jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, 8),
                jnp.linspace(-1.0, 1.0, 8),
                jnp.linspace(-1.0, 1.0, 8),
            ),
            axis=-1,
        )
        grid = pts[:, :, 4, :].reshape(8, 8, 3)
        dists = model_jit_fn(params, grid.reshape(-1, 3)).reshape(8, 8)

        heat_ax = sns.heatmap(
            np.array(dists),
            annot=True,
            fmt=".1f",
            vmin=-0.8,
            vmax=0.8,
            center=0,
            cmap="RdBu_r",
        )
        heat_ax.set_aspect("equal")

        # contour_ax.set_aspect("equal")
        # cs = contour_ax.contour(grid[:, :, 0], grid[:, :, 1], dists)

    plt.ion()
    for epoch in range(num_epochs):
        params = get_params(optimizer_state)

        rng, subrng = jax.random.split(rng)
        pts = generate_samples(subrng)

        with jax.disable_jit():
            gradient, losses = value_and_grad_loss_fn(params, pts)
        losses = tuple(np.array(loss) for loss in losses)

        # gradient = clip_grads(gradient, 1.0)
        print(f"epoch {epoch}: loss {losses}")

        if epoch % validation_skip == 0:
            drawnow(make_fig)

        optimizer_state = update(epoch, gradient[0], optimizer_state)

    return get_params(optimizer_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2 ** 14)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation_skip", type=int, default=100)
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary, frozen_box=True, box_it_up=True)

    rng = jax.random.PRNGKey(config.experiment.seed)

    params = generate_initialization(
        config.sdrf.model,
        config_args.batch_size,
        config_args.epochs,
        config_args.validation_skip,
        config.sdrf.model.optimizer.initial_lr,
        rng,
    )

    with open(config_args.output, "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    main()
