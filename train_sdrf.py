#!/usr/bin/env python3

import argparse
import functools
from pathlib import Path
from datetime import datetime
from collections import namedtuple

import numpy as np
import yaml
from box import Box
import jax
from jax import jit, vmap, pmap, grad, value_and_grad
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from jax.experimental.optimizers import adam
import haiku as hk

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from nerf import loader, sampler
from util import get_ray_bundle
from sdrf import (
    SDRFParams,
    SDRF,
    Siren,
    run_one_iter_of_sdrf,
    eikonal_loss,
    manifold_loss,
)

Losses = namedtuple("Losses", ["rgb_loss", "eikonal_loss", "inter_loss"])
register_pytree_node(Losses, lambda xs: (tuple(xs), None), lambda _, xs: Losses(*xs))


def init_networks(config, rng):
    geometry_fn = hk.transform(
        lambda x: Siren(
            3,
            1,
            config.geometry.num_layers,
            config.geometry.hidden_size,
            config.geometry.outermost_linear,
            config.geometry.activation,
        )(x)
    )
    appearance_fn = hk.transform(
        lambda pt, rd: Siren(
            6,
            3,
            config.geometry.num_layers,
            config.geometry.hidden_size,
            config.geometry.outermost_linear,
            config.geometry.activation,
        )(jnp.concatenate((pt, rd), axis=-1))
    )

    geometry_params = geometry_fn.init(rng[0], jnp.ones([3,]))
    appearance_params = appearance_fn.init(rng[0], jnp.ones([3,]), jnp.ones([3,]))

    geometry_fn = hk.without_apply_rng(geometry_fn)
    appearance_fn = hk.without_apply_rng(appearance_fn)

    return (
        SDRF(
            # the sum is a bit odd, but it avoids issues with batch size (1,) vs ()
            geometry=lambda pt, params: geometry_fn.apply(params, pt).sum(),
            appearance=lambda pt, rd, params: appearance_fn.apply(params, pt, rd),
        ),
        SDRFParams(geometry=geometry_params, appearance=appearance_params),
    )


def train_sdrf(config):
    rng = jax.random.PRNGKey(config.experiment.seed)
    rng, *subrng = jax.random.split(rng, 3)

    sdrf, sdrf_params = init_networks(config.sdrf.model, subrng)

    basedir = config.dataset.basedir
    print(f"Loading images/poses from {basedir}...")
    images, poses, intrinsics = loader(
        Path(".") / basedir, config.dataset.filter_chain, jax.devices()[0],
    )
    print("...done!")

    # TODO: figure out optimizer
    num_decay_steps = config.sdrf.model.optimizer.lr_decay * 1000
    init_adam, update, get_params = adam(
        lambda iteration: config.sdrf.model.optimizer.initial_lr
        * (config.sdrf.model.optimizer.lr_decay_factor ** (iteration / num_decay_steps))
    )
    optimizer_state = init_adam(sdrf_params)

    logdir = (
        Path("logs") / "sdrf" / "lego" / datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    )
    logdir.mkdir(exist_ok=True)
    writer = SummaryWriter(logdir.absolute())
    (logdir / "config.yml").open("w").write(config.to_yaml())

    rng, subrng_img = jax.random.split(rng, 2)
    train_image_seq = jax.random.randint(
        subrng_img,
        shape=(config.experiment.train_iters,),
        minval=0,
        maxval=images["train"].shape[0],
        dtype=jnp.uint32,
    )

    def loss_fn(subrng, params, image_id, iteration):
        ray_origins, ray_directions, target_s = sampler(
            images["train"][image_id],
            poses["train"][image_id],
            intrinsics["train"],
            subrng[0],
            config.dataset.sampler,
        )

        # pick a bunch of random points to use for eikonal/manifold loss
        # TODO: Maybe also sample on the isosurface?
        eikonal_samples = (
            jax.random.uniform(subrng[1], (config.sdrf.eikonal.num_samples, 3))
            * config.sdrf.eikonal.scale
        )

        manifold_samples = (
            jax.random.uniform(subrng[2], (config.sdrf.manifold.num_samples, 3))
            * config.sdrf.manifold.scale
        )

        # render
        rgb, depth = run_one_iter_of_sdrf(
            sdrf,
            params,
            ray_origins,
            ray_directions,
            iteration,
            config.sdrf,
            subrng[3],
        )

        rgb_loss = jnp.mean(((target_s[..., :3] - rgb) ** 2.0).flatten())

        e_loss, m_loss = (
            eikonal_loss(sdrf.geometry, eikonal_samples, params.geometry),
            manifold_loss(sdrf.geometry, manifold_samples, params.geometry),
        )

        losses = jnp.array([rgb_loss, e_loss, m_loss])

        loss_weights = jnp.array([3e3, 5e1, 1e2])

        return jnp.dot(losses, loss_weights), losses

    value_and_grad_fn = jit(value_and_grad(loss_fn, argnums=(1,), has_aux=True))

    for i in trange(0, config.experiment.train_iters, config.experiment.jit_loop):
        rng, *subrng = jax.random.split(rng, 5)
        params = get_params(optimizer_state)

        (loss, losses), (params,) = value_and_grad_fn(
            subrng, params, train_image_seq[i], i
        )

        optimizer_state = update(i, params, optimizer_state)

        print(loss, losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary)

    train_sdrf(config)


if __name__ == "__main__":
    import cv2
    import time

    main()
