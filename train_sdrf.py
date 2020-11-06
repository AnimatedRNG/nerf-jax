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
from jax.experimental.optimizers import adam, clip_grads
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

Losses = namedtuple("Losses", ["rgb_loss", "eikonal_loss", "manifold_loss"])
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

        losses = Losses(rgb_loss=rgb_loss, eikonal_loss=e_loss, manifold_loss=m_loss)

        loss_weights = jnp.array([3e3, 5e1, 1e2])

        return jnp.dot(jnp.array([rgb_loss, e_loss, m_loss]), loss_weights), losses

    @jit
    def validation(subrng, params, image_id, iteration):
        H, W, focal = (
            intrinsics["val"].height,
            intrinsics["val"].width,
            intrinsics["val"].focal_length,
        )

        ray_origins, ray_directions = get_ray_bundle(
            H, W, focal, poses["val"][0][:3, :4].astype(np.float32),
        )

        rgb, depth = run_one_iter_of_sdrf(
            sdrf,
            params,
            ray_origins.reshape(-1, 3),
            ray_directions.reshape(-1, 3),
            iteration,
            config.sdrf,
            subrng[3],
        )

        return rgb.reshape(H, W, 3), depth.reshape(H, W, 1)

    value_and_grad_fn = jit(value_and_grad(loss_fn, argnums=(1,), has_aux=True))
    sdf_jit_fn = jit(lambda pts, ps: vmap(lambda pt: sdrf.geometry(pt, ps))(pts))

    height, width = intrinsics["val"].height, intrinsics["val"].width

    for i in trange(0, config.experiment.train_iters, config.experiment.jit_loop):
        rng, *subrng = jax.random.split(rng, 5)
        params = get_params(optimizer_state)

        (loss, losses), (grads,) = value_and_grad_fn(
            subrng, params, train_image_seq[i], i
        )
        grads = clip_grads(grads, 1.0)

        optimizer_state = update(i, grads, optimizer_state)

        if (
            i % config.experiment.print_every == 0
            or i == config.experiment.train_iters - 1
        ):
            tqdm.write(f"Iter {i}: Loss {loss}")

        writer.add_scalar("train/loss", loss, i)
        writer.add_scalar("train/rgb_loss", losses.rgb_loss, i)
        writer.add_scalar("train/eikonal_loss", losses.eikonal_loss, i)
        writer.add_scalar("train/manifold_loss", losses.manifold_loss, i)

        if i % config.experiment.validate_every == 0:
            start = time.time()
            rgb, depth = validation(subrng, params, 0, i)
            end = time.time()

            pts = jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(-2.0, 2.0, 64),
                    jnp.linspace(-2.0, 2.0, 64),
                    jnp.linspace(-2.0, 2.0, 64),
                ),
                axis=-1,
            )
            grid = pts[:, :, 32, :].reshape(64, 64, 3)
            dists = sdf_jit_fn(grid.reshape(-1, 3), params.geometry).reshape(64, 64, 1)
            dists_min, dists_max = dists.flatten().min(), dists.flatten().max()
            dists_span = dists_max - dists_min

            to_img = lambda x: np.array(
                np.clip(jnp.transpose(x, axes=(2, 1, 0)), 0.0, 1.0) * 255
            ).astype(np.uint8)

            writer.add_image("validation/rgb", to_img(rgb), i)
            writer.add_image("validation/depth", to_img(depth), i)
            writer.add_image(
                "validation/dists", to_img(dists + 2.0), i
            )
            print(f"Time to render {width}x{height} image: {(end - start)}")


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
