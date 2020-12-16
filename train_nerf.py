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
from nerf import run_one_iter_of_nerf, run_network
from nerf import FlexibleNeRFModel, compute_embedding_size
from reference import torch_to_jax
from util import get_ray_bundle


Losses = namedtuple("Losses", ["coarse_loss", "fine_loss"])
register_pytree_node(Losses, lambda xs: (tuple(xs), None), lambda _, xs: Losses(*xs))


def create_networks(config):
    coarse_embedding = compute_embedding_size(
        include_input_xyz=True,
        include_input_dir=True,
        num_encoding_fn_xyz=config.nerf.model.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=config.nerf.model.coarse.num_encoding_fn_dir,
    )
    fine_embedding = compute_embedding_size(
        include_input_xyz=True,
        include_input_dir=True,
        num_encoding_fn_xyz=config.nerf.model.fine.num_encoding_fn_xyz,
        num_encoding_fn_dir=config.nerf.model.fine.num_encoding_fn_dir,
    )

    model_coarse = hk.transform(
        lambda x: FlexibleNeRFModel(
            num_encoding_fn_xyz=config.nerf.model.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=config.nerf.model.coarse.num_encoding_fn_dir,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=config.nerf.model.coarse.use_viewdirs,
        )(x)
    )

    model_fine = hk.transform(
        lambda x: FlexibleNeRFModel(
            num_encoding_fn_xyz=config.nerf.model.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=config.nerf.model.fine.num_encoding_fn_dir,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=config.nerf.model.fine.use_viewdirs,
        )(x)
    )

    return (
        model_coarse,
        model_fine,
        coarse_embedding,
        fine_embedding,
    )


def init_networks(
    rng, model_coarse, model_fine, coarse_embedding, fine_embedding, config
):
    dummy_input_coarse = jnp.zeros((config.nerf.train.chunksize, sum(coarse_embedding)))
    dummy_input_fine = jnp.zeros((config.nerf.train.chunksize, sum(fine_embedding)))

    coarse_params = model_coarse.init(rng[0], dummy_input_coarse)
    fine_params = model_fine.init(rng[1], dummy_input_fine)

    return (coarse_params, fine_params)


def load_networks_from_torch(checkpoint_file="./checkpoint/checkpoint199999.ckpt"):
    checkpoint = torch.load(checkpoint_file)
    model_coarse_params = torch_to_jax(
        checkpoint["model_coarse_state_dict"], "flexible_ne_rf_model"
    )
    model_fine_params = torch_to_jax(
        checkpoint["model_fine_state_dict"], "flexible_ne_rf_model"
    )

    return (model_coarse_params, model_fine_params)


def train_nerf(config):
    # Create random number generator
    rng = jax.random.PRNGKey(config.experiment.seed)

    # create models
    model_coarse, model_fine, coarse_embedding, fine_embedding = create_networks(config)
    # model_coarse_params, model_fine_params = load_networks_from_torch(
    #    "checkpoint/checkpoint00000.ckpt"
    # )
    rng, *subrng = jax.random.split(rng, 3)
    model_coarse_params, model_fine_params = init_networks(
        subrng, model_coarse, model_fine, coarse_embedding, fine_embedding, config
    )

    model_coarse, model_fine = (
        hk.without_apply_rng(model_coarse),
        hk.without_apply_rng(model_fine),
    )

    # Create loader
    basedir = config.dataset.basedir
    print(f"Loading images/poses from {basedir}...")
    images, poses, intrinsics = loader(
        Path(".") / basedir, config.dataset.filter_chain, jax.devices()[0],
    )
    print("...done!")

    # TODO: figure out optimizer
    num_decay_steps = config.nerf.model.optimizer.lr_decay * 1000
    init_adam, update, get_params = adam(
        lambda iteration: config.nerf.model.optimizer.initial_lr
        * (config.nerf.model.optimizer.lr_decay_factor ** (iteration / num_decay_steps))
    )
    optimizer_state = init_adam((model_coarse_params, model_fine_params))

    # Logging
    logdir = Path("logs") / "lego" / datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
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

    def loss_fn(f_rng, cp, fp, image_id):
        H, W, focal = (
            intrinsics["train"].height,
            intrinsics["train"].width,
            intrinsics["train"].focal_length,
        )

        ray_origins, ray_directions, target_s = sampler(
            images["train"][image_id],
            poses["train"][image_id],
            intrinsics["train"],
            f_rng[0],
            config.dataset.sampler,
        )

        _, rendered_images = run_one_iter_of_nerf(
            H,
            W,
            focal,
            functools.partial(model_coarse.apply, cp),
            functools.partial(model_fine.apply, fp),
            ray_origins,
            ray_directions,
            config.nerf.train,
            config.nerf.model,
            config.dataset.projection,
            f_rng[1],
            False,
        )

        rgb_coarse, _, _, rgb_fine, _, _ = (
            rendered_images[..., :3],
            rendered_images[..., 3:4],
            rendered_images[..., 4:5],
            rendered_images[..., 5:8],
            rendered_images[..., 8:9],
            rendered_images[..., 9:10],
        )

        coarse_loss = jnp.mean(((target_s[..., :3] - rgb_coarse) ** 2.0).flatten())
        loss = coarse_loss
        if config.nerf.train.num_fine > 0:
            fine_loss = jnp.mean(((target_s[..., :3] - rgb_fine) ** 2.0).flatten())
            loss = loss + fine_loss
        return loss, Losses(coarse_loss=coarse_loss, fine_loss=fine_loss)

    @jit
    def validation(f_rng, cp, fp, image_id):
        H, W, focal = (
            intrinsics["val"].height,
            intrinsics["val"].width,
            intrinsics["val"].focal_length,
        )

        ray_origins, ray_directions = get_ray_bundle(
            H, W, focal, poses["val"][0][:3, :4].astype(np.float32),
        )

        rng, rendered_images = run_one_iter_of_nerf(
            H,
            W,
            focal,
            functools.partial(model_coarse.apply, cp),
            functools.partial(model_fine.apply, fp),
            ray_origins,
            ray_directions,
            config.nerf.validation,
            config.nerf.model,
            config.dataset.projection,
            f_rng,
            True,
        )

        rgb_coarse, _, _, rgb_fine, _, _ = (
            rendered_images[..., :3],
            rendered_images[..., 3:4],
            rendered_images[..., 4:5],
            rendered_images[..., 5:8],
            rendered_images[..., 8:9],
            rendered_images[..., 9:10],
        )

        return rgb_coarse, rgb_fine

    @jit
    def update_loop(rng, optimizer_state, start, num_iterations):
        def inner(i, rng_optimizer_state):
            rng, optimizer_state, _ = rng_optimizer_state

            rng, *subrng = jax.random.split(rng, 3)

            (model_coarse_params, model_fine_params) = get_params(optimizer_state)

            (_, losses), (cp_grad, fp_grad) = value_and_grad(
                loss_fn, argnums=(1, 2), has_aux=True
            )(subrng, model_coarse_params, model_fine_params, train_image_seq[i])

            optimizer_state = update(i, (cp_grad, fp_grad), optimizer_state)

            return rng, optimizer_state, losses

        return jax.lax.fori_loop(
            start,
            start + num_iterations,
            inner,
            (rng, optimizer_state, Losses(coarse_loss=0.0, fine_loss=0.0)),
        )

    for i in trange(0, config.experiment.train_iters, config.experiment.jit_loop):
        rng, optimizer_state, losses = update_loop(
            rng, optimizer_state, i, config.experiment.jit_loop
        )
        loss = losses.coarse_loss + losses.fine_loss

        # Validation
        if (
            i % config.experiment.print_every == 0
            or i == config.experiment.train_iters - 1
        ):
            tqdm.write(f"Iter {i}: Loss {loss}")

        writer.add_scalar("train/loss", loss, i)
        writer.add_scalar("train/coarse_loss", losses.coarse_loss, i)
        writer.add_scalar("train/fine_loss", losses.fine_loss, i)

        if i % config.experiment.validate_every == 0:
            start = time.time()
            rgb_coarse, rgb_fine = validation(rng, *get_params(optimizer_state), 0)
            end = time.time()

            to_img = lambda x: np.array(
                np.clip(jnp.transpose(x, axes=(2, 1, 0)), 0.0, 1.0) * 255
            ).astype(np.uint8)

            writer.add_image("validation/rgb_coarse", to_img(rgb_coarse), i)
            writer.add_image("validation/rgb_fine", to_img(rgb_fine), i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary)

    train_nerf(config)


if __name__ == "__main__":
    import torch
    from reference import *
    import time

    main()
