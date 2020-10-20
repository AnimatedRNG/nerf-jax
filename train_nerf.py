#!/usr/bin/env python3

import argparse
import functools
from pathlib import Path

import numpy as np
import yaml
from box import Box
import jax
from jax import jit, vmap, pmap, grad, value_and_grad
import jax.numpy as jnp
from jax.experimental.optimizers import adam
import haiku as hk

# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from nerf import get_ray_bundle
from nerf import loader, sampler
from nerf import run_one_iter_of_nerf, run_network
from nerf import FlexibleNeRFModel
from reference import torch_to_jax


def train_nerf(config):
    # Create random number generator
    rng = jax.random.PRNGKey(config.experiment.seed)

    # create models
    model_coarse = hk.without_apply_rng(
        hk.transform(
            lambda x: FlexibleNeRFModel(
                num_encoding_fn_xyz=config.model.coarse.num_encoding_fn_xyz,
                num_encoding_fn_dir=config.model.coarse.num_encoding_fn_dir,
                include_input_xyz=True,
                include_input_dir=True,
                use_viewdirs=config.model.coarse.use_viewdirs,
            )(x)
        )
    )

    model_fine = hk.without_apply_rng(
        hk.transform(
            lambda x: FlexibleNeRFModel(
                num_encoding_fn_xyz=config.model.fine.num_encoding_fn_xyz,
                num_encoding_fn_dir=config.model.fine.num_encoding_fn_dir,
                include_input_xyz=True,
                include_input_dir=True,
                use_viewdirs=config.model.fine.use_viewdirs,
            )(x)
        )
    )

    # Load checkpoint (delete this later)
    checkpoint = torch.load(
        "./checkpoint/checkpoint199999.ckpt"
        # "./checkpoint/checkpoint95000.ckpt"
    )
    model_coarse_params = torch_to_jax(
        checkpoint["model_coarse_state_dict"], "flexible_ne_rf_model"
    )
    model_fine_params = torch_to_jax(
        checkpoint["model_fine_state_dict"], "flexible_ne_rf_model"
    )

    # Create loader
    basedir = config.dataset.basedir
    images, poses, intrinsics = loader(
        Path(".") / basedir, config.dataset.filter_chain, jax.devices()[0],
    )

    # TODO: figure out optimizer
    num_decay_steps = config.model.optimizer.lr_decay * 1000
    init_adam, update, get_params = adam(
        lambda iteration: config.model.optimizer.initial_lr
        * (config.model.optimizer.lr_decay_factor ** (iteration / num_decay_steps))
    )
    optimizer_state = init_adam((model_coarse_params, model_fine_params))

    # Logging
    logdir = Path("logs") / "lego"
    logdir.mkdir(exist_ok=True)
    # writer = SummaryWriter(logdir.absolute())
    (logdir / "config.yml").open("w").write(config.to_yaml())

    mode = "train"
    H, W, focal = (
        intrinsics[mode].height,
        intrinsics[mode].width,
        intrinsics[mode].focal_length,
    )

    #@jit
    def loss_fn(f_rng, cp, fp, image_id):
        ray_origins, ray_directions, target_s = sampler(
            images["train"][0],
            poses["train"][0],
            intrinsics["train"],
            rng,
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
            config.model,
            config.dataset.projection,
            f_rng,
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

        loss = jnp.mean(((target_s[..., :3] - rgb_coarse) ** 2.0).flatten())
        if config.nerf.train.num_fine > 0:
            loss = loss + jnp.mean(((target_s[..., :3] - rgb_fine) ** 2.0).flatten())
        return loss

    """ray_origins, ray_directions = get_ray_bundle(
        H, W, focal, poses[mode][0][:3, :4].astype(np.float32),
    )"""

    rng, subrng_img = jax.random.split(rng, 2)
    train_image_seq = jax.random.randint(
        subrng_img,
        shape=(config.experiment.train_iters,),
        minval=0,
        maxval=images["train"].shape[0],
        dtype=jnp.uint32,
    )

    @functools.partial(jit, static_argnums=(2,))
    def update_loop(rng, optimizer_state, num_iterations):
        def inner(i, rng_optimizer_state):
            rng, optimizer_state = rng_optimizer_state

            rng, subrng = jax.random.split(rng, 2)

            (model_coarse_params, model_fine_params) = get_params(optimizer_state)

            loss, (cp_grad, fp_grad) = value_and_grad(loss_fn, argnums=(1, 2))(
                subrng, model_coarse_params, model_fine_params, train_image_seq[i]
            )

            optimizer_state = update(i, (cp_grad, fp_grad), optimizer_state)

            return rng, optimizer_state
        return jax.lax.fori_loop(0, num_iterations, inner, (rng, optimizer_state))

    for i in trange(0, config.experiment.train_iters, config.experiment.jit_loop):
        rng, optimizer_state = update_loop(rng, optimizer_state, config.experiment.jit_loop)

    """rng, rendered_images = run_one_iter_of_nerf(
        H,
        W,
        focal,
        functools.partial(model_coarse.apply, model_coarse_params),
        functools.partial(model_fine.apply, model_fine_params),
        ray_origins,
        ray_directions,
        config.nerf.validation,
        config.model,
        config.dataset.projection,
        rng,
        True,
    )

    rgb_coarse, _, _, rgb_fine, _, _ = (
        rendered_images[..., :3],
        rendered_images[..., 3:4],
        rendered_images[..., 4:5],
        rendered_images[..., 5:8],
        rendered_images[..., 8:9],
        rendered_images[..., 9:10],
    )"""

    # cv2.imshow("reference", np.array(images["val"][0]))
    # cv2.imshow("render", np.array(rgb_fine))
    # cv2.waitKey(0)

    # Optimization loop
    # for i in trange(0, config.experiment.train_iters):
    #    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary)

    train_nerf(config)


if __name__ == "__main__":
    import cv2
    import torch
    from reference import *
    import time

    main()
