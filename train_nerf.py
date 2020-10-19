#!/usr/bin/env python3

import argparse
import functools
from pathlib import Path

import numpy as np
import yaml
from box import Box
import jax
from jax import jit, vmap, pmap, grad
import jax.numpy as jnp
import haiku as hk
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from nerf_helpers import get_ray_bundle
from nerf_dataset import loader, sampler
from train_utils import run_one_iter_of_nerf, run_network
from models import FlexibleNeRFModel
from torch_to_jax import torch_to_jax


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

    # Logging
    logdir = Path("logs") / "lego"
    logdir.mkdir(exist_ok=True)
    writer = SummaryWriter(logdir.absolute())
    (logdir / "config.yml").open("w").write(config.to_yaml())

    mode = "val"
    H, W, focal = (
        intrinsics[mode].height,
        intrinsics[mode].width,
        intrinsics[mode].focal_length,
    )

    ray_origins, ray_directions = get_ray_bundle(
        H, W, focal, poses[mode][0][:3, :4].astype(np.float32),
    )

    rng, images = run_one_iter_of_nerf(
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
    images.block_until_ready()

    rgb_coarse, _, _, rgb_fine, _, _ = (
        images[..., :3],
        images[..., 3:4],
        images[..., 4:5],
        images[..., 5:8],
        images[..., 8:9],
        images[..., 9:10],
    )

    cv2.imshow("img", np.array(rgb_fine))
    cv2.waitKey(0)

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
    from torch_impl import *
    import time

    main()
