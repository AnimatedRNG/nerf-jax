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
from sdrf import (
    SDRFGrid,
    SDRFParams,
    FeatureGrid,
    IGR,
    DumbDecoder,
    eikonal_loss,
    manifold_loss,
)
from reference import torch_to_jax
from util import get_ray_bundle, create_mrc


Losses = namedtuple(
    "Losses", ["coarse_loss", "fine_loss", "eikonal_loss", "manifold_loss"]
)
register_pytree_node(Losses, lambda xs: (tuple(xs), None), lambda _, xs: Losses(*xs))


def init_feature_grids(config, rng):
    def feature_grid_fns():
        create_geometry_fn = lambda: IGR(
            [
                16,
            ],
            skip_in=tuple(),
            beta=100.0,
        )
        create_appearance_fn = lambda: DumbDecoder(
            [16, 16, 3],
        )
        grid = FeatureGrid(
            64,
            lambda x, *args: (
                create_geometry_fn()(x),
                create_appearance_fn()(x, *args) if len(tuple(args)) > 0 else None,
            ),
            # grid_min=jnp.array([-2.0, -2.0, -2.0]),
            # grid_max=jnp.array([2.0, 2.0, 2.0]),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=16,
        )

        def init(pt, scale_factor):
            return grid.sample(grid(scale_factor), pt, [pt])

        return init, (
            grid,
            grid.sample,
        )

    feature_grid = hk.multi_transform(feature_grid_fns)

    params = feature_grid.init(
        rng,
        jnp.ones(
            [
                3,
            ]
        ),
        1.0,
    )

    downsample, point_sample = feature_grid.apply

    return (
        SDRFGrid(
            downsample=lambda params, scale_factor: downsample(
                params, None, scale_factor
            ),
            geometry=lambda mipmap, pt, params: point_sample(params, None, mipmap, pt)[
                0
            ],
            appearance=lambda mipmap, pt, rd, params: point_sample(
                params, None, mipmap, pt, [rd]
            )[1],
        ),
        SDRFParams(geometry=params, appearance=params),
    )


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
        lambda x, view: FlexibleNeRFModel(
            num_encoding_fn_xyz=config.nerf.model.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=config.nerf.model.coarse.num_encoding_fn_dir,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=config.nerf.model.coarse.use_viewdirs,
            # geometric_init=False,
        )(x, view)
    )

    model_fine = hk.transform(
        lambda x, view: FlexibleNeRFModel(
            num_encoding_fn_xyz=config.nerf.model.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=config.nerf.model.fine.num_encoding_fn_dir,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=config.nerf.model.fine.use_viewdirs,
            # geometric_init=False,
        )(x, view),
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
    dummy_input_coarse_xyz, dummy_input_coarse_dir = (
        jnp.zeros((config.nerf.train.chunksize, coarse_embedding[0])),
        jnp.zeros((config.nerf.train.chunksize, coarse_embedding[1])),
    )
    dummy_input_fine_xyz, dummy_input_fine_dir = (
        jnp.zeros((config.nerf.train.chunksize, fine_embedding[0])),
        jnp.zeros((config.nerf.train.chunksize, fine_embedding[1])),
    )

    coarse_params = model_coarse.init(
        rng[0], dummy_input_coarse_xyz, dummy_input_coarse_dir
    )
    fine_params = model_fine.init(rng[1], dummy_input_fine_xyz, dummy_input_fine_dir)

    return (coarse_params, fine_params)


def train_nerf(config):
    # Create random number generator
    rng = jax.random.PRNGKey(config.experiment.seed)
    rng, *subrng = jax.random.split(rng, 3)

    # create models
    """model_coarse, model_fine, coarse_embedding, fine_embedding = create_networks(config)

    model_coarse_params, model_fine_params = init_networks(
        subrng, model_coarse, model_fine, coarse_embedding, fine_embedding, config
    )

    model_coarse, model_fine = (
        hk.without_apply_rng(model_coarse),
        hk.without_apply_rng(model_fine),
    )
    ps = (model_coarse_params, model_fine_params)"""

    sdrf, ps = init_feature_grids(config, rng)

    def scene_fn(i, downsampled, ps, pt, view, sdf=False):
        sigma = 1e-1 * (0.01 ** (i / 200000))
        volsdf_psi = lambda dist: jax.lax.cond(
            (dist <= 0.0)[0],
            dist,
            lambda x: 0.5 * jnp.exp(x / sigma),
            dist,
            lambda x: 1 - 0.5 * jnp.exp(-x / sigma),
        )
        # volsdf_phi = lambda dist: 1e3 * volsdf_psi(-dist)
        volsdf_phi = lambda dist: (sigma ** -1) * volsdf_psi(-dist)

        if sdf:
            alpha = sdrf.geometry(downsampled, pt, ps.geometry)
        else:
            alpha = volsdf_phi(sdrf.geometry(downsampled, pt, ps.geometry))
        rgb = sdrf.appearance(downsampled, pt, view, ps.appearance)
        return (rgb, alpha)

    # Create loader
    basedir = config.dataset.basedir
    print(f"Loading images/poses from {basedir}...")
    images, poses, intrinsics = loader(
        Path(".") / basedir,
        config.dataset.filter_chain,
        jax.devices()[0],
    )
    print("...done!")

    # TODO: figure out optimizer
    num_decay_steps = config.nerf.model.optimizer.lr_decay * 1000
    init_adam, update, get_params = adam(
        lambda iteration: config.nerf.model.optimizer.initial_lr
        * (config.nerf.model.optimizer.lr_decay_factor ** (iteration / num_decay_steps))
    )
    optimizer_state = init_adam(ps)

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

    def loss_fn(f_rng, ps, i, image_id):
        # cp, fp = ps

        H, W, focal = (
            intrinsics["train"].height,
            intrinsics["train"].width,
            intrinsics["train"].focal_length,
        )

        uv, ray_origins, ray_directions, target_s = sampler(
            images["train"][image_id],
            poses["train"][image_id],
            intrinsics["train"],
            f_rng[0],
            config.dataset.sampler,
        )

        downsampled = sdrf.downsample(ps.geometry, 0.1)
        scene_fn_e = functools.partial(scene_fn, i, downsampled, ps)
        scene_fn_sdf = lambda pt: scene_fn(
            i,
            downsampled,
            ps,
            pt,
            jnp.ones(
                3,
            ),
            sdf=True,
        )[1]

        _, rendered_images = run_one_iter_of_nerf(
            H,
            W,
            focal,
            # functools.partial(model_coarse.apply, cp),
            # functools.partial(model_fine.apply, fp),
            scene_fn_e,
            scene_fn_e,
            ray_origins,
            ray_directions,
            config.nerf.train,
            config.nerf.model,
            config.dataset.projection,
            f_rng[1],
            False,
        )

        eikonal_samples = (
            jax.random.uniform(
                f_rng[2], (config.sdrf.eikonal.num_samples, 3), minval=-1.0, maxval=1.0
            )
            * config.sdrf.eikonal.scale
        )

        manifold_samples = (
            jax.random.uniform(
                f_rng[3], (config.sdrf.manifold.num_samples, 3), minval=-1.0, maxval=1.0
            )
            * config.sdrf.manifold.scale
        )

        e_loss, m_loss = (
            eikonal_loss(scene_fn_sdf, eikonal_samples),
            manifold_loss(scene_fn_sdf, manifold_samples),
        )

        rgb_coarse, _, _, rgb_fine, _, _ = (
            rendered_images[..., :3],
            rendered_images[..., 3:4],
            rendered_images[..., 4:5],
            rendered_images[..., 5:8],
            rendered_images[..., 8:9],
            rendered_images[..., 9:10],
        )

        # weights = (3e3, 1e2, 5e1)
        weights = (3e3, 1e2, 1e-9)

        coarse_loss = jnp.mean(((target_s[..., :3] - rgb_coarse) ** 2.0).flatten())
        loss = coarse_loss * weights[0] + e_loss * weights[1] + m_loss * weights[2]
        if config.nerf.train.num_fine > 0:
            fine_loss = jnp.mean(((target_s[..., :3] - rgb_fine) ** 2.0).flatten())
            loss = loss + fine_loss * weights[0]
            losses = Losses(
                coarse_loss=coarse_loss,
                fine_loss=fine_loss,
                eikonal_loss=e_loss,
                manifold_loss=m_loss,
            )
        else:
            losses = Losses(
                coarse_loss=coarse_loss,
                fine_loss=0.0,
                eikonal_loss=e_loss,
                manifold_loss=m_loss,
            )
        return loss, losses

    @jit
    def validation(f_rng, i, ps, image_id):
        cp, fp = ps

        H, W, focal = (
            intrinsics["val"].height,
            intrinsics["val"].width,
            intrinsics["val"].focal_length,
        )

        uv, ray_origins, ray_directions = get_ray_bundle(
            H,
            W,
            focal,
            poses["val"][0][:3, :4].astype(np.float32),
        )

        downsampled = sdrf.downsample(ps.geometry, 0.1)

        rng, rendered_images = run_one_iter_of_nerf(
            H,
            W,
            focal,
            # functools.partial(model_coarse.apply, cp),
            # functools.partial(model_fine.apply, fp),
            functools.partial(scene_fn, i, downsampled, ps),
            functools.partial(scene_fn, i, downsampled, ps),
            ray_origins,
            ray_directions,
            config.nerf.validation,
            config.nerf.model,
            config.dataset.projection,
            f_rng,
            True,
        )

        rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = (
            rendered_images[..., :3],
            rendered_images[..., 3:4],
            rendered_images[..., 4:5],
            rendered_images[..., 5:8],
            rendered_images[..., 8:9],
            rendered_images[..., 9:10],
        )

        return rgb_coarse, disp_coarse, rgb_fine, disp_fine

    @jit
    def update_loop(rng, optimizer_state, start, num_iterations):
        def inner(i, rng_optimizer_state):
            rng, optimizer_state, _ = rng_optimizer_state

            rng, *subrng = jax.random.split(rng, 5)

            ps = get_params(optimizer_state)

            (_, losses), ps_grad = value_and_grad(loss_fn, argnums=(1,), has_aux=True)(
                subrng, ps, i, train_image_seq[i]
            )

            optimizer_state = update(i, ps_grad[0], optimizer_state)

            return rng, optimizer_state, losses

        return jax.lax.fori_loop(
            start,
            start + num_iterations,
            inner,
            (
                rng,
                optimizer_state,
                Losses(
                    coarse_loss=0.0, fine_loss=0.0, eikonal_loss=0.0, manifold_loss=0.0
                ),
            ),
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
        writer.add_scalar("train/eikonal_loss", losses.eikonal_loss, i)
        writer.add_scalar("train/manifold_loss", losses.manifold_loss, i)

        if i % config.experiment.validate_every == 0:
            start = time.time()
            rgb_coarse, disp_coarse, rgb_fine, disp_fine = validation(
                rng, i, get_params(optimizer_state), 0
            )
            end = time.time()

            downsampled = sdrf.downsample(ps.geometry, 0.1)
            ps = get_params(optimizer_state)
            '''create_mrc(
                str(logdir / "test.mrc"),
                jax.vmap(
                    lambda pt: scene_fn(
                        i,
                        downsampled,
                        ps,
                        pt,
                        jnp.ones(
                            3,
                        ),
                        sdf=False,
                    )[1]
                ),
                grid_min=jnp.array([-2.0, -2.0, -2.0]),
                grid_max=jnp.array([2.0, 2.0, 2.0]),
                resolution=256,
            )'''

            to_img = lambda x: np.array(
                np.clip(jnp.transpose(x, axes=(2, 1, 0)), 0.0, 1.0) * 255
            ).astype(np.uint8)

            writer.add_image("validation/rgb_coarse", to_img(rgb_coarse), i)
            writer.add_image(
                "validation/disp_coarse", to_img(disp_coarse.repeat(3, axis=-1)), i
            )
            if config.nerf.validation.num_fine > 0:
                writer.add_image("validation/rgb_fine", to_img(rgb_fine), i)
                writer.add_image(
                    "validation/disp_fine", to_img(disp_fine.repeat(3, axis=-1)), i
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary, frozen_box=True, box_it_up=True)

    train_nerf(config)


if __name__ == "__main__":
    import torch
    from reference import *
    import time

    main()
