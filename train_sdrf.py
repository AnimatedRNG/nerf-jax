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
from jax.tree_util import register_pytree_node, tree_map
from jax.experimental.optimizers import adam
import haiku as hk

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from nerf import loader, sampler
from nerf import run_one_iter_of_nerf, run_network
from nerf import FlexibleNeRFModel, compute_embedding_size
from sdrf import (
    SDRF,
    SDRFGrid,
    SDRFParams,
    FeatureGrid,
    IGR,
    Siren,
    DumbDecoder,
    ConstantInitializer,
    ZeroInitializer,
    RadianceInitializer,
    SHInitializer,
    SirenInitializer,
    eikonal_loss,
    manifold_loss,
    sample_real_sh,
    run_one_iter_of_sdrf_nerflike,
    run_one_iter_of_sdrf,
)
from util import get_ray_bundle, create_mrc, img2mse, mse2psnr, save, restore


Losses = namedtuple(
    "Losses", ["coarse_loss", "fine_loss", "root_loss", "eikonal_loss", "manifold_loss"]
)
register_pytree_node(Losses, lambda xs: (tuple(xs), None), lambda _, xs: Losses(*xs))


def init_feature_grids(config, rng):
    def feature_grid_fns():
        create_geometry_fn = lambda: lambda x: x
        # create_ddf_fn = lambda: lambda x: 1.0
        create_ddf_fn = lambda: lambda x: jax.nn.sigmoid(x) * 1e-2
        create_appearance_fn = lambda: DumbDecoder(
            [16, 16, 3],
        )

        def siren_appearance_decoder(x, view):
            inp = jnp.concatenate((x, Siren(3, 16, 1, 16)(view)), axis=-1)
            return Siren(x.shape[-1], 3, 2, 16)(inp)

        def sh_appearance_fn(x, view):
            return jnp.concatenate(
                [
                    sample_real_sh(view, c)[jnp.newaxis]
                    for c in jnp.split(x, 3, axis=-1)
                ],
                axis=-1,
            )

        sdf_grid = FeatureGrid(
            64,
            lambda x: create_geometry_fn()(x),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=1,
        )
        warp_grid = FeatureGrid(
            64,
            lambda x: 2.0 * jax.nn.sigmoid(x) - 1.0,
            grid_min=jnp.array([0.0, 0.0, 0.0]),
            grid_max=jnp.array([1.0, 1.0, 1.0]),
            feature_size=1,
            feature_initializer_fn=ZeroInitializer,
        )
        ddf_grid = FeatureGrid(
            64,
            lambda x: create_ddf_fn()(x),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=1,
            feature_initializer_fn=ConstantInitializer,
        )
        radiance_grid = FeatureGrid(
            64,
            lambda x, view: create_appearance_fn()(x, view),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=16,
            feature_initializer_fn=RadianceInitializer,
            warp_field=warp_grid.sample,
        )
        """radiance_grid = FeatureGrid(
            64,
            lambda x, view: siren_appearance_decoder(x, view),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=16,
            feature_initializer_fn=SirenInitializer,
        )"""
        """radiance_grid = FeatureGrid(
            64,
            lambda x, view: sh_appearance_fn(x, view),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=27,
            feature_initializer_fn=SHInitializer,
        )"""
        """radiance_grid = FeatureGrid(
            64,
            lambda x, view: create_appearance_fn()(x, view),
            grid_min=jnp.array([-1.5, -1.5, -1.5]),
            grid_max=jnp.array([1.5, 1.5, 1.5]),
            feature_size=16,
            feature_initializer_fn=RadianceInitializer,
        )"""

        def init(pt):
            return (
                sdf_grid.sample(pt),
                ddf_grid.sample(pt),
                radiance_grid.sample(pt, [pt]),
                sdf_grid.finite_difference(),
            )

        return init, (
            sdf_grid.sample,
            ddf_grid.sample,
            radiance_grid.sample,
            sdf_grid.finite_difference,
        )

    feature_grid = hk.multi_transform(feature_grid_fns)

    params = feature_grid.init(
        rng,
        jnp.ones(
            [
                3,
            ]
        ),
    )

    (
        sdf_point_sample,
        ddf_point_sample,
        radiance_point_sample,
        finite_difference,
    ) = feature_grid.apply

    return (
        SDRFGrid(
            geometry=lambda pt, params: sdf_point_sample(params, None, pt),
            ddf=lambda pt, params: ddf_point_sample(params, None, pt),
            appearance=lambda pt, rd, params: radiance_point_sample(
                params, None, pt, [rd]
            ),
            finite_difference=lambda params: finite_difference(params, None),
        ),
        SDRFParams(geometry=params, appearance=params),
    )


def train_nerf(config):
    # Create random number generator
    rng = jax.random.PRNGKey(config.experiment.seed)
    rng, *subrng = jax.random.split(rng, 3)

    # create models
    sdrf, ps = init_feature_grids(config, rng)

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
    checkpoint_dir = logdir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
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
    print("focal length", intrinsics["test"].focal_length)

    def loss_fn(f_rng, ps, i, image_id, use_root=False):
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

        sdrf_f = SDRF(
            geometry=lambda pt: sdrf.geometry(pt, ps.geometry),
            ddf=lambda pt: sdrf.ddf(pt, ps.geometry),
            appearance=lambda pt, view: sdrf.appearance(pt, view, ps.appearance),
        )

        eikonal_samples = (
            jax.random.uniform(
                f_rng[2], (config.sdrf.manifold.num_samples, 3), minval=-2.0, maxval=2.0
            )
            * config.sdrf.manifold.scale
        )

        manifold_samples = (
            jax.random.uniform(
                f_rng[3], (config.sdrf.manifold.num_samples, 3), minval=-2.0, maxval=2.0
            )
            * config.sdrf.manifold.scale
        )

        df_di = sdrf.finite_difference(ps.geometry)
        e_loss, m_loss = (
            jnp.mean(jnp.square(1 - jax.vmap(jnp.linalg.norm)(df_di)).ravel()),
            manifold_loss(sdrf_f.geometry, manifold_samples),
        )
        # let's use the old version for now
        e_loss = eikonal_loss(sdrf_f.geometry, eikonal_samples)

        weights = (3e3, 1e2, 5e1)
        # weights = (3e3, 1e-9, 1e-9)

        if use_root:
            rgb_root, z_vals = run_one_iter_of_sdrf(
                sdrf_f,
                ray_origins,
                ray_directions,
                i,
                intrinsics,
                config.nerf.train,
                config.sdrf,
                config.dataset.projection,
                f_rng[1],
                False,
            )

            recon_loss = jnp.mean(((target_s[..., :3] - rgb_root) ** 2.0).flatten())
            loss = recon_loss * weights[0] + e_loss * weights[1] + m_loss * weights[2]

            losses = Losses(
                coarse_loss=0.0,
                fine_loss=0.0,
                root_loss=recon_loss,
                eikonal_loss=e_loss,
                manifold_loss=m_loss,
            )
        else:
            rendered_images = run_one_iter_of_sdrf_nerflike(
                sdrf_f,
                ray_origins,
                ray_directions,
                i,
                intrinsics,
                config.nerf,
                config.sdrf,
                config.dataset.projection,
                f_rng[1],
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

            coarse_loss = jnp.mean(((target_s[..., :3] - rgb_coarse) ** 2.0).flatten())
            loss = coarse_loss * weights[0] + e_loss * weights[1] + m_loss * weights[2]
            if config.nerf.train.num_fine > 0:
                fine_loss = jnp.mean(((target_s[..., :3] - rgb_fine) ** 2.0).flatten())
                loss = loss + fine_loss * weights[0]
                losses = Losses(
                    coarse_loss=coarse_loss,
                    fine_loss=fine_loss,
                    root_loss=0.0,
                    eikonal_loss=e_loss,
                    manifold_loss=m_loss,
                )
            else:
                losses = Losses(
                    coarse_loss=coarse_loss,
                    fine_loss=0.0,
                    root_loss=0.0,
                    eikonal_loss=e_loss,
                    manifold_loss=m_loss,
                )

        return loss, losses

    @functools.partial(
        jit,
        static_argnums=(
            3,
            4,
        ),
    )
    def validation(f_rng, i, ps, image_id, dset_name="val"):
        H, W, focal = (
            intrinsics[dset_name].height,
            intrinsics[dset_name].width,
            intrinsics[dset_name].focal_length,
        )

        uv, ray_origins, ray_directions = get_ray_bundle(
            H,
            W,
            focal,
            poses[dset_name][image_id][:3, :4].astype(np.float32),
        )

        sdrf_f = SDRF(
            geometry=lambda pt: sdrf.geometry(pt, ps.geometry),
            ddf=lambda pt: sdrf.ddf(pt, ps.geometry),
            appearance=lambda pt, view: sdrf.appearance(pt, view, ps.appearance),
        )

        rendered_images = run_one_iter_of_sdrf_nerflike(
            sdrf_f,
            ray_origins,
            ray_directions,
            i,
            intrinsics,
            config.nerf,
            config.sdrf,
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

        rgb_root, z_vals = run_one_iter_of_sdrf(
            sdrf_f,
            ray_origins,
            ray_directions,
            i,
            intrinsics,
            config.nerf.validation,
            config.sdrf,
            config.dataset.projection,
            rng,
            True,
        )

        return rgb_coarse, disp_coarse, rgb_fine, disp_fine, rgb_root, z_vals

    value_and_grad_fn = value_and_grad(loss_fn, argnums=(1,), has_aux=True)
    value_and_grad_fn_jit = jax.jit(value_and_grad_fn, static_argnums=(4,))

    def update_loop(rng, optimizer_state, i, use_root=False):
        rng, *subrng = jax.random.split(rng, 5)

        ps = get_params(optimizer_state)

        (_, losses), ps_grad = value_and_grad_fn_jit(
            subrng, ps, i, train_image_seq[i], use_root
        )

        optimizer_state = update(i, ps_grad[0], optimizer_state)
        return optimizer_state, losses

    for i in trange(0, config.experiment.train_iters, config.experiment.jit_loop):
        rng, subrng = jax.random.split(rng, 2)
        use_root = i > 400
        optimizer_state, losses = update_loop(subrng, optimizer_state, i, use_root)
        loss = losses.coarse_loss + losses.fine_loss + losses.root_loss

        """try:
            ps = get_params(optimizer_state)
            ps_np = tree_map(np.array, ps)
            q.put_nowait((ps_np, i))
        except queue.Full as _:
            pass"""

        # Validation
        if (
            i % config.experiment.print_every == 0
            or i == config.experiment.train_iters - 1
        ):
            tqdm.write(f"Iter {i}: Loss {loss}")

        writer.add_scalar("train/loss", loss, i)
        writer.add_scalar("train/coarse_loss", losses.coarse_loss, i)
        writer.add_scalar("train/fine_loss", losses.fine_loss, i)
        writer.add_scalar("train/root_loss", losses.root_loss, i)
        writer.add_scalar("train/eikonal_loss", losses.eikonal_loss, i)
        writer.add_scalar("train/manifold_loss", losses.manifold_loss, i)

        if i % config.experiment.validate_every == 0:
            start = time.time()
            image_id = (i // config.experiment.validate_every) % 200
            (
                rgb_coarse,
                disp_coarse,
                rgb_fine,
                disp_fine,
                rgb_root,
                z_vals,
            ) = validation(rng, i, get_params(optimizer_state), image_id, "test")
            end = time.time()
            z_vals = (z_vals - config.dataset.projection.near) / (
                config.dataset.projection.far - config.dataset.projection.near
            )

            target_img = images["test"][image_id]
            validation_psnr_coarse = mse2psnr(float(img2mse(rgb_coarse, target_img)))
            validation_psnr_root = mse2psnr(float(img2mse(rgb_root, target_img)))

            ps = get_params(optimizer_state)
            """create_mrc(
                str(logdir / "test.mrc"),
                jax.vmap(lambda pt: sdrf.geometry(pt, ps.geometry)),
                grid_min=jnp.array([-2.0, -2.0, -2.0]),
                grid_max=jnp.array([2.0, 2.0, 2.0]),
                resolution=256,
            )"""

            # save model
            checkpoint_subdir = checkpoint_dir / str(i)
            checkpoint_subdir.mkdir(exist_ok=True)
            save(checkpoint_subdir, ps)

            to_img = lambda x: np.array(
                np.clip(jnp.transpose(x, axes=(2, 1, 0)), 0.0, 1.0) * 255
            ).astype(np.uint8)

            writer.add_image("validation/rgb_coarse", to_img(rgb_coarse), i)
            writer.add_image(
                "validation/disp_coarse", to_img(disp_coarse.repeat(3, axis=-1)), i
            )
            writer.add_image("validation/rgb_root", to_img(rgb_root), i)
            writer.add_image(
                "validation/z_vals", to_img(z_vals[..., 0:1].repeat(3, axis=-1)), i
            )
            writer.add_image("validation/reference", to_img(target_img), i)
            writer.add_scalar("validation/psnr_coarse", validation_psnr_coarse, i)
            writer.add_scalar("validation/psnr_root", validation_psnr_root, i)
            if config.nerf.validation.num_fine > 0:
                validation_psnr_fine = mse2psnr(float(img2mse(rgb_fine, target_img)))
                writer.add_image("validation/rgb_fine", to_img(rgb_fine), i)
                writer.add_image(
                    "validation/disp_fine", to_img(disp_fine.repeat(3, axis=-1)), i
                )
                writer.add_scalar("validation/psnr_fine", validation_psnr_fine, i)

    # kill other thread
    # q.put_nowait(False)


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
    import time

    main()
