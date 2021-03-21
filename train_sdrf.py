#!/usr/bin/env python3

import argparse
import functools
from pathlib import Path
from datetime import datetime
from collections import namedtuple
import pickle

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

from nerf import loader, positional_encoding, sampler, FlexibleNeRFModel, NeRFModelMode
from util import get_ray_bundle, gradient_visualization, serialize_box
from sdrf import (
    SDRFParams,
    SDRF,
    Siren,
    extract_debug,
    run_one_iter_of_sdrf,
    eikonal_loss,
    manifold_loss,
)

Losses = namedtuple("Losses", ["rgb_loss", "eikonal_loss", "manifold_loss"])
register_pytree_node(Losses, lambda xs: (tuple(xs), None), lambda _, xs: Losses(*xs))


def init_networks(config, rng):
    geometry_fn = hk.transform(
        lambda x: FlexibleNeRFModel(
            num_layers=config.network.num_layers,
            hidden_size=config.network.hidden_size,
            skip_connect_every=config.network.skip_connect_every,
            geometric_init=False,
        )(
            # positional_encoding(x, config.network.num_encoding_fn_xyz),
            x,
            None,
            mode=NeRFModelMode.GEOMETRY,
        )
    )
    appearance_fn = hk.transform(
        lambda x, view: FlexibleNeRFModel(
            num_layers=config.network.num_layers,
            hidden_size=config.network.hidden_size,
            skip_connect_every=config.network.skip_connect_every,
            geometric_init=False,
        )(
            # positional_encoding(x, config.network.num_encoding_fn_xyz),
            # positional_encoding(view, config.network.num_encoding_fn_dir),
            x,
            view,
            mode=NeRFModelMode.APPEARANCE,
        )
    )

    geometry_params = geometry_fn.init(
        rng[0],
        jnp.ones(
            [
                3,
            ]
        ),
    )
    appearance_params = appearance_fn.init(
        rng[0],
        jnp.ones(
            [
                3,
            ]
        ),
        jnp.ones(
            [
                3,
            ]
        ),
    )

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


def log_debug_imgs(writer, reap_dict, height, width, i):
    to_img = lambda x: np.array(
        np.clip(jnp.transpose(gradient_visualization(x), axes=(2, 1, 0)), 0.0, 1.0)
        * 255
    ).astype(np.uint8)

    uv = reap_dict["uv"]
    for key, debug_img in reap_dict.items():
        if key == "uv":
            continue
        if len(debug_img.shape) == 3:
            extracted_img = extract_debug(uv, debug_img, height, width)
            writer.add_image(key, to_img(extracted_img), i)
        elif len(debug_img.shape) == 4:
            for subimg_idx in range(debug_img.shape[2]):
                extracted_img = extract_debug(
                    uv, debug_img[:, :, subimg_idx, :], height, width
                )
                writer.add_image(f"{key}_{subimg_idx}", to_img(extracted_img), i)


def train_sdrf(config):
    if config.sdrf.render.oryx_debug:
        import oryx
        import oryx.core as core

    rng = jax.random.PRNGKey(config.experiment.seed)
    rng, *subrng = jax.random.split(rng, 3)

    sdrf, sdrf_params = init_networks(config.sdrf.model, subrng)
    # with open("experiment/sphere_nerf.pkl", "rb") as pkl:
    with open("experiment/sphere_nerf_penc.pkl", "rb") as pkl:
        g_a_params = pickle.load(pkl)
        sdrf_params = SDRFParams(geometry=g_a_params, appearance=g_a_params)

    basedir = config.dataset.basedir
    print(f"Loading images/poses from {basedir}...")
    images, poses, intrinsics = loader(
        Path(".") / basedir,
        config.dataset.filter_chain,
        jax.devices()[0],
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

    config = serialize_box("SDRFConfig", config)

    def loss_fn(subrng, params, image_id, iteration):
        uv, ray_origins, ray_directions, target_s = sampler(
            images["train"][image_id],
            poses["train"][image_id],
            intrinsics["train"],
            subrng[0],
            config.dataset.sampler,
        )

        """(uv, ray_origins, ray_directions), target_s = (
            get_ray_bundle(
                intrinsics["train"].height,
                intrinsics["train"].width,
                intrinsics["train"].focal_length,
                poses["train"][image_id][:3, :4],
            ),
            images["train"][image_id],
        )"""
        ray_origins, ray_directions, target_s = (
            ray_origins.reshape(1, -1, 3),
            ray_directions.reshape(1, -1, 3),
            target_s.reshape(1, -1, 3),
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
            uv,
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

        # loss_weights = jnp.array([3e3, 5e1, 1e2])
        # loss_weights = jnp.array([3e3, 1e-9, 1e-9])
        # loss_weights = jnp.array([3e3, 1e2, 1e2])
        # loss_weights = jnp.array([3e3, 1e2, 1e-9])
        loss_weights = jnp.array([3e3, 1e2, 1e2])

        return jnp.dot(jnp.array([rgb_loss, e_loss, m_loss]), loss_weights), losses

    @jit
    def validation(subrng, params, image_id, iteration):
        H, W, focal = (
            intrinsics["val"].height,
            intrinsics["val"].width,
            intrinsics["val"].focal_length,
        )

        uv, ray_origins, ray_directions = get_ray_bundle(
            H,
            W,
            focal,
            # poses["val"][0][:3, :4].astype(np.float32),
            poses["val"][image_id][:3, :4].astype(np.float32),
        )

        rgb, depth = run_one_iter_of_sdrf(
            sdrf,
            params,
            uv.reshape(-1, 3),
            ray_origins.reshape(-1, 3),
            ray_directions.reshape(-1, 3),
            iteration,
            config.sdrf,
            subrng[3],
        )

        return rgb.reshape(H, W, 3), depth.reshape(H, W, 1)

    value_and_grad_fn = jit(value_and_grad(loss_fn, argnums=(1,), has_aux=True))
    if config.sdrf.render.oryx_debug:
        reap_fn = jit(
            core.reap(value_and_grad(loss_fn, argnums=(1,), has_aux=True), tag="vjp")
        )
    sdf_jit_fn = jit(lambda pts, ps: vmap(lambda pt: sdrf.geometry(pt, ps))(pts))

    height, width = intrinsics["val"].height, intrinsics["val"].width

    for i in trange(0, config.experiment.train_iters, config.experiment.jit_loop):
        rng, *subrng = jax.random.split(rng, 5)
        params = get_params(optimizer_state)

        (loss, losses), (grads,) = value_and_grad_fn(
            subrng, params, train_image_seq[i], i
        )
        # grads = clip_grads(grads, 1.0)

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
            if config.sdrf.render.oryx_debug:
                reap_dict = reap_fn(subrng, params, train_image_seq[i], i)
                log_debug_imgs(writer, reap_dict, height, width, i)

            start = time.time()
            rgb, depth = validation(subrng, params, 0, i)
            # rgb, depth = validation(subrng, params, train_image_seq[i], i)
            end = time.time()

            to_img = lambda x: np.array(
                np.clip(jnp.transpose(x, axes=(2, 1, 0)), 0.0, 1.0) * 255
            ).astype(np.uint8)

            writer.add_image("validation/rgb", to_img(rgb), i)
            writer.add_image("validation/depth", to_img(depth / depth.max()), i)
            writer.add_image("validation/target", to_img(images["val"][0]))
            # writer.add_image("validation/target", to_img(images["val"][train_image_seq[i]]), i)

            print(f"Time to render {width}x{height} image: {(end - start)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary, frozen_box=True, box_it_up=True)

    train_sdrf(config)


if __name__ == "__main__":
    import time

    main()
