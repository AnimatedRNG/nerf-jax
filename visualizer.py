#!/usr/bin/env python3

import functools
from collections import namedtuple
import yaml
from pathlib import Path
import math
import time

import numpy as np
import argparse
from box import Box
import jax
from jax import jit, vmap, pmap
import jax.numpy as jnp
from jax.tree_util import tree_map
import haiku as hk

import pyglet
from pyglet.gl import *
from pyglet.window import key

from nerf import loader, sampler, Intrinsics
from train_sdrf import init_feature_grids
from sdrf import SDRF, SDRFGrid, SDRFParams, FeatureGrid, run_one_iter_of_sdrf
from util import get_ray_bundle, FirstPersonCamera, restore


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def render(height, width, config, sdrf_grid, ps, pose, intrinsics, iteration, rng):
    uv, ray_origins, ray_directions = get_ray_bundle(
        height,
        width,
        intrinsics.focal_length,
        pose,
    )

    sdrf_f = SDRF(
        geometry=lambda pt: sdrf_grid.geometry(pt, ps.geometry),
        ddf=lambda pt: sdrf_grid.ddf(pt, ps.geometry),
        appearance=lambda pt, view: sdrf_grid.appearance(pt, view, ps.appearance),
    )

    rgb_root, z_vals = run_one_iter_of_sdrf(
        sdrf_f,
        ray_origins,
        ray_directions,
        iteration,
        intrinsics,
        config.nerf.validation,
        config.sdrf,
        config.dataset.projection,
        rng,
        True,
    )
    return rgb_root


class Renderer(object):
    def __init__(
        self,
        config,
        sdrf_grid,
        sdrf_params,
        focal,
        iteration,
        rng,
        width=200,
        height=200,
        window_width=1000,
        window_height=1000,
        title="Renderer",
        fps=30,
        show_fps=False,
    ):
        """Initialize and run."""
        self.config, self.sdrf_grid = config, sdrf_grid
        self.sdrf_params = sdrf_params
        self.width, self.height = width, height
        self.window_width, self.window_height = window_width, window_height
        self.window = pyglet.window.Window(
            self.window_width, self.window_height, fullscreen=False
        )
        self.intrinsics = Intrinsics(
            focal_length=focal, width=window_width, height=window_height
        )
        self.rng = rng
        self.camera = FirstPersonCamera(self.window)
        self.iteration = iteration

        self.window.set_exclusive_mouse(True)

        @self.window.event
        def on_draw():
            self.window.clear()

            pose = self.camera.view_matrix

            start = time.time()
            rgb_root = render(
                self.height,
                self.width,
                self.config,
                self.sdrf_grid,
                self.sdrf_params,
                pose,
                self.intrinsics,
                self.iteration,
                self.rng,
            )
            end = time.time()
            print("render time", end - start)
            rgb_root = jax.image.resize(
                rgb_root, (self.window_width, self.window_height, 3), method="nearest"
            )

            rgb_root_np = np.array((rgb_root * 256.0).astype(jnp.uint8).ravel())
            del rgb_root

            image_data = pyglet.image.ImageData(
                self.window_width, self.window_height, "RGB", rgb_root_np.ctypes.data
            )

            image_data.blit(0, 0)

        @self.window.event
        def on_activate():
            self.window.set_exclusive_mouse(True)

        @self.window.event
        def on_deactivate():
            self.window.set_exclusive_mouse(False)

        def on_update(dt):
            self.camera.update(dt)

        pyglet.clock.schedule(on_update)
        pyglet.app.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    assert logdir.exists()

    with open(logdir / "config.yml", "r") as f:
        config_dictionary = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_dictionary, frozen_box=True, box_it_up=True)

    # set up rng
    rng = jax.random.PRNGKey(config.experiment.seed)
    rng, *subrng = jax.random.split(rng, 3)

    checkpoint_dir = logdir / "checkpoint"
    assert checkpoint_dir.exists()

    checkpoint_i = (int(f.stem) for f in checkpoint_dir.iterdir() if f.is_dir())
    last_checkpoint = max(checkpoint_i)
    checkpoint_subdir = checkpoint_dir / str(last_checkpoint)
    assert checkpoint_subdir.exists()

    print("Opening checkpoint", checkpoint_subdir)

    ps = restore(checkpoint_subdir)
    ps = tree_map(jnp.array, ps)
    sdrf_grid, _ = init_feature_grids(config, subrng[0])

    renderer = Renderer(config, sdrf_grid, ps, 138.0, last_checkpoint, subrng[1])


if __name__ == "__main__":
    main()
