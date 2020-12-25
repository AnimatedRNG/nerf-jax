#!/usr/bin/env python3

import os
from pathlib import Path
import json
from io import StringIO
import functools
from collections import namedtuple

from frozendict import frozendict
import numpy as np
import imageio
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from util import get_ray_bundle


Intrinsics = namedtuple("Intrinsics", ["focal_length", "width", "height"])
register_pytree_node(
    Intrinsics, lambda xs: (tuple(xs), None), lambda _, xs: Intrinsics(*xs)
)


def filter_chain(img, options):
    # first, normalize from [0.0, 1.0]
    img = img.astype(jnp.float32) / 255.0

    # next, resize to half-resolution
    if options.downscale != 1:
        factor = options.downscale
        img = jax.image.resize(
            img,
            shape=(img.shape[0] // factor, img.shape[1] // factor, img.shape[2]),
            method=jax.image.ResizeMethod.LINEAR,
            antialias=True,
        )

    if options.white_background:
        img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])

    return img[:, :, :3]


def loader(data_dir, filter_chain_options, device):
    """
    Loads images from disk into a big numpy array, which will
    later be pmapped onto all devices.
    """
    splits = [entry for entry in data_dir.iterdir() if entry.is_dir()]

    metadata = {
        split: json.load(
            StringIO((data_dir / f"transforms_{split.name}.json").read_text())
        )
        for split in splits
    }

    vmap_filter_chain = vmap(
        lambda imgs: jit(filter_chain, static_argnums=(1,), device=device)(
            imgs, filter_chain_options
        ),
    )

    frame_iterator = lambda f, mdata: jnp.stack(
        [
            f(frame)
            for idx, frame in enumerate(mdata["frames"])
            if idx % filter_chain_options.skiptest == 0
        ],
        axis=0,
    )

    images = {
        split.name: vmap_filter_chain(
            jnp.array(
                frame_iterator(
                    lambda frame: imageio.imread(
                        data_dir / f"{frame['file_path']}.png"
                    ),
                    mdata,
                )
            )
        )
        for split, mdata in metadata.items()
    }

    poses = {
        split.name: frame_iterator(
            lambda frame: jnp.array(frame["transform_matrix"]), mdata
        )
        for split, mdata in metadata.items()
    }

    intrinsics = frozendict(
        {
            split.name: Intrinsics(
                focal_length=0.5
                * images[split.name].shape[2]
                / np.tan(0.5 * float(mdata["camera_angle_x"])),
                width=images[split.name].shape[2],
                height=images[split.name].shape[1],
            )
            for split, mdata in metadata.items()
        }
    )

    return images, poses, intrinsics


@functools.partial(jit, static_argnums=(2, 4))
def sampler(img_target, pose, intrinsics, rng, options):
    """
    Given a single image, samples rays
    """
    pose_target = pose[:3, :4]

    ray_origins, ray_directions = get_ray_bundle(
        intrinsics.height, intrinsics.width, intrinsics.focal_length, pose_target
    )

    coords = jnp.stack(
        jnp.meshgrid(
            jnp.arange(intrinsics.height), jnp.arange(intrinsics.width), indexing="xy"
        ),
        axis=-1,
    ).reshape((-1, 2))

    select_inds = jax.random.choice(
        rng, coords.shape[0], shape=(options.num_random_rays,), replace=False
    )
    select_inds = coords[select_inds]

    ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
    ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]

    target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

    return ray_origins, ray_directions, target_s


if __name__ == "__main__":
    from collections import namedtuple

    import cv2

    # TODO: Write real tests for this component

    # example setup with the lego data
    FilterChainOptions = namedtuple(
        "FilterChainOptions", ["skiptest", "downscale", "white_background"]
    )
    example_options = FilterChainOptions(skiptest=1, downscale=2, white_background=True)

    # devices = jax.devices("cpu")
    devices = jax.devices("gpu")

    images, poses, intrinsics = loader(
        Path(".") / "data" / "nerf_synthetic" / "lego",
        example_options,
        devices[0],
    )

    SamplerOptions = namedtuple("SamplerOptions", ["num_random_rays"])
    sampler_options = SamplerOptions(num_random_rays=1024)

    rng = jax.random.PRNGKey(1010)
    ray_origins, ray_directions, target_s = sampler(
        images["train"][0], poses["train"][0], intrinsics["train"], rng, sampler_options
    )

    for image in images["train"]:
        cv2.imshow("img", image[:, :, [2, 1, 0]])
        cv2.waitKey(1)
