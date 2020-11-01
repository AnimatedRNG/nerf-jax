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
from rendering import SDRFParams, SDRF

Losses = namedtuple("Losses", ["rgb_loss", "eikonal_loss", "inter_loss"])
register_pytree_node(Losses, lambda xs: (tuple(xs), None), lambda _, xs: Losses(*xs))


def init_networks(config, rng):
    geometry_fn = hk.transform(lambda x: Siren(3, 1, 4, 32, False, "relu")(x))
    appearance_fn = hk.transform(lambda x: Siren(6, 3, 4, 32, False, "relu")(x))

    geometry_params = geometry_fn.init(rng[0], jnp.ones([3,]))
    appearance_params = appearance_fn.init(rng[0], jnp.ones([6,]))

    return (
        SDRF(
            geometry=lambda pt, params: geometry_fn.apply(params, pt),
            appearance=lambda pt, rd, params: appearance_fn.apply(params, pt, rd),
        ),
        SDRFParams(geometry=geometry_params, appearance=appearance_params),
    )
