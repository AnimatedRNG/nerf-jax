#!/usr/bin/env python3

import numpy as np

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import sph_harm
from jax import lax

from functools import partial


def y00(theta, phi):
    return jnp.ones_like(theta) * (1.0 / 2.0) * jnp.sqrt(1.0 / jnp.pi)


def y1_1(theta, phi):
    return (1.0 / 2.0) * jnp.sqrt(3.0 / jnp.pi) * jnp.sin(phi) * jnp.sin(theta)


def y10(theta, phi):
    return (1.0 / 2.0) * jnp.sqrt(3.0 / jnp.pi) * jnp.cos(theta)


def y11(theta, phi):
    return (1.0 / 2.0) * jnp.sqrt(3.0 / jnp.pi) * jnp.cos(phi) * jnp.sin(theta)


def y2_2(theta, phi):
    return (
        (1.0 / 2.0)
        * jnp.sqrt(15.0 / jnp.pi)
        * jnp.sin(phi)
        * jnp.cos(phi)
        * jnp.square(jnp.sin(theta))
    )


def y2_1(theta, phi):
    return (
        (1.0 / 2.0)
        * jnp.sqrt(15.0 / jnp.pi)
        * jnp.sin(phi)
        * jnp.sin(theta)
        * jnp.cos(phi)
    )


def y20(theta, phi):
    return (
        (1.0 / 4.0) * jnp.sqrt(5.0 / jnp.pi) * (3.0 * jnp.square(jnp.cos(theta)) - 1.0)
    )


def y21(theta, phi):
    return (
        (1.0 / 2.0)
        * jnp.sqrt(15.0 / jnp.pi)
        * jnp.cos(phi)
        * jnp.sin(theta)
        * jnp.cos(theta)
    )


def y22(theta, phi):
    return (
        (1.0 / 4.0)
        * jnp.sqrt(15.0 / jnp.pi)
        * (jnp.square(jnp.cos(phi)) - jnp.square(jnp.sin(phi)))
        * jnp.square(jnp.sin(theta))
    )


def real_valued_sh(
    ls: np.ndarray, ms: np.ndarray, theta: jnp.ndarray, phi: jnp.ndarray
):
    assert tuple(ls.shape) == tuple(ms.shape)
    y = {
        (0, 0): y00,
        (1, -1): y1_1,
        (1, 0): y10,
        (1, 1): y11,
        (2, -2): y2_2,
        (2, -1): y2_1,
        (2, 0): y20,
        (2, 1): y21,
        (2, 2): y22,
    }
    return jnp.stack([y[l, m](theta, phi)[..., 0] for l, m in zip(ls, ms)], axis=0)


def sample_real_sh(v, coeffs):
    order = np.sqrt(coeffs.shape[-1]).astype(np.int32) - 1
    assert (order + 1) * (order + 1) == coeffs.shape[-1]

    phi = jnp.arctan(
        jnp.sqrt(jnp.square(v[..., 0:1]) + jnp.square(v[..., 1:2])) / v[..., 2:3]
    )
    at = jnp.arctan(v[..., 1:2] / v[..., 0:1])
    theta = jnp.where(
        v[..., 0:1] > 0.0,
        at,
        jnp.where(v[..., 0:1] < 0.0, at + jnp.pi, jnp.pi / 2),
    )

    orders = np.arange(int(jnp.square(order + 1)))
    ls = np.sqrt(orders).astype(jnp.int32)
    ms = ((orders - np.square(ls)) - ls).astype(np.int32)
    return coeffs.dot(real_valued_sh(ls, ms, theta, phi))


if __name__ == "__main__":
    print(
        sample_real_sh(
            jnp.array([0, 0, 1]),
            jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
        )
    )
