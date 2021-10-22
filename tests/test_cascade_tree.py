import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from sdrf import exp_smin, exp_smax


def test_exp_smin_grads():
    check_grads(exp_smin, (jnp.array(1.0), jnp.array(2.0)), order=1)
    check_grads(exp_smin, (jnp.array(1.0), jnp.array(1.1)), order=1)
    check_grads(exp_smin, (jnp.array(-1.0), jnp.array(-2.0)), order=1)

def test_exp_smax_grads():
    check_grads(exp_smax, (jnp.array(1.0), jnp.array(2.0)), order=1)
    check_grads(exp_smax, (jnp.array(1.0), jnp.array(1.2)), order=1)
