import pytest

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad

from util import dvmap, dvmap_while


def test_dvmap_equality():
    num_devices = jax.local_device_count()

    inp_len = 2 ** 12

    inp = jnp.array(np.random.random((num_devices, inp_len)))

    # If you couple dvmap with pmap, you likely want to JIT through the pmap, despite
    # https://github.com/google/jax/issues/2926
    jit_dvmap = jax.jit(
        lambda fn, n, xd: pmap(lambda x: dvmap(fn, n, x))(xd), static_argnums=(0,)
    )
    jit_vmap = jax.jit(
        lambda fn, xd: pmap(lambda x: vmap(fn)(x))(xd), static_argnums=(0,)
    )

    fn = jnp.exp

    ref = jit_vmap(fn, inp)

    step = inp_len // 16
    for i in range(step, inp_len, step):
        assert jnp.allclose(jit_dvmap(jnp.exp, i, inp)[:, :i], ref[:, :i])


def test_dvmap_while_sqrt():
    np.random.seed(1)
    # the index parameter is unnecessary, we just have it so that
    # the standard JAX while loop will terminate after 100 iterations
    # if it doesn't find anything
    def example_while_cond(carry):
        x, x_prev, a, i = carry
        return (jnp.abs(x - x_prev) > 1e-2) & (i < 100)

    def example_body(carry):
        x, x_prev, a, i = carry
        return 0.5 * (x + a / x), x, a, i + 1

    #elems = np.random.random(2 ** 10) * 100
    elems = np.random.random(10) * 100

    carry_0 = (elems, elems, elems, jnp.zeros_like(elems, dtype=jnp.uint32))
    x_1 = vmap(example_body)(carry_0)

    jit_dvmap_while = jax.jit(dvmap_while, static_argnums=(0, 1, 3))
    jax_while = jax.jit(
        lambda cond, body, xs: vmap(lambda x: jax.lax.while_loop(cond, body, x))(xs),
        static_argnums=(0, 1),
    )

    sqrt_results = jit_dvmap_while(example_while_cond, example_body, x_1, 100)[0]
    ref_results = jax_while(example_while_cond, example_body, x_1)[0]
    true_answer = jnp.sqrt(elems)

    assert jnp.allclose(sqrt_results, ref_results, rtol=1e-2)
    assert jnp.allclose(sqrt_results, true_answer, rtol=1e-2)


def test_dvmap_even_simpler():
    def really_simple_cond(carry):
        return carry > 30

    def really_simple_body(carry):
        return carry - 1

    elems = np.arange(100)
    expected = jnp.concatenate(
        [jnp.arange(31), jnp.ones(69, dtype=elems.dtype) * 30], axis=0
    )

    jit_dvmap_while = jax.jit(dvmap_while, static_argnums=(0, 1, 3))
    assert all(
        expected == jit_dvmap_while(really_simple_cond, really_simple_body, elems, 100)
    )
