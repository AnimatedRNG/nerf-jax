#!/usr/bin/env python3


import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp


def pad_or_cut(tensor, nd):
    if nd < tensor.shape[0]:
        return tensor[:nd]
    else:
        return jnp.pad(
            tensor,
            ((0, nd - tensor.shape[0]),) + ((0, 0),) * (len(tensor.shape) - 1),
            "constant",
            constant_values=0,
        )


def eval_static(fn, nd, n, *vectorized_args, **other_args):
    vectorized_args = tuple(
        jax.tree_util.tree_map(lambda v: pad_or_cut(v, nd), va)
        for va in vectorized_args
    )
    fn_out = vmap(lambda va: fn(*va, **other_args))(vectorized_args)

    static_output = jax.tree_util.tree_map(lambda o: pad_or_cut(o, n), fn_out)
    return static_output


def dvmap(fn, n, *vectorized_args, num_segments=10, **other_args):
    """
    Vaguely analogous to `vmap` or `jax.lax.map`, but will only process
    >= `n` elements. Creates multiple kernels and selects
    the right one at runtime.

    `vmap`-ping `dvmap` will incur worst-case behavior. See
    https://github.com/google/jax/issues/2947 for more details.

    :param fn:                  mapped function, applied to leading axis
    :param n:                   length of the dynamically-sized input
    :param num_segments:        number of kernels to generate
    :param *vectorized_args:    similar to arguments to `vmap`
    :param **other_args:        passed directly into fn
    """
    flattened = jax.tree_util.tree_flatten(vectorized_args[0])[0]
    array_size = flattened[0].shape[0]
    assert all(va.shape[0] == array_size for va in flattened)

    segment_size = array_size / num_segments

    return jax.lax.switch(
        jnp.ceil(n / segment_size).astype(int),
        [
            lambda _, i=i: eval_static(
                fn, int(i * segment_size), array_size, *vectorized_args, **other_args
            )
            for i in range(num_segments + 1)
        ],
        n,
    )


def reorder_pytree(xs, indices):
    return jax.tree_util.tree_map(
        # lambda leaf_xs: leaf_xs[indices, ...], xs
        lambda leaf_xs: jnp.take(leaf_xs, indices, axis=0),
        xs,
    )


def dvmap_while(cond, body, xs, max_iters=30, num_segments=10, use_dvmap=True):
    tensor_shape = jax.tree_util.tree_flatten(xs)[0][0].shape
    max_length = tensor_shape[0]

    # print([xs.shape for x in xs])

    initial_uvs = jnp.arange(max_length, dtype=jnp.uint32)
    original_uvs = jnp.array(initial_uvs)

    def for_body_fn(_, carry):
        initial_xs, uvs = carry

        # Every value is True if we want to keep iterating and False otherwise
        mask = vmap(cond)(initial_xs)

        # Move True values to the start and False values to the end.
        # A full sort rather than partitioning is definitely somewhat wasteful.
        # On the TPU target this takes ~14ms for 30 iterations over 2 ** 14
        # elements.
        indices = jnp.argsort(~mask)

        # re-order the input accordingly
        staging_xs = reorder_pytree(initial_xs, indices)

        # and do the same thing for the uvs
        staging_uvs = reorder_pytree(uvs, indices)

        num_valid = mask.sum()

        # perform the body
        if use_dvmap:
            output_xs = dvmap(body, num_valid, staging_xs, num_segments=num_segments)
        else:
            output_xs = vmap(body)(staging_xs)

        tree_select = lambda a, b, mi: jax.tree_util.tree_multimap(
            lambda a_t, b_t: jax.lax.select(mi < num_valid, a_t, b_t), a, b
        )

        combined_xs = vmap(
            lambda output_xs_i, staging_xs_i, mi: tree_select(
                output_xs_i, staging_xs_i, mi
            )
        )(output_xs, staging_xs, original_uvs)

        return combined_xs, staging_uvs

    final_xs, final_uvs = jax.lax.fori_loop(
        0, max_iters, for_body_fn, (xs, initial_uvs)
    )

    return reorder_pytree(final_xs, jnp.argsort(final_uvs))
