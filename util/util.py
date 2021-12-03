import operator
import functools
from itertools import accumulate
from collections import namedtuple
import os
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.ops import index, index_update
from jax.tree_util import register_pytree_node, tree_map
from box import Box
import matplotlib.pyplot as plt
import mrcfile


def grid_sample(f, grid_min, grid_max, resolution=16):
    dimensions = grid_min.shape[0]

    ds = jnp.stack(
        jnp.meshgrid(
            *tuple(
                jnp.linspace(grid_min[di], grid_max[di], resolution)
                for di in range(dimensions)
            )
        ),
        axis=-1,
    )
    out_shape = ds.shape[:-1]

    if resolution ** dimensions > 2 ** 16:
        return (
            ds,
            jax.lax.map(lambda grid: f(grid.reshape(-1, dimensions)), ds).reshape(
                *out_shape, -1
            ),
        )
    else:
        return ds, f(ds.reshape(-1, dimensions)).reshape(*out_shape, -1)


def plot_iso(f, grid_min, grid_max, resolution=16):
    ds, d = jax.jit(grid_sample, static_argnums=(0, 3))(
        f, grid_min, grid_max, resolution
    )
    plt.contour(
        ds[:, :, 0],
        ds[:, :, 1],
        d.squeeze(),
        levels=[
            0,
        ],
    )


def plot_heatmap(f, grid_min, grid_max, resolution=16):
    ds, d = jax.jit(grid_sample, static_argnums=(0, 3))(
        f, grid_min, grid_max, resolution
    )
    only_valid = lambda a: jnp.clip(a, a_min=0, a_max=1)
    if d.shape[-1] == 2:
        d = jnp.concatenate((d, jnp.zeros((*d.shape[:-1], 1))), axis=-1)
    if d.shape[-1] == 1 or d.ndim == 2:
        if d.ndim == 3:
            d = d[..., 0]
        d = jnp.stack([only_valid(d), only_valid(-d), jnp.zeros_like(d)], axis=-1)
    plt.imshow(abs(d))


def get_ray_bundle(height, width, focal_length, tfrom_cam2world):
    ii, jj = jnp.meshgrid(
        jnp.arange(
            width,
            dtype=jnp.float32,
        ),
        jnp.arange(
            height,
            dtype=jnp.float32,
        ),
        indexing="xy",
    )

    directions = jnp.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -jnp.ones_like(ii),
        ],
        axis=-1,
    )

    ray_directions = jnp.sum(
        directions[..., None, :] * tfrom_cam2world[:3, :3], axis=-1
    )
    ray_origins = jnp.broadcast_to(tfrom_cam2world[:3, -1], ray_directions.shape)

    uv = jnp.stack(
        jnp.meshgrid(
            jnp.arange(ray_origins.shape[0]),
            jnp.arange(ray_origins.shape[1]),
            ordering="xy",
            dtype=jnp.int64,
        ),
        axis=-1,
    )
    uv = jnp.concatenate((uv, jnp.zeros(uv.shape[:2] + (1,), dtype=jnp.int64)), axis=-1)
    uv = uv.reshape((-1, 3))

    return uv, ray_origins, ray_directions


def get_fan(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        raise Exception("invalid shape for SIREN!")

    return fan_in, fan_out


def map_batched_tuple(tensors, f, chunksize, use_vmap, rng=None):
    """
    Calls map_batched/map_batched_rng on lists/tuple of tensors that share a
    leading dimension
    """
    shapes = tuple(tensor.shape[1:] for tensor in tensors)
    lens = tuple(sum(shape) for shape in shapes)
    offsets = (0,) + tuple(accumulate(lens))

    reshape_arg = (
        lambda arg, i: arg[offsets[i] : offsets[i + 1]].reshape(shapes[i])
        if use_vmap
        else arg[:, offsets[i] : offsets[i + 1]].reshape(arg.shape[:1] + shapes[i])
    )

    reshaped = tuple(tensor.reshape(tensor.shape[0], -1) for tensor in tensors)
    if rng is None:
        return map_batched(
            jnp.concatenate(reshaped, axis=1),
            lambda chunk_args: f(
                *tuple(reshape_arg(chunk_args, i) for i in range(len(offsets) - 1))
            ),
            chunksize,
            use_vmap,
        )
    else:
        return map_batched_rng(
            jnp.concatenate(reshaped, axis=1),
            lambda mixed_args: f(
                *(
                    tuple(
                        reshape_arg(mixed_args[0], i) for i in range(len(offsets) - 1)
                    )
                    + (mixed_args[1],)
                )
            ),
            chunksize,
            use_vmap,
            rng,
        )


# @functools.partial(jit, static_argnums=(1, 2, 3))
def map_batched(tensor, f, chunksize, use_vmap):
    if tensor.shape[0] < chunksize:
        return vmap(f)(tensor) if use_vmap else f(tensor)
    else:
        tensor_diff = -tensor.shape[0] % chunksize
        initial_len = tensor.shape[0]
        tensor_len = tensor.shape[0] + tensor_diff
        extra_dims = tuple((0, 0) for _ in range(len(tensor.shape) - 1))
        tensor = jnp.pad(tensor, ((0, tensor_diff), *extra_dims), "constant")
        tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])

        if use_vmap:
            out = jax.lax.map(lambda chunk: vmap(f)(chunk), tensor)
        else:
            out = jax.lax.map(f, tensor)

        # out = out.reshape(-1, *out.shape[2:])[:initial_len]
        fix_shape = lambda output: output.reshape(-1, *output.shape[2:])[:initial_len]
        if isinstance(out, (tuple, list)):
            out = tuple(fix_shape(output) for output in out)
        else:
            out = fix_shape(out)
        return out


# TODO: This is awful, please rewrite it
# @functools.partial(jit, static_argnums=(1, 2, 3))
def map_batched_rng(tensor, f, chunksize, use_vmap, rng):
    if tensor.shape[0] < chunksize:
        if use_vmap:
            rngs = jax.random.split(rng, tensor.shape[0] + 1)
            rng, subrng = rngs[0], rngs[1:]
            return vmap(f)((tensor, subrng)), rng
        else:
            # rng, subrng = jax.random.split(rng)
            rngs = jax.random.split(rng, tensor.shape[0] + 1)
            rng, subrng = rngs[0], rngs[1:]
            return f((tensor, subrng)), rng
    else:
        tensor_diff = -tensor.shape[0] % chunksize
        initial_len = tensor.shape[0]
        tensor_len = tensor.shape[0] + tensor_diff
        extra_dims = tuple((0, 0) for _ in range(len(tensor.shape) - 1))
        tensor = jnp.pad(tensor, ((0, tensor_diff), *extra_dims), "constant")
        tensor = tensor.reshape(tensor_len // chunksize, chunksize, *tensor.shape[1:])

        if use_vmap:
            rngs = jax.random.split(rng, tensor.shape[0] * tensor.shape[1] + 1)
            rng, subrng = rngs[0], rngs[1:]
            subrng = jax.random.split(rng, tensor.shape[0] * tensor.shape[1]).reshape(
                *tensor.shape[:2], 2
            )

            out = jax.lax.map(
                lambda chunk: vmap(f)((chunk[0], chunk[1])), (tensor, subrng)
            )
        else:
            rngs = jax.random.split(rng, tensor.shape[0] * tensor.shape[1] + 1)
            rng, subrng = rngs[0], rngs[1:]
            subrng = jax.random.split(rng, tensor.shape[0] * tensor.shape[1]).reshape(
                *tensor.shape[:2], 2
            )

            out = jax.lax.map(f, (tensor, subrng))

        fix_shape = lambda output: output.reshape(-1, *output.shape[2:])[:initial_len]
        if isinstance(out, (tuple, list)):
            out = tuple(fix_shape(output) for output in out)
        else:
            out = fix_shape(out)
        return out, rng


def normalize(a):
    return a / jnp.linalg.norm(a, ord=2, axis=-1)


def look_at(eye, center, up):
    n = normalize(center - eye)
    u = normalize(up)
    v = normalize(jnp.cross(n, u))
    u = jnp.cross(v, n, axis=-1)

    out = jnp.zeros((4, 4))

    out = index_update(out, jax.ops.index[0, :3], v)
    out = index_update(out, jax.ops.index[1, :3], u)
    out = index_update(out, jax.ops.index[2, :3], -n)

    out = index_update(out, jax.ops.index[0, 3], jnp.dot(-1.0 * v, eye))
    out = index_update(out, jax.ops.index[1, 3], jnp.dot(-1.0 * u, eye))
    out = index_update(out, jax.ops.index[2, 3], jnp.dot(1.0 * n, eye))
    out = index_update(out, jax.ops.index[3, 3], 1.0)

    return out


def img2mse(img_src, img_tgt):
    return ((img_src.ravel() - img_tgt.ravel()) ** 2.0).sum()


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


def serialize_box(base_name, box):
    box_namedtuple_cls = namedtuple(base_name, tuple(box.keys()))

    child_nodes = {
        key: serialize_box(key, value) if isinstance(value, Box) else value
        for key, value in box.items()
    }

    box_namedtuple = box_namedtuple_cls(**child_nodes)

    return box_namedtuple


def gradient_visualization(g, min_val=None, max_val=None):
    if min_val is not None and max_val is not None:
        pass
    elif len(g.shape) > 2:
        min_val, max_val = jnp.amin(g, axis=(0, 1)), jnp.amax(g, axis=(0, 1))
    else:
        min_val, max_val = jnp.min(g.ravel()), jnp.max(g.ravel())

    scaled = (g - min_val) / (max_val - min_val + 1e-9)

    return scaled


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_mrc(filename, fn, grid_min, grid_max, resolution=256, batch_size=2 ** 13):
    dimensions = grid_min.shape[0]
    sdf = jax.jit(grid_sample, static_argnums=(0, 3))(
        fn, grid_min, grid_max, resolution=resolution
    )[1][..., 0]
    sdf = np.array(sdf)

    with mrcfile.new_mmap(
        filename, overwrite=True, shape=(resolution,) * dimensions, mrc_mode=2
    ) as mrc:
        mrc.data[:] = -sdf
