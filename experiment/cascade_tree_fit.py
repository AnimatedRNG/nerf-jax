import os
import sys
import functools

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import grad, vmap
import pywavefront

from sdrf import IGR, CascadeTree, exp_smin
from util import plot_iso, plot_heatmap, create_mrc


def get_model(filename):
    model_scene = pywavefront.Wavefront(filename, collect_faces=True)
    assert model_scene.materials["default0"].vertex_format == "V3F"
    model_vertices = jnp.array(model_scene.materials["default0"].vertices).reshape(
        -1, 3
    )

    # normalize vertices
    model_min, model_max = jnp.min(model_vertices, axis=0), jnp.max(
        model_vertices, axis=0
    )
    scale_factor = jnp.max(model_max - model_min, axis=0)
    model_vertices = (model_vertices - model_min) / (scale_factor)
    model_vertices = model_vertices * 2 - 1

    return model_vertices


def fit(model_vertices):
    rng = jax.random.PRNGKey(1024)

    decoder_fn = hk.transform(lambda x: IGR([32, 32], beta=0)(x))

    feature_size = 16
    ps = decoder_fn.init(
        rng,
        jnp.ones(
            [
                feature_size,
            ]
        ),
    )
    decoder_fn = hk.without_apply_rng(decoder_fn)
    max_depth = 5

    grid_min = jnp.array([-1.0, -1.0, -1.0])
    grid_max = jnp.array([1.0, 1.0, 1.0])

    scene = hk.transform(
        lambda p: CascadeTree(
            [lambda p: decoder_fn.apply(ps, p) for _ in range(max_depth)],
            grid_min=grid_min,
            grid_max=grid_max,
            union_fn=lambda a, b: exp_smin(a, b, 32),
            max_depth=max_depth,
            feature_size=feature_size,
        )(p)
    )
    params = scene.init(
        rng,
        jnp.ones(
            [
                3,
            ]
        ),
    )
    scene = hk.without_apply_rng(scene)
    scene_fn = lambda params, pt: scene.apply(params, pt)[0]

    grad_sample_fn = grad(scene_fn, argnums=(1,))

    lr = 1e-3
    batch_size = 2 ** 13

    init_adam, update, get_params = adam(lambda _: lr)
    optimizer_state = init_adam(params)

    def loss_fn(params, rng):
        on_surface_pts = model_vertices

        off_surface_pts = jax.random.uniform(
            rng, (batch_size // 2, 3), minval=-1.0, maxval=1.0
        )

        total_pts = jnp.concatenate((on_surface_pts, off_surface_pts), axis=0)

        def reconstruction_loss_fn(pt, params):
            model_output = scene_fn(params, pt)

            return (model_output ** 2).sum()

        def eikonal_loss_fn(pt, params):
            grad_sample = grad_sample_fn(params, pt)
            # grad_sample = clip_grads(grad_sample[0], 1.0)
            grad_sample = grad_sample[0]

            return (1.0 - jnp.linalg.norm(grad_sample)) ** 2.0

        def inter_loss_fn(pt, params):
            sdf = scene_fn(params, pt)
            return (1 - sdf) * jnp.exp(-1e2 * jnp.abs(sdf)).sum()

        reconstruction_losses = vmap(lambda pt: reconstruction_loss_fn(pt, params))(
            on_surface_pts
        ).sum()

        eikonal_losses = vmap(lambda pt: eikonal_loss_fn(pt, params))(total_pts).sum()

        inter_losses = vmap(lambda pt: inter_loss_fn(pt, params))(off_surface_pts).sum()

        # losses = (reconstruction_losses, eikonal_losses, inter_losses)
        losses = (reconstruction_losses, eikonal_losses, inter_losses)
        weights = (3e3, 1e2, 5e1)
        return (
            sum(loss_level * weight for loss_level, weight in zip(losses, weights)),
            losses,
        )

    value_loss_fn_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True))
    # value_loss_fn_jit = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)

    for epoch in range(10000):
        rng, subrng = jax.random.split(rng)
        params = get_params(optimizer_state)

        if epoch % 100 == 0:
            plot_iso(
                lambda pt: scene_fn(params, jnp.array([pt[0], 0.0, pt[2]])),
                jnp.array([-1.0, -1.0]),
                jnp.array([1.0, 1.0]),
                256,
            )

            plot_heatmap(
                lambda pt: scene_fn(params, jnp.array([pt[0], 0.0, pt[2]])).repeat(
                    3, axis=-1
                ),
                jnp.array([-1.0, -1.0]),
                jnp.array([1.0, 1.0]),
                256,
            )
            create_mrc(
                "test.mrc", functools.partial(scene_fn, params), grid_min, grid_max, 256
            )

        (loss, losses), gradient = value_loss_fn_jit(params, subrng)
        print(f"epoch {epoch}; loss {loss}, losses: {losses}")

        # gradient = clip_grads(gradient[0], 1.0)
        gradient = gradient[0]

        optimizer_state = update(epoch, gradient, optimizer_state)


if __name__ == "__main__":
    fit(get_model("../data/stanford-bunny.obj"))
