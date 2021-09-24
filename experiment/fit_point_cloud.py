import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import util
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.experimental.optimizers import adam, clip_grads
import haiku as hk
from sdrf import Siren
from torch.utils.tensorboard import SummaryWriter

import configargparse
from torch.utils.data import Dataset
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import mrcfile


p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# General training options
p.add_argument('--batch_size', type=int, default=1024)
p.add_argument('--hidden_features', type=int, default=128)
p.add_argument('--hidden_layers', type=int, default=4)
p.add_argument('--name', type=str, default='pointcloud',
               help='path to directory where checkpoints & tensorboard events will be saved.')

p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')

p.add_argument('--validation_skip', type=int, default=100,
               help='validation.')
p.add_argument('--num_epochs', type=int, default=1000,
               help='number of training epochs.')
p.add_argument('--pc', type=str, default='../data/bunny.ply', help='path to pointcloud')

# logging options
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')

opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        mesh = o3d.io.read_triangle_mesh(pointcloud_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        print("Finished loading point cloud")

        coords = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.ones((total_samples, 1))  # on-surface = 1
        sdf[self.on_surface_points:, :] = 0  # off-surface = 0

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': coords}, {'sdf': sdf, 'normals': normals}


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = pixel_coords.reshape(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.transpose(1, 0).reshape(channels, height, width)


def make_contour_plot(array_2d, mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode == 'log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode == 'lin'):
        num_levels = 10
        levels = np.linspace(-.5, .5, num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_sdf_summary(model_fn, params, writer, total_steps, prefix='train_'):

    eval_fn = lambda x: model_fn.apply(params, x)

    slice_coords_2d = get_mgrid(512)

    yz_slice_coords = np.concatenate((np.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), axis=-1)

    sdf_values = vmap(eval_fn)(yz_slice_coords)
    sdf_values = lin2img(sdf_values).squeeze()
    fig = make_contour_plot(sdf_values)
    writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

    xz_slice_coords = np.concatenate((slice_coords_2d[:, :1],
                                      np.zeros_like(slice_coords_2d[:, :1]),
                                      slice_coords_2d[:, -1:]), axis=-1)

    sdf_values = vmap(eval_fn)(xz_slice_coords)
    sdf_values = lin2img(sdf_values).squeeze()
    fig = make_contour_plot(sdf_values)
    writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

    xy_slice_coords = np.concatenate((slice_coords_2d[:, :2],
                                      -0.75*np.ones_like(slice_coords_2d[:, :1])), axis=-1)

    sdf_values = vmap(eval_fn)(xy_slice_coords)
    sdf_values = lin2img(sdf_values).squeeze()
    fig = make_contour_plot(sdf_values)
    writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

    N = 128
    slice_coords_3d = get_mgrid(N, dim=3)
    sdf_values = np.zeros((N**3, 1))
    bsize = int(64**2)
    for i in tqdm(range(int(N**3 / bsize))):
        coords = slice_coords_3d[i*bsize:(i+1)*bsize, :]
        sdf_values[i*bsize:(i+1)*bsize] = eval_fn(coords)

    sdf_values = sdf_values.reshape(N, N, N)
    with mrcfile.new_mmap('sdf_values.mrc', overwrite=True, shape=(N, N, N), mrc_mode=2) as mrc:
        mrc.data[:] = sdf_values


def cosine_similarity(a, b, eps=1e-8):
    return jnp.dot(a, b) / jnp.maximum((jnp.linalg.norm(a)*jnp.linalg.norm(b)), eps)


def compute_loss(params, pts, model_fn, grad_model_fn, sdfs, normals):
    def loss_fn(pt, sdf, normal):
        model_output = model_fn.apply(params, pt)
        grad_output = grad_model_fn(params, pt)[0]

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        surface_loss = (sdf * model_output)[0]
        offsurface_loss = ((1 - sdf) * jnp.exp(-1e2 * jnp.abs(model_output)))[0]
        normal_loss = (sdf * (1 - cosine_similarity(grad_output, normal)))[0]
        eikonal_loss = jnp.linalg.norm(grad_output) - 1

        return jnp.array([surface_loss, offsurface_loss, normal_loss, eikonal_loss])

    losses = jnp.mean(vmap(loss_fn)(pts, sdfs, normals)**2, axis=0)

    return losses[0] * 3e3 + losses[1] * 1e2 + losses[2] * 1e2 + losses[3] * 5e1, losses


def train():
    # set up model
    rng = jax.random.PRNGKey(1024)
    model_fn = hk.transform(lambda x: Siren(3, 1, opt.hidden_layers, opt.hidden_features)(x))
    params = model_fn.init(rng, jnp.ones([3, ]))
    model_fn = hk.without_apply_rng(model_fn)

    # set up optimizer
    init_adam, update, get_params = adam(lambda _: opt.lr)
    optimizer_state = init_adam((params))

    # batched model and gradient evaluation
    grad_model_fn = grad(lambda params, pt: model_fn.apply(params, pt)[0], argnums=(1,))

    # set up loss function
    value_and_grad_loss_fn = jit(grad(compute_loss, argnums=(0,), has_aux=True))

    jnp.set_printoptions(precision=4, suppress=True)

    # dataset
    dataset = PointCloud(opt.pc, opt.batch_size)

    # logging
    summaries_dir = os.path.join(opt.logging_root, opt.name, 'summaries')
    util.cond_mkdir(summaries_dir)
    writer = SummaryWriter(summaries_dir)
    loss_names = ['surface', 'off-surface', 'normal', 'eikonal']

    # train
    total_steps = 0
    pbar = tqdm(total=len(dataset) * opt.num_epochs)

    for epoch in range(opt.num_epochs):
        for model_input, gt in dataset:
            params = get_params(optimizer_state)

            rng, subrng = jax.random.split(rng)

            pts = model_input['coords']

            with jax.disable_jit():
                gradient, losses = value_and_grad_loss_fn(params, pts, model_fn, grad_model_fn,
                                                          gt['sdf'], gt['normals'])

            losses = tuple(np.array(loss) for loss in losses)
            gradient = clip_grads(gradient, 1.0)

            for loss_val, name in zip(losses, loss_names):
                writer.add_scalar(name, loss_val, total_steps)

            if total_steps % opt.validation_skip == 0:
                write_sdf_summary(model_fn, params, writer, total_steps)
                tqdm.write(f'Epoch {epoch}, Loss {losses[0]:.02e}, {losses[1]:.02e}, {losses[2]:.02e}, {losses[3]:.02e}')

            optimizer_state = update(epoch, gradient[0], optimizer_state)

            total_steps += 1
            pbar.update(1)


if __name__ == '__main__':
    train()
