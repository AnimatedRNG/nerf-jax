import mrcfile
import matplotlib.pyplot as plt
import numpy as np

def make_contour_plot(array_2d, mode='log', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    else:
        fig = plt.gcf()

    if(mode == 'log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS, ax=ax)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig

with mrcfile.open('test.mrc') as f:
    out = f.data

fig, axs = plt.subplots(1, 3, figsize=(2.75*3, 2.75), dpi=100)

plt.subplot(131)
fig = make_contour_plot(out[128, :, :], ax=axs[0])
plt.subplot(132)
fig = make_contour_plot(out[:, 128, :], ax=axs[1])
plt.subplot(133)
fig = make_contour_plot(out[:, :, 128], ax=axs[2])
plt.show()
