import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

DEFAULT_FIGSIZE = (8, 8)
DEFAULT_VIEW_INIT = (0, 0)


def get_default_cbar_kwargs(figsize):
    return {'num_ticks': 6, 'decimals': 4, 'labelpad': figsize[0] * 2,
            'labelsize': figsize[0] * 1.5, 'fontsize': figsize[0],
            'fraction': 0.1, 'shrink': 0.5}


def get_fig_ax(**kwargs):
    """
    Default setting for matplotlib figure
    :param kwargs: kwargs for matplotlib figure
        figsize : (float, float) figure size
        view_init : (float, float) view init from matplotlib
        show_axis : <bool> If True, axis are visible. (default : False)
        show_colorbar : <bool> If True, colorbar are visible. (default : True)
    :return: <matplotlib.figure>, <matplotlib.axes>, <matplotlib.cmap>
    """
    figsize = kwargs.get('figsize', DEFAULT_FIGSIZE)
    view_init = kwargs.get('view_init', DEFAULT_VIEW_INIT)
    show_axis = kwargs.get('axis', False)

    # Set default setting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(*view_init)
    ax.set_proj_type("ortho")
    ax.grid(visible=False)
    if not show_axis:
        ax.set_axis_off()

    return fig, ax


def set_fig_ax(ax, **kwargs):
    if not kwargs:
        return
    if view_init := kwargs.get('view_init'):  # view_init
        ax.view_init(*view_init)
    if show_axis := kwargs.get('show_axis'):  # show axis
        if show_axis:
            ax.set_axis_on()
        else:
            ax.set_axis_off()


def get_cmap(cmap=None):
    if isinstance(cmap, str):
        return sns.color_palette(cmap, as_cmap=True)
    elif isinstance(cmap, ListedColormap):
        return cmap
    elif cmap is None:
        middle_color = np.array([162, 219, 183, 256 * 1]) / 256
        start_color = np.array([1, 1, 1, 1])
        end_color = np.array([0, 0, 0, 1])
        idx = 256 // 2

        cmap1 = np.linspace(start_color, middle_color, num=idx)
        cmap2 = np.linspace(middle_color, end_color, num=256 - idx)
        cmap = np.concatenate((cmap1, cmap2), axis=0)
        custom_cmap = ListedColormap(cmap)
        return custom_cmap
    else:
        raise TypeError(f'cmap must be str, ListedColormap, or None, not {type(cmap)}')


def set_axes_equal(ax, scale_factor=1):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc...  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    code from 'https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to'

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.3 / scale_factor * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
