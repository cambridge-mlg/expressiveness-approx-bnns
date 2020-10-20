import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import pickle


""" Constants: Do not change POINTS_PER_AXIS (spacing of points for the 2D 
grid), or NUM_POINTS (number of points on the diagonal slice), or other 
constants, else the results will be inconsistent when saving and loading.
"""
POINTS_PER_AXIS = 40
NUM_POINTS = 500
X1_RANGE = (-2., 2.)
X2_RANGE = (-2., 2.)


matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
font = {'family': 'cmr10', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


def load_X_y(dataset):
    """Load the 2D synthetic dataset.
    Args:
        Dataset (string): 'origin' or 'axis'. origin has clusters diagonally
        across from the origin, axis has clustered parallel to one of the
        input data dimensions.

    Returns:
        X ([N, d_in] numpy array, N=100, d_in=2): Input data.
        y ([N, d_out] numpy array, N=100, d_out=1): Output data.

    """
    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(filepath)
    if dataset == 'origin':
        x_path = Path(dirpath, '../datasets/origin/origin_x.txt')
        y_path = Path(dirpath, '../datasets/origin/origin_y.txt')
    elif dataset == 'axis':
        x_path = Path(dirpath, '../datasets/axis/axis_x.txt')
        y_path = Path(dirpath, '../datasets/axis/axis_y.txt')
    else:
        raise NotImplementedError
    X = np.loadtxt(x_path)
    y = np.loadtxt(y_path)
    y = y[:, None]
    return X, y


def load_2d_pred(pickle_path, hmc=False):
    """Load the 2d predictions from a pickle file, and return the predictive
    means and standard deviations needed to make the 2d plot.
    Args:
        pickle_path (Path object): Pickle file containing 2d prediction data
        hmc (bool): If True, loads the hmc data following the old saving
            format.

    Returns:
        contour_std ([POINTS_PER_AXIS, POINTS_PER_AXIS] numpy array):
            Predictive standard deviation on the 2D grid.
        slice_mean ([NUM_POINTS] numpy array): Predictive mean along the slice
            in 2d input space.
        slice_std ([NUM_POINTS] numpy array): Predictive standard deviation
            along the slice in 2d input space.
    """
    if hmc:  # data saved in old format
        with open(pickle_path, 'rb') as f:
            inputs, outputs, pred_mean, pred_var, lambdas, xlambdas, \
            pred_mean_lambdas, pred_var_lambdas = pickle.load(f)
        contour_std = np.sqrt(pred_var)
        contour_std = contour_std.reshape(POINTS_PER_AXIS, POINTS_PER_AXIS)
        slice_mean = pred_mean_lambdas
        slice_std = np.sqrt(pred_var_lambdas)
    else:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        contour_std = data['contour_std']
        slice_mean = data['slice_mean']
        slice_std = data['slice_std']
    return contour_std, slice_mean, slice_std


def gen_grid_inputs(points_per_axis, x1_range, x2_range):
    """Generate input locations for 2D contour plot.
    Args:
        points_per_axis (int): Number of points per axis. Do not change for
            consistency.
        x1_range (tuple): (x1_min, x1_max)
        x2_range (tuple): (x2_min, x2_max)

    Returns:
        inputs_flattened ([points_per_axis ** 2, 2] numpy array):
            Input locations for prediction on the 2d grid.
        x1_grid ([points_per_axis, points_per_axis] numpy array):
            x1 coordinates, output of meshgrid.
        x2_grid: ([points_per_axis, points_per_axis] numpy array):
            x2 coordinates, output of meshgrid.
    """
    x1 = np.linspace(x1_range[0], x1_range[1], points_per_axis)
    x2 = np.linspace(x2_range[0], x2_range[1], points_per_axis)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x1_flattened = x1_grid.reshape(-1)
    x2_flattened = x2_grid.reshape(-1)
    inputs_flattened = np.stack((x1_flattened, x2_flattened), axis=-1)
    return inputs_flattened, x1_grid, x2_grid


def get_slice_points(dataset):
    """Compute data relevant to the slice through 2d input space
    Args:
        dataset (string): 'origin' or 'axis'. Origin has a diagonal slice
            through the origin, axis has a vertical slice that is off-centre.

    Returns:
        slice_points ([NUM_POINTS, 2] numpy array): 2d input points along the
            slice.
        slice_param ([NUM_POINTS] numpy array): Parameter along the slice,
            distance travelled in the slice direction.
        unit_vec ([1, 2] numpy array): Unit vector in the direction of the
            slice.
    """
    if dataset == 'origin':
        len_slice = 4. * np.sqrt(2.)
        unit_vec = np.array([1 / np.sqrt(2.), 1 / np.sqrt(2.)])[None, :]
        offset = np.array([0., 0.])[None, :]
    elif dataset == 'axis':
        len_slice = 4.
        unit_vec = np.array([0., 1.])[None, :]
        offset = np.array([0.5, 0.])[None, :]
    else:
        raise NotImplementedError
    slice_param = np.linspace(-len_slice / 2., len_slice / 2., NUM_POINTS)
    slice_points = slice_param[:, None] * unit_vec + offset
    return slice_points, slice_param, unit_vec


def get_2d_pred(model, dataset):
    """Get the predictive means and standard deviations from the model needed
    to make the 2d plot.
    Args:
        model (BNN): model to get predictions from.
        dataset (string): 'origin' or 'axis'.

    Returns:
        contour_std ([POINTS_PER_AXIS, POINTS_PER_AXIS] numpy array):
            Predictive standard deviation on the 2D grid.
        slice_mean ([NUM_POINTS] numpy array): Predictive mean along the slice
            in 2d input space.
        slice_std ([NUM_POINTS] numpy array): Predictive standard deviation
            along the slice in 2d input space.
    """
    inputs_flattened, _, _ = gen_grid_inputs(POINTS_PER_AXIS,
                                             X1_RANGE, X2_RANGE)
    mean, std = model.pred_mean_std(inputs_flattened)
    std = std[:, 0]
    contour_std = std.reshape(POINTS_PER_AXIS, POINTS_PER_AXIS)

    slice_points, _, _ = get_slice_points(dataset)
    mean, std = model.pred_mean_std(slice_points, num_samples=500)
    slice_mean = mean[:, 0]
    slice_std = std[:, 0]
    return contour_std, slice_mean, slice_std


def make_2d_plot(contour_std, slice_mean, slice_std, dataset, figpath):
    """Plot and save the 2d plot given the prediction data.
    Args:
        contour_std ([POINTS_PER_AXIS, POINTS_PER_AXIS] numpy array):
            Predictive standard deviation on the 2D grid.
        slice_mean ([NUM_POINTS] numpy array): Predictive mean along the slice
            in 2d input space.
        slice_std ([NUM_POINTS] numpy array): Predictive standard deviation
            along the slice in 2d input space.
        dataset (string): 'origin' or 'axis'
        figpath (Path object): Path to figure to save.
    """
    X, y = load_X_y(dataset)
    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           gridspec_kw={'height_ratios': [3, 1]},
                           figsize=(6.1, 7.8))
    make_contour_plot(fig, ax[0], contour_std, X, dataset)
    _, slice_param, unit_vec = get_slice_points(dataset)
    make_slice_plot(ax[1], slice_mean, slice_std, slice_param, unit_vec,
                    X, y, dataset)
    plt.subplots_adjust(left=0.15, top=.9, hspace=0.25)
    plt.savefig(figpath)
    plt.close()


def make_contour_plot(fig, ax, std, X, dataset):
    """Add the 2d contour plot to a figure.
    Args:
        fig: Figure handle
        ax: Axis handle
        std ([POINTS_PER_AXIS, POINTS_PER_AXIS] numpy array): Predictive
            standard deviation on the 2D grid.
        X ([N, d_in] numpy array, N=100, d_in=2): Input data.
        dataset (string): 'origin' or 'axis'
    """
    _, x1_grid, x2_grid = gen_grid_inputs(POINTS_PER_AXIS,
                                                         X1_RANGE, X2_RANGE)
    cnt = ax.contourf(x1_grid, x2_grid, std, levels=200)
    for c in cnt.collections:
        c.set_edgecolor("face") # get rid of contour lines

    # get the colorbar in the right place
    bbox_ax_top = ax.get_position()
    cbar_ax = fig.add_axes([0.9, bbox_ax_top.y0 + 0.02, 0.02,
                            bbox_ax_top.y1 - bbox_ax_top.y0])
    cbar_ax.tick_params(labelsize=14)
    plt.colorbar(cnt, cax=cbar_ax, format='%0.2f')

    # add dashed lines for slices
    if dataset == 'origin':
        line_x = [-2., 2.]
        line_y = [-2., 2.]
    elif dataset == 'axis':
        line_x = [.5, .5]
        line_y = [-2., 2.]
    else:
        raise NotImplementedError
    ax.plot(line_x, line_y, 'w--')

    ax.scatter(X[:, 0], X[:, 1], marker='+', color='red')
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('$\sigma[f(\mathbf{x})]$')


def make_slice_plot(ax, mean, std, slice_param, unit_vec, X, y, dataset):
    """Add the prediction-along-a-slice plot to a figure.
    Args:
        ax: Axis handle
        mean ([NUM_POINTS] numpy array): Predictive mean along the slice
            in 2d input space.
        std ([NUM_POINTS] numpy array): Predictive standard deviation
            along the slice in 2d input space.
        slice_param ([NUM_POINTS] numpy array): Parameter along the slice,
            distance travelled in the slice direction.
        unit_vec ([1, d_in] numpy array): Unit vector in the direction of the
            slice.
        X ([N, d_in] numpy array): Input data.
        y ([N, d_out] numpy array): Output data.
        dataset (string): 'origin' or 'axis'
    """
    ax.plot(slice_param, mean)
    ax.fill_between(slice_param, mean + 2 * std, mean - 2 * std, alpha=0.3)

    # Plot the projection of the datapoints onto the slice
    X_projected = X @ np.transpose(unit_vec)
    X_projected = X_projected[:, 0]
    ax.scatter(X_projected, y[:, 0], marker='+', color='red')

    if dataset == 'origin':
        ax.set_xlim([-2 * np.sqrt(2), 2 * np.sqrt(2)])
        ax.set_ylim([-6, 6])
    elif dataset == 'axis':
        ax.set_xlim([-2, 2])
        ax.set_ylim([-3, 8])
    else:
        pass
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$f(\mathbf{x}(\lambda))$')


def save_2d(contour_std, slice_mean, slice_std, pickle_path):
    """Save the 2d prediction data as a dictionary.
    Args:
        contour_std ([POINTS_PER_AXIS, POINTS_PER_AXIS] numpy array):
            Predictive standard deviation on the 2D grid.
        slice_mean ([NUM_POINTS] numpy array): Predictive mean along the slice
            in 2d input space.
        slice_std ([NUM_POINTS] numpy array): Predictive standard deviation
            along the slice in 2d input space.
        pickle_path (Path object): Path to pickle file to save in.
    """
    data = dict(contour_std=contour_std,
                slice_mean=slice_mean,
                slice_std=slice_std)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle)
