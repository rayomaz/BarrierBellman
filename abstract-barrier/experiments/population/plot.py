import json
import math
from typing import Tuple

import matplotlib
import numpy as np
import torch
from bound_propagation import HyperRectangle, IntervalBounds, LinearBounds
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from torch import nn
from tqdm import tqdm, trange

from abstract_barrier.bounds import MinimizeGap, GaussianProbability


def bound_propagation(model, lower_x, upper_x):
    input_bounds = HyperRectangle(lower_x, upper_x)

    ibp_bounds = model.ibp(input_bounds).cpu()
    crown_bounds = model.crown(input_bounds, alpha=False).cpu()

    input_bounds = input_bounds.cpu()

    return input_bounds, ibp_bounds, crown_bounds


def plot_partition(model, args, input_bounds, ibp_bounds, crown_bounds, out_i=0):
    x1, x2 = input_bounds.lower, input_bounds.upper

    plt.clf()
    ax = plt.axes(projection='3d')

    x1, x2 = torch.meshgrid(torch.linspace(x1[0].item(), x2[0].item(), 10), torch.linspace(x1[1].item(), x2[1].item(), 10))

    # # Plot IBP
    y1, y2 = ibp_bounds.lower[out_i, 0].item(), ibp_bounds.upper[out_i, 0].item()
    y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

    surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot LBP linear bounds
    y_lower = crown_bounds.lower[0][out_i, 0, 0, 0] * x1 + crown_bounds.lower[0][out_i, 0, 0, 1] * x2 + crown_bounds.lower[1][out_i, 0, 0]
    y_upper = crown_bounds.upper[0][out_i, 0, 0, 0] * x1 + crown_bounds.upper[0][out_i, 0, 0, 1] * x2 + crown_bounds.upper[1][out_i, 0, 0]

    surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y_upper, color='blue', alpha=0.4, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot function
    x1, x2 = input_bounds.lower, input_bounds.upper
    x1, x2 = torch.meshgrid(torch.linspace(x1[0].item(), x2[0].item(), 500), torch.linspace(x1[1].item(), x2[1].item(), 500))
    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(-1, 500, 500)
    y = y[out_i].cpu()

    surf = ax.plot_surface(x1, x2, y, color='red', label='Function to bound', shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # General plot config
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(f'Bound propagation')
    plt.legend()

    plt.show()


@torch.no_grad()
def plot_bounds_2d(factory, dynamics, args, config):
    x_space1 = torch.linspace(-2.0, 2.0, 21, device=args.device)
    cell_width1 = (x_space1[1:] - x_space1[:-1]) / 2
    slice_centers1 = (x_space1[:-1] + x_space1[1:]) / 2

    x_space2 = torch.linspace(-2.0, 2.0, 81, device=args.device)
    cell_width2 = (x_space2[1:] - x_space2[:-1]) / 2
    slice_centers2 = (x_space2[:-1] + x_space2[1:]) / 2

    cell_centers = torch.cartesian_prod(slice_centers1, slice_centers2)
    cell_widths = torch.cartesian_prod(cell_width1, cell_width2)
    lower_x, upper_x = cell_centers - cell_widths, cell_centers + cell_widths
    set = HyperRectangle(lower_x, upper_x)
    safe = dynamics.safe(set.center, set.width / 2)
    set = HyperRectangle(lower_x[safe], upper_x[safe])

    model = analytic_transition_prob(factory, dynamics, set.lower, set.upper, device=args.device)
    linear_bounds = all_linear_bounds(model, set)
    ibp_bounds = model.ibp(set)

    x1_space = torch.linspace(x_space1[0], x_space1[-1], 500)
    x2_space = torch.linspace(x_space2[0], x_space2[-1], 500)
    x1, x2 = torch.meshgrid(x1_space, x2_space)

    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(-1, 500, 500).cpu()

    for j in trange(len(set)):
        # Plot function over entire space
        plt.clf()
        ax = plt.axes(projection='3d')

        surf = ax.plot_surface(x1, x2, y[j], color='red', alpha=0.8)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        circle_unsafe = plt.Circle((0, 0), 2.0, color=sns.color_palette('deep')[0], fill=True, alpha=0.3)
        ax.add_patch(circle_unsafe)
        art3d.pathpatch_2d_to_3d(circle_unsafe, z=0, zdir='z')

        circle_init = plt.Circle((0, 0), 0.5, color=sns.color_palette('deep')[2], fill=True, alpha=0.5)
        ax.add_patch(circle_init)
        art3d.pathpatch_2d_to_3d(circle_init, z=0, zdir='z')

        output_region = set[j]
        output_region = plt.Polygon(np.array([
            [output_region.lower[0].item(), output_region.lower[1].item()],
            [output_region.upper[0].item(), output_region.lower[1].item()],
            [output_region.upper[0].item(), output_region.upper[1].item()],
            [output_region.lower[0].item(), output_region.upper[1].item()]
        ]), color=sns.color_palette('deep')[1], fill=True, alpha=1)
        ax.add_patch(output_region)
        art3d.pathpatch_2d_to_3d(output_region, z=0, zdir='z')

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Transition probability')
        plt.show()

        for i in trange(len(set)):
            lb = linear_bounds[i]
            ib = ibp_bounds[..., i, :]

            if ib.upper[j].item() > 1e-3:
                plot_partition(model, args, set[i], ib, lb, j)


def all_linear_bounds(transition_prob, set):
    linear_bounds = []
    for l, u in tqdm(list(zip(set.lower, set.upper)), desc='In'):
        linear_bounds.append(region_linear_bounds(transition_prob, l, u))

    return linear_bounds


def region_linear_bounds(transition_prob, l, u):
    linear_bounds = transition_prob.crown(HyperRectangle(l.unsqueeze(0), u.unsqueeze(0)), alpha=False)

    return linear_bounds.to(torch.device('cpu'))


def analytic_transition_prob(factory, dynamics, lower, upper, device=None):
    loc, scale = dynamics.v
    loc, scale = loc.to(device), scale.to(device)

    return factory.build(MinimizeGap(
        dynamics,
        GaussianProbability(loc, scale, lower, upper)
    )).to(device)
