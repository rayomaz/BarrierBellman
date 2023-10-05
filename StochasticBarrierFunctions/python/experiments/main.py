import math
import os
import logging
from argparse import ArgumentParser

import sys
sys.path.append("..")  # Add the parent directory to the import path

import xarray as xr
import torch
from bound_propagation import HyperRectangle, LinearBounds

import matplotlib.pyplot as plt
import scipy.io

from bounds.bounds import MinimizeGap, MinimizePosteriorRect
from bounds.certifier import GaussianCertifier, check_gap
from linear.linear import LinearExperiment
from nndm.nndm import NNDMExperiment
from unicycle.nominal.unicycle import NominalUnicycleExperiment
from unicycle.zeroV.unicycle import ZeroVUnicycleExperiment
from harrier.harrier import HarrierExperiment

from log import configure_logging
from utils import load_config

logger = logging.getLogger(__name__)


def plot_partition(model, args, crown_bounds, mat_bounds=None, points=100, subtract_lower=True):
    # If we subtract the lower bound (from our bounds) then the surface is almost flat,
    # and the plotting area shows much clearer where bounds are too loose.

    x1, x2 = crown_bounds.region.lower, crown_bounds.region.upper
    x1f, x2f = torch.meshgrid(torch.linspace(x1[0], x2[0], points), torch.linspace(x1[1], x2[1], points))
    X = torch.cat(tuple(torch.dstack([x1f, x2f]))).to(args.device)
    y = model(X).view(points, points, -1)
    y = y.cpu()

    for i in [0, 1]:
        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = crown_bounds.region.lower, crown_bounds.region.upper
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], points), torch.linspace(x1[1], x2[1], points))

        if subtract_lower:
            y_flatten = crown_bounds.lower[0][i, 0] * x1 + crown_bounds.lower[0][i, 1] * x2 + crown_bounds.lower[1][i]
        else:
            y_flatten = 0

        # Plot LBP linear bounds
        y_lower = crown_bounds.lower[0][i, 0] * x1 + crown_bounds.lower[0][i, 1] * x2 + crown_bounds.lower[1][i]
        y_upper = crown_bounds.upper[0][i, 0] * x1 + crown_bounds.upper[0][i, 1] * x2 + crown_bounds.upper[1][i]

        surf = ax.plot_surface(x1, x2, y_lower - y_flatten, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper - y_flatten, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        if mat_bounds is not None:
            y_lower = mat_bounds.lower[0][i, 0] * x1 + mat_bounds.lower[0][i, 1] * x2 + mat_bounds.lower[1][i]
            y_upper = mat_bounds.upper[0][i, 0] * x1 + mat_bounds.upper[0][i, 1] * x2 + mat_bounds.upper[1][i]

            surf = ax.plot_surface(x1, x2, y_lower - y_flatten, color='blue', label='MAT linear', alpha=0.4, shade=False)
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            surf = ax.plot_surface(x1, x2, y_upper - y_flatten, color='blue', alpha=0.4, shade=False)
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1f, x2f, y[..., i] - y_flatten, color='red', label='Function to bound', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.title(f'y{i + 1}')
        plt.legend()

        plt.show()


def load_mat_linear_bounds(path):
    mat = scipy.io.loadmat(path)
    M_h = torch.as_tensor(mat['M_h']).transpose(-1, -2)
    M_l = torch.as_tensor(mat['M_l']).transpose(-1, -2)
    B_h = torch.as_tensor(mat['B_h'])[:, :, 0]
    B_l = torch.as_tensor(mat['B_l'])[:, :, 0]
    lower, upper = torch.as_tensor(mat['partitions'][:, 0]), torch.as_tensor(mat['partitions'][:, 1])

    mat_input_set = HyperRectangle(lower, upper)
    mat_dynamics_bounds = LinearBounds(mat_input_set, (M_l, B_l), (M_h, B_h))
    check_gap(mat_dynamics_bounds)

    return mat_dynamics_bounds


def experiment_builder(args, config):
    if config['system'] == 'linear':
        return LinearExperiment(args, config)
    elif config['system'] == 'nndm':
        return NNDMExperiment(args, config)
    elif config['system'] == 'unicycle_nominal':
        return NominalUnicycleExperiment(args, config)
    elif config['system'] == 'unicycle_zerov':
        return ZeroVUnicycleExperiment(args, config)
    elif config['system'] == 'harrier':
        return HarrierExperiment(args, config)
    else:
        raise ValueError(f'System "{config["system"]}" not defined')


class Runner:
    """
    A class to construct the experiment and certifier, call the certifier to retrieve bounds, and save bounds to
    a .netcdf file.
    """

    def __init__(self, args, config, construct_experiment=experiment_builder):
        self.args = args
        self.config = config
        self.experiment = construct_experiment(self.args, self.config)

    @property
    def device(self):
        return self.args.device

    @property
    def safe_set(self):
        return self.experiment.dynamics.safe_set

    @property
    def dim(self):
        return self.experiment.dynamics.dim

    @property
    def noise(self):
        return self.experiment.dynamics.v

    @torch.no_grad()
    def bound_transition_prob(self):

        # Cons is a name from Lisp for order pairs, which allows shortened form unpacking. Nothing major.
        cons = self.experiment.dynamics, self.experiment.factory

        partition = self.experiment.grid_partition()
        number_hypercubes = len(partition)
        logger.debug(f'Number of hypercubes = {number_hypercubes}')

        # Create certifiers
        certifier = GaussianCertifier(*cons, partition, device=self.device, alpha=True)
        logger.info('Certifier created ... ')

        # Compute the probability bounds of transition from each hypercube to each other
        probability_bounds = certifier.regular_probability_bounds()
        logger.info('Regular probability bounds obtained ...')

        # Compute the probability bounds of transition from each hypercube to the safe set
        unsafe_probability_bounds = certifier.unsafe_probability_bounds()
        logger.info('Unsafe probability bounds obtained ...')

        region = ('region', list(range(1, len(partition) + 1)))
        to = ('to', list(range(1, len(partition) + 1)))
        _from = ('from', list(range(1, len(partition) + 1)))
        lu = ('dir', ['lower', 'upper'])
        x = ('x', list(range(1, self.dim + 1)))
        p = ('p', [1])

        safe_set = xr.DataArray(
            name='safe_set',
            data=torch.stack((self.safe_set[0], self.safe_set[1]), dim=0).numpy(),
            coords=[lu, x]
        )

        regions = xr.DataArray(
            name='regions',
            data=torch.stack((partition.lower, partition.upper), dim=1).numpy(),
            coords=[region, lu, x]
        )

        PA = xr.DataArray(
            name='transition_prob_A',
            data=torch.stack((probability_bounds.lower[0], probability_bounds.upper[0]), dim=2).numpy(),
            coords=[to, _from, lu, p, x]
        )

        Pb = xr.DataArray(
            name='transition_prob_b',
            data=torch.stack((probability_bounds.lower[1], probability_bounds.upper[1]), dim=2).numpy(),
            coords=[to, _from, lu, p]
        )

        PuA = xr.DataArray(
            name='transition_prob_unsafe_A',
            data=torch.stack((unsafe_probability_bounds.lower[0], unsafe_probability_bounds.upper[0]), dim=1).numpy(),
            coords=[_from, lu, p, x]
        )

        Pub = xr.DataArray(
            name='transition_prob_unsafe_b',
            data=torch.stack((unsafe_probability_bounds.lower[1], unsafe_probability_bounds.upper[1]), dim=1).numpy(),
            coords=[_from, lu, p]
        )

        ds = xr.Dataset(
            data_vars=dict(
                safe_set=safe_set,
                regions=regions,
                transition_prop_A=PA,
                transition_prob_b=Pb,
                transition_prop_unsafe_A=PuA,
                transition_prob_unsafe_b=Pub
            ),
            attrs=dict(
                num_regions=number_hypercubes,
                noise=self.noise[1].numpy()
            )
        )

        path = self.config['save_path']['transition_prob'].format(regions=number_hypercubes, noise=self.noise[1].tolist())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds.to_netcdf(path)

        logger.info("Probability data saved to file {}".format(path))

    @torch.no_grad()
    def bound_nominal_dynamics(self):
        partition = self.experiment.grid_partition()
        number_hypercubes = len(partition)
        logger.debug(f'Number of hypercubes = {number_hypercubes}')

        model = self.experiment.factory.build(MinimizePosteriorRect(self.experiment.dynamics)).to(self.device)
        logger.info('Bound propagation model created ... ')

        input_set = HyperRectangle(partition.lower, partition.upper)
        dynamics_bounds = model.crown(input_set, alpha=True)
        check_gap(dynamics_bounds)
        logger.info('Dynamics bounds obtained ...')

        # mat_dynamics_bounds = load_mat_linear_bounds("../../../PiecewiseConstant/synthesize/models/pendulum/partition_data_120.mat")
        #
        # for i in range(number_hypercubes):
        #     plot_partition(model, self.args, dynamics_bounds[i], mat_dynamics_bounds[i])

        region = ('region', list(range(1, len(partition) + 1)))
        lu = ('dir', ['lower', 'upper'])
        x = ('x', list(range(1, self.dim + 1)))
        y = ('y', list(range(1, self.dim + 1)))

        safe_set = xr.DataArray(
            name='safe_set',
            data=torch.stack((self.safe_set[0], self.safe_set[1]), dim=0).numpy(),
            coords=[lu, x]
        )

        regions = xr.DataArray(
            name='regions',
            data=torch.stack((partition.lower, partition.upper), dim=1).numpy(),
            coords=[region, lu, x]
        )

        nominal_dynamics_A = xr.DataArray(
            name='nominal_dynamics_A',
            data=torch.stack((dynamics_bounds.lower[0], dynamics_bounds.upper[0]), dim=1).numpy(),
            coords=[region, lu, y, x]
        )

        nominal_dynamics_b = xr.DataArray(
            name='nominal_dynamics_b',
            data=torch.stack((dynamics_bounds.lower[1], dynamics_bounds.upper[1]), dim=1).numpy(),
            coords=[region, lu, y]
        )

        ds = xr.Dataset(
            data_vars=dict(
                safe_set=safe_set,
                regions=regions,
                nominal_dynamics_A=nominal_dynamics_A,
                nominal_dynamics_b=nominal_dynamics_b
            ),
            attrs=dict(
                num_regions=number_hypercubes
            )
        )

        path = self.config['save_path']['nominal_dynamics'].format(regions=number_hypercubes)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds.to_netcdf(path)

        logger.info("Dynamics data saved to file {}".format(path))


def main(args):

    config = load_config(args.config_path)

    logger.info('Called runner ... ')
    runner = Runner(args, config)

    if args.task == 'bound_transition_prob':
        runner.bound_transition_prob()
    elif args.task == 'bound_nominal_dynamics':
        runner.bound_nominal_dynamics()
    else:
        raise ValueError('Task \'{task}\' does not exist'.format(task=args.task))


def parse_arguments():
    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--task', choices=['bound_nominal_dynamics', 'bound_transition_prob'], type=str, default='bound_nominal_dynamics')
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default=device_default, help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, default= 'unicycle/zeroV/unicycle.json', help='Path to configuration of experiment.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_file)

    torch.set_default_dtype(torch.float64)
    main(args)
