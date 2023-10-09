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
from unicycle.unicycle import UnicycleExperiment
from harrier.harrier import HarrierExperiment

from log import configure_logging
from utils import load_config

logger = logging.getLogger(__name__)


def experiment_builder(args, config):
    if config['system'] == 'linear':
        return LinearExperiment(args, config)
    elif config['system'] == 'nndm':
        return NNDMExperiment(args, config)
    elif config['system'] == 'unicycle':
        return UnicycleExperiment(args, config)
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

    runner.bound_nominal_dynamics()


def parse_arguments():
    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--task', choices=['bound_nominal_dynamics', 'bound_transition_prob'], type=str, default='bound_nominal_dynamics')
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default=device_default, help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_file)

    torch.set_default_dtype(torch.float64)
    main(args)