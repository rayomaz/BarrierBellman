import logging
from argparse import ArgumentParser

import h5py as h5py
import torch

from bounds.certifier import GaussianCertifier
from linear.linear import LinearExperiment

from log import configure_logging
from utils import load_config

logger = logging.getLogger(__name__)


def experiment_builder(args, config):
    if config['system'] == 'linear':
        return LinearExperiment(args, config)
    else:
        raise ValueError(f'System "{config["system"]}" not defined')


class Runner:
    """
    A class to construct the experiment and certifier, call the certifier to retrieve bounds, and save bounds to
    a .hdf5 file.
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

    def run(self):

        # Cons is a name from Lisp for order pairs, which allows shortened form unpacking. Nothing major.
        cons = self.experiment.dynamics, self.experiment.factory

        partition = self.experiment.grid_partition()
        logger.debug(f'Number of hypercubes = {len(partition)}')

        # Create certifiers
        certifier = GaussianCertifier(*cons, partition, device=self.device)
        logger.info('Certifier created ... ')

        # Compute the probability bounds of transition from each hypercube to each other
        probability_bounds = certifier.regular_probability_bounds()
        logger.info('Regular probability bounds obtained ...')

        # Compute the probability bounds of transition from each hypercube to the safe set
        unsafe_probability_bounds = certifier.unsafe_probability_bounds()
        logger.info('Unsafe probability bounds obtained ...')

        number_hypercubes = self.config["partitioning"]['num_slices'][0]

        with h5py.File('../../../tests/partitions/linearsystem_' + str(number_hypercubes) + '.hdf5', 'w') as f:
            safe_set = f.create_group('safe_set')
            safe_set.create_dataset('lower', data=self.safe_set[0])
            safe_set.create_dataset('upper', data=self.safe_set[1])

            partitioning = f.create_group('partitioning')
            partitioning.create_dataset('lower', data=partition.lower.numpy(), compression='gzip')
            partitioning.create_dataset('upper', data=partition.upper.numpy(), compression='gzip')

            # Separate data in A matrix and b vector (lower and upper)
            # A has shape [i, j, p, x]   (transition from j to i)
            # b has shape [i, j, p]
            prob_bounds = f.create_group('prob_bounds')

            lower_prob_bounds = prob_bounds.create_group('lower')
            lower_prob_bounds.create_dataset('A', data=probability_bounds.lower[0].numpy(), compression='gzip')
            lower_prob_bounds['A'].dims[0].label = 'i'
            lower_prob_bounds['A'].dims[1].label = 'j'
            lower_prob_bounds['A'].dims[2].label = 'p'
            lower_prob_bounds['A'].dims[3].label = 'x'

            lower_prob_bounds.create_dataset('b', data=probability_bounds.lower[1].numpy(), compression='gzip')
            lower_prob_bounds['b'].dims[0].label = 'i'
            lower_prob_bounds['b'].dims[1].label = 'j'
            lower_prob_bounds['b'].dims[2].label = 'p'

            upper_prob_bounds = prob_bounds.create_group('upper')

            upper_prob_bounds.create_dataset('A', data=probability_bounds.upper[0].numpy(), compression='gzip')
            upper_prob_bounds['A'].dims[0].label = 'i'
            upper_prob_bounds['A'].dims[1].label = 'j'
            upper_prob_bounds['A'].dims[2].label = 'p'
            upper_prob_bounds['A'].dims[3].label = 'x'

            upper_prob_bounds.create_dataset('b', data=probability_bounds.upper[1].numpy(), compression='gzip')
            upper_prob_bounds['b'].dims[0].label = 'i'
            upper_prob_bounds['b'].dims[1].label = 'j'
            upper_prob_bounds['b'].dims[2].label = 'p'

            # Separate data in A matrix and b vector (lower and upper)
            # A has shape [j, p, x]   (transition from j to u)
            # b has shape [j, p]
            unsafe_prob_bounds = f.create_group('unsafe_prob_bounds')

            unsafe_lower_prob_bounds = unsafe_prob_bounds.create_group('lower')

            unsafe_lower_prob_bounds.create_dataset('A', data=unsafe_probability_bounds.lower[0].numpy(), compression='gzip')
            unsafe_lower_prob_bounds['A'].dims[0].label = 'j'
            unsafe_lower_prob_bounds['A'].dims[1].label = 'p'
            unsafe_lower_prob_bounds['A'].dims[2].label = 'x'

            unsafe_lower_prob_bounds.create_dataset('b', data=unsafe_probability_bounds.lower[1].numpy(), compression='gzip')
            unsafe_lower_prob_bounds['b'].dims[0].label = 'j'
            unsafe_lower_prob_bounds['b'].dims[1].label = 'p'

            unsafe_upper_prob_bounds = unsafe_prob_bounds.create_group('upper')

            unsafe_upper_prob_bounds.create_dataset('A', data=unsafe_probability_bounds.upper[0].numpy(), compression='gzip')
            unsafe_upper_prob_bounds['A'].dims[0].label = 'j'
            unsafe_upper_prob_bounds['A'].dims[1].label = 'p'
            unsafe_upper_prob_bounds['A'].dims[2].label = 'x'

            unsafe_upper_prob_bounds.create_dataset('b', data=unsafe_probability_bounds.upper[1].numpy(), compression='gzip')
            unsafe_upper_prob_bounds['b'].dims[0].label = 'j'
            unsafe_upper_prob_bounds['b'].dims[1].label = 'p'

        logger.info("Probability data saved to file ... ")


def main(args):

    config = load_config(args.config_path)

    logger.info('Called runner ... ')
    runner = Runner(args, config)
    runner.run()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cpu', help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_file)

    torch.set_default_dtype(torch.float64)
    main(args)
