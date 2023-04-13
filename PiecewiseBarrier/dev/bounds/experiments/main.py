import logging
from argparse import ArgumentParser

import torch
import sys

import numpy as np
import scipy.io

sys.path.append('../')

from abstract_barrier.certifier import GaussianCertifier
from linear.linear import LinearExperiment

from log import configure_logging
from utils import load_config

logger = logging.getLogger(__name__) 


class Runner:
    def __init__(self, args, config, type):
        self.args = args
        self.config = config
        self.experiment = LinearExperiment(self.args, self.config)
        self.type = type
            
    @property
    def horizon(self):
        return self.config['dynamics']['horizon']

    @property
    def device(self):
        return self.args.device

    def run(self):

        print(" Called on certifier ... ")
        cons = self.experiment.dynamics, self.experiment.factory
        partition = self.experiment.grid_partition()

        lower_partition, upper_partition = partition.safe.lower, partition.safe.upper 

        print('Number of hypercubes, lower bound = ', partition.safe.lower.size())
        print('Number of hypercubes, upper bound = ', partition.safe.upper.size())

        print(partition.safe.lower)
        print(partition.safe.upper)

        # Create certifiers
        certifier = GaussianCertifier(*cons, partition, type=self.type, horizon=self.horizon, device=self.device)
        print( " Certifier created ... ")

        # Compute the probability bounds of transition from each hypercube to the other 
        probability_bounds_safe = certifier.probability_bounds()
        print(" Probability bounds obtained ...")

        lower_probability_bounds = probability_bounds_safe.lower
        upper_probability_bounds = probability_bounds_safe.upper

        return lower_partition, upper_partition, lower_probability_bounds, upper_probability_bounds

def main(args, config):

    print(" Called runner ... ")
    type = "normal"
    runner = Runner(args, config, type)
    lower_partition, upper_partition, lower_probability_bounds, upper_probability_bounds = runner.run()
    print(" Regular probability bounds obtained ... ")

    type = "safe_set"
    runner = Runner(args, config, type)
    lower_partition_safe_set, upper_partition_safe_set, lower_probability_bounds_safe_set, upper_probability_bounds_safe_set = runner.run()
    print(" Safe set probability bounds obtained ... ")

    ''' # First element of the tuple represents the A matrix
        # Second element of the tuple represents the b vector
        + Together these make up the linear bounds '''

    # Convert torch to numpy arrays 
    lower_partition = lower_partition.numpy()
    upper_partition = upper_partition.numpy()

    lower_partition_safe_set = lower_partition_safe_set.numpy()
    upper_partition_safe_set = upper_partition_safe_set.numpy()

    # Separate data in A matrix and b vector (lower and upper)
    lower_probability_bounds_A_matrix = lower_probability_bounds[0]
    lower_probability_bounds_b_vector = lower_probability_bounds[1]

    lower_probability_bounds_A_matrix_safe_set = lower_probability_bounds_safe_set[0]
    lower_probability_bounds_b_vector_safe_set = lower_probability_bounds_safe_set[1]

    upper_probability_bounds_A_matrix = upper_probability_bounds[0]
    upper_probability_bounds_b_vector = upper_probability_bounds[1]

    upper_probability_bounds_A_matrix_safe_set = upper_probability_bounds_safe_set[0]
    upper_probability_bounds_b_vector_safe_set = upper_probability_bounds_safe_set[1]

    return lower_partition, upper_partition, lower_probability_bounds_A_matrix, upper_probability_bounds_A_matrix, \
            lower_probability_bounds_b_vector, upper_probability_bounds_b_vector,  \
            lower_partition_safe_set, upper_partition_safe_set, \
            lower_probability_bounds_A_matrix_safe_set, lower_probability_bounds_b_vector_safe_set, \
            upper_probability_bounds_A_matrix_safe_set, upper_probability_bounds_b_vector_safe_set
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda', help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')
    parser.add_argument('--task', type=str, choices=['certify'], default='certify')
    parser.add_argument('--space', type=str, choices=['equivalent_space', 'modified_space'], default='equivalent_space')
    # Here equivalent space means the state space ranges in all dimensions are similar

    return parser.parse_args()

if __name__ == '__main__':

    # Define parsing arguments
    args = parse_arguments()
    configure_logging(args.log_file)

    # Define configuration path
    CONFIG_PATH = "linear.json"
    config = load_config(CONFIG_PATH)

    # Set default torch float64
    torch.set_default_dtype(torch.float64)

    # Call main to obtain probability bounds
    lower_partition, upper_partition, lower_probability_bounds_A_matrix, upper_probability_bounds_A_matrix, \
    lower_probability_bounds_b_vector, upper_probability_bounds_b_vector, \
    lower_partition_safe_set, upper_partition_safe_set, \
    lower_probability_bounds_A_matrix_safe_set, lower_probability_bounds_b_vector_safe_set, \
    upper_probability_bounds_A_matrix_safe_set, upper_probability_bounds_b_vector_safe_set = main(args, config)

    state_space = np.array(config['partitioning']['state_space'])

    # Create array dictionary with needed data
    probability_array = {'state_space': state_space, 'lower_partition': lower_partition, 'upper_partition': upper_partition, 
                         'lower_probability_bounds_A': lower_probability_bounds_A_matrix.numpy(), 'upper_probability_bounds_A': upper_probability_bounds_A_matrix.numpy(),
                         'lower_probability_bounds_b': lower_probability_bounds_b_vector.numpy(), 'upper_probability_bounds_b': upper_probability_bounds_b_vector.numpy()}

    scipy.io.savemat('linearsystem_5.mat', probability_array)

    probability_array_safe = {'state_space': state_space, 'lower_partition': lower_partition_safe_set, 'upper_partition': upper_partition_safe_set, 
                         'lower_probability_bounds_A': lower_probability_bounds_A_matrix_safe_set.numpy(), 'upper_probability_bounds_A': upper_probability_bounds_A_matrix_safe_set.numpy(),
                         'lower_probability_bounds_b': lower_probability_bounds_b_vector_safe_set.numpy(), 'upper_probability_bounds_b': upper_probability_bounds_b_vector_safe_set.numpy()}

    scipy.io.savemat('linearsystem_safe_5.mat', probability_array_safe)

    print("Probability data saved to .mat file ... ")