''' Math utilities '''

import sys
import math
import copy
import numpy as np

from typing import Tuple

sys.path.append('src')

class utilities():

    ''' Hypercube generation of discrete spaces given input dimensions
    Outputs:
        1. hypercubes:  partitioned state space
        2. states: center points for input to optimization
    '''

    def __init__(self, **kwargs):
        super().__init__()
        self.system_dimension = kwargs['system_dimension']
        self.state_space = kwargs['state_space']
        self.epsilon = kwargs['epsilon']

    def create_hypercube_single_eps(self) -> Tuple[list, list, list]:
        """
        A helper method to create hypercube intervals in each dimension using the same eps value
        """

        # Define self variables
        state_space = self.state_space
        eps = self.epsilon
        n = self.system_dimension

        # Initialize
        hypercubes_partitions = []
        hypercubes_partitions_centers = []
        partition_dimension = []

        for ii in range(n):
            ith_space = state_space[ii]
            range_per_interval = 2*eps
            length = max(ith_space) - min(ith_space)

            # For partition of the space the length should be at least 4 epsilon
            if length < 4*eps:
                print('Error: state space', ii, 'too small for partitioning over chosen epsilon:', eps)
                print('Try decreasing epsilon [0-1] or normalizing the state space')

            number_of_intervals = math.floor(length/range_per_interval)

            x_ith_low = min(ith_space)
            r = range_per_interval
            m = int(number_of_intervals)
            partition_dimension.append(m)

            jth_hyper = []
            jth_hyper_center = []
            # Assumption: partition floating point errors for domain R negligible
            for jj in range(int(number_of_intervals)):
                x_ith_partitions = [x_ith_low + jj*r, x_ith_low + (jj+1)*r]
                jth_hyper.append(x_ith_partitions)
                jth_hyper_center.append(min(jth_hyper[jj]) + eps)

            hypercubes_partitions.append(jth_hyper)
            hypercubes_partitions_centers.append(jth_hyper_center)

        return hypercubes_partitions, hypercubes_partitions_centers, partition_dimension


    def create_hypercube_variable_eps(self) -> Tuple[list, list, list]:
        """
            A helper method to create hypercube intervals in each dimension using the variable eps value
        """

        # Define self variables
        state_space = self.state_space
        eps = self.eps
        n = self.system_dimension

        # Initialize
        if eps.shape[0] != state_space.shape[0]:
            print("Please ensure that the epsilon is specified for each dimension")
        hypercubes_partitions = []
        hypercubes_partitions_centers = []
        partition_dimension = []

        for ii in range(n):
            ith_space = state_space[ii]
            range_per_interval = 2 * eps[ii]
            length = max(ith_space) - min(ith_space)

            # For partition of the space the length should be at least 4 epsilon
            if length < 4 * eps[ii]:
                print('Error: state space', ii, 'too small for partitioning over chosen epsilon:', eps)
                print('Try decreasing epsilon [0-1] or normalizing the state space')

            number_of_intervals = math.floor(length/range_per_interval)

            x_ith_low = min(ith_space)
            r = range_per_interval
            m = int(number_of_intervals)
            partition_dimension.append(m)

            jth_hyper = []
            jth_hyper_center = []
            # Assumption: partition floating point errors for domain R negligible
            for jj in range(int(number_of_intervals)):
                x_ith_partitions = [x_ith_low + jj * r, x_ith_low + (jj + 1) * r]
                jth_hyper.append(x_ith_partitions)
                jth_hyper_center.append(min(jth_hyper[jj]) + eps[ii])

            hypercubes_partitions.append(jth_hyper)
            hypercubes_partitions_centers.append(jth_hyper_center)

        return hypercubes_partitions, hypercubes_partitions_centers, partition_dimension

    def recursive_for(hypercube_partitions, dim_count, partition_count, dim, element, hypermatrix):

        for zz in range(len(hypercube_partitions[partition_count])):
            element1 = copy.deepcopy(element)
            element1.append(hypercube_partitions[partition_count][zz])

            if partition_count < dim:
                partition_count += 1
                utilities.recursive_for(hypercube_partitions, dim_count, partition_count, dim, element1, hypermatrix)
                partition_count -= 1
            else:
                element2 = copy.deepcopy(element1)
                hypermatrix.append(element2)
                element1.remove(element1[-1])

        return hypermatrix

    def hypercubes(self):

        # Define self variables
        eps = self.epsilon

        if isinstance(eps, float) or isinstance(eps, int):
            print("Using the same epsilon for each dimension")
            hypercubes_partitions, hypercubes_partitions_centers, partition_dimension = \
                utilities.create_hypercube_single_eps(self)
        elif isinstance(eps, np.ndarray):
            print("Using variable epsilon")
            hypercubes_partitions, hypercubes_partitions_centers, partition_dimension = \
                utilities.create_hypercube_variable_eps(self)
        else:
            print("Please enter a valid type of EPS, either an array or a scalar value")
            sys.exit(-1)

        # Generate hyper matrix containing all combinations in n-dim space [recursive for loops]
        element = []
        hypermatrix = []
        dim = len(partition_dimension) - 1
        dim_count = 2
        partition_count = 0

        hypercubes = utilities.recursive_for(hypercubes_partitions, dim_count, partition_count, dim, element, hypermatrix)
        hypercubes = np.array(hypercubes)

        # Generate hyper matrix containing all center points in s_dim-dim space [recursive for loops]
        element = []
        hypermatrix = []
        dim = len(partition_dimension) - 1
        dim_count = 2
        partition_count = 0

        states = utilities.recursive_for(hypercubes_partitions_centers, dim_count, partition_count, dim, element, hypermatrix)
        # partitions = generate_grid(hypercubes_partitions_centers)  # Generate hypercubes
        states = np.array(states)

        return hypercubes, states