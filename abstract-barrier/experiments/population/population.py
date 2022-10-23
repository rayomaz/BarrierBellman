import logging

from torch import nn

from abstract_barrier.bounds import BarrierBoundModelFactory, GaussianProbability, MinimizeGap

from .dynamics import Population
from .partitioning import population_grid_partition
from .plot import plot_bounds_2d

logger = logging.getLogger(__name__)


class PopulationExperiment:

    def __init__(self, args, config):
        logger.info('Constructing model')

        self.factory = BarrierBoundModelFactory()
        self.dynamics = Population(config['dynamics']).to(args.device)

        self.config = config
        self.args = args

    def grid_partition(self):
        return population_grid_partition(self.config, self.dynamics)

    def plot(self):
        plot_bounds_2d(self.factory, self.dynamics, self.args, self.config)
