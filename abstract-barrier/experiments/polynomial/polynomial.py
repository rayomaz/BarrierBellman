import logging

from .dynamics import Polynomial, PolynomialUpdate, BoundPolynomialUpdate
from .partitioning import polynomial_grid_partition
from .plot import plot_bounds_2d

from abstract_barrier.discretization import ButcherTableau, BoundButcherTableau
from abstract_barrier.bounds import BarrierBoundModelFactory

logger = logging.getLogger(__name__)


class PolynomialExperiment:

    def __init__(self, args, config):
        logger.info('Constructing model')

        factory = BarrierBoundModelFactory()
        factory.register(PolynomialUpdate, BoundPolynomialUpdate)
        factory.register(ButcherTableau, BoundButcherTableau)

        self.factory = factory
        self.dynamics = Polynomial(config['dynamics']).to(args.device)

        self.config = config
        self.args = args

    def grid_partition(self):
        return polynomial_grid_partition(self.config, self.dynamics)

    def plot(self):
        plot_bounds_2d(self.dynamics, self.args, self.config)
