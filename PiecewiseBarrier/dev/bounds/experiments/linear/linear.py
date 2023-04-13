import logging

from torch import nn

from abstract_barrier.bounds import BarrierBoundModelFactory

from .dynamics import Linear
from .partitioning import linear_grid_partition

logger = logging.getLogger(__name__)


class LinearExperiment:

    def __init__(self, args, config):
        logger.info('Constructing model')

        self.factory = BarrierBoundModelFactory()
        self.dynamics = Linear(config['dynamics']).to(args.device)

        self.config = config
        self.args = args

    def grid_partition(self):
        return linear_grid_partition(self.args, self.config, self.dynamics)

