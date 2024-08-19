import logging

from bounds.bounds import BarrierBoundModelFactory

from .dynamics import Harrier
from experiments.utils import grid_partition

logger = logging.getLogger(__name__)


class HarrierExperiment:

    def __init__(self, args, config):
        logger.info('Constructing model')

        self.factory = BarrierBoundModelFactory()
        self.dynamics = [Harrier(config['dynamics']).to(args.device)]

        self.config = config
        self.args = args

    def grid_partition(self):
        return grid_partition(self.args, self.config)
