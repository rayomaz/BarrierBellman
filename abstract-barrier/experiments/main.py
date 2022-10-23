import logging
from argparse import ArgumentParser

import torch

from abstract_barrier.monte_carlo import monte_carlo_simulation
from abstract_barrier.certifier import AdditiveGaussianCertifierV1, AdditiveGaussianCertifierV2, \
    AdditiveGaussianIMDPCertifier, AdditiveGaussianCertifierV3

from population.population import PopulationExperiment
from polynomial.polynomial import PolynomialExperiment

from log import configure_logging
from utils import load_config

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.experiment = self.create_experiment()

    def create_experiment(self):
        if self.config['system'] == 'population':
            return PopulationExperiment(self.args, self.config)
        elif self.config['system'] == 'polynomial':
            return PolynomialExperiment(self.args, self.config)
        else:
            raise ValueError(f'System {self.config["system"]} not defined')

    def create_certifier(self):
        cons = self.experiment.dynamics, self.experiment.factory
        if self.config['certify']['method'] == 'lp-v1':
            partition = self.experiment.grid_partition()
            return AdditiveGaussianCertifierV1(*cons, partition, horizon=self.horizon, device=self.device)
        elif self.config['certify']['method'] == 'lp-v2':
            partition = self.experiment.grid_partition()
            return AdditiveGaussianCertifierV2(*cons, partition, horizon=self.horizon, device=self.device)
        elif self.config['certify']['method'] == 'lp-v3':
            partition = self.experiment.grid_partition()
            return AdditiveGaussianCertifierV3(*cons, partition, horizon=self.horizon, device=self.device)
        elif self.config['certify']['method'] == 'imdp':
            partition = self.experiment.grid_partition()
            return AdditiveGaussianIMDPCertifier(*cons, partition, horizon=self.horizon, device=self.device)
        else:
            raise ValueError('Invalid certifier configuration')

    @property
    def horizon(self):
        return self.config['dynamics']['horizon']

    @property
    def device(self):
        return self.args.device

    def run(self):
        certifier = self.create_certifier()

        if args.task == 'certify':
            self.certify(certifier, self.config['certify'])
        elif args.task == 'plot':
            self.plot()
        elif args.task == 'simulate':
            self.monte_carlo()
        else:
            raise ValueError('Invalid task')

    def monte_carlo(self):
        monte_carlo_simulation(self.args, self.config, self.experiment.dynamics)

    def plot(self):
        self.experiment.plot()

    @torch.no_grad()
    def certify(self, certifier, test_config, alpha=False):
        certifier.alpha = alpha

        unsafety_prob = certifier.certify()

        msg = f'Safety certificate: {1.0 - unsafety_prob:>7f}'
        logger.info(msg)


def main(args):
    torch.set_default_dtype(torch.float64)
    config = load_config(args.config_path)

    runner = Runner(args, config)
    runner.run()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda', help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')
    parser.add_argument('--task', type=str, choices=['certify', 'plot', 'simulate'], default='certify', help='Certify will run our method to certify a system according to the config path. Test and plot with load the barrier and do their respective operations.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_file)

    torch.set_default_dtype(torch.float64)
    main(args)
