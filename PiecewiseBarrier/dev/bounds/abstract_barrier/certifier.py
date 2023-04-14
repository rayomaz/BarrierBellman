import logging

import torch
from torch import nn
from tqdm import tqdm

from bound_propagation import LinearBounds, HyperRectangle
from .bounds import GaussianProbability, MinimizeGap
from .dynamics import AdditiveGaussianDynamics

logger = logging.getLogger(__name__)


class GaussianCertifier(nn.Module):
    """
    - LP with linear bounds as constraints
    - Grid-based
    """
    def __init__(self, dynamics, factory, partition, type, horizon, lbp_method='crown-ibp', alpha=False, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        # Assumptions:
        # 1. Partitions containing the boundary of the safe / unsafe set belong to the unsafe subpartition
        #    to ensure correctness
        # 2. Regions are hyperrectangular and non-overlapping

        self.partition = partition.to(device)
        self.dynamics = dynamics.to(device)
        self.factory = factory

        self.horizon = horizon
        self.alpha = alpha
        self.lbp_method = lbp_method
        self.device = device
        self.type = type

    @torch.no_grad()
    def probability_bounds(self):
        return self.linear_bounds()

    @torch.no_grad()
    def modifified_erf_bound(self):
        return self.linear_bounds()

    def linear_bounds(self):
        in_batch_size = 1000
        in_set = self.partition.safe

        lower, upper = in_set.lower.split(in_batch_size), in_set.upper.split(in_batch_size)

        linear_bounds = []
        if self.type == "normal":
            for l, u in tqdm(list(zip(lower, upper)), desc='In'):
                linear_bounds.append(self.batch_linear_bounds(HyperRectangle(l, u)))

        elif self.type == "safe_set":
            _dimension = self.partition.safe.lower.size()
            _lower_bounds_state = self.partition.safe.lower[0]
            _upper_bounds_state = self.partition.safe.upper[-1]
            for idx in range(_dimension[0]):
                for l, u in tqdm(list(zip(lower, upper)), desc='In'):
                    lower_bound_current = l[idx]
                    upper_bound_current = u[idx]
                    l = torch.cat((lower_bound_current.unsqueeze(0), _lower_bounds_state.unsqueeze(0)), dim=0)
                    u = torch.cat((upper_bound_current.unsqueeze(0), _upper_bounds_state.unsqueeze(0)), dim=0)
                    linear_bounds.append(self.batch_linear_bounds(HyperRectangle(l, u)))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.lower[1] for bounds in linear_bounds], dim=1)),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.upper[1] for bounds in linear_bounds], dim=1)),
        )

    def batch_linear_bounds(self, in_set):
        out_batch_size = 5000
        out_set = self.partition.safe

        lower, upper = out_set.lower.split(out_batch_size), out_set.upper.split(out_batch_size)
        transition_probs = [self.analytic_transition_prob(l, u) for l, u in zip(lower, upper)]


        linear_bounds = []
        for transition_prob in tqdm(transition_probs, desc='Out'):
            linear_bounds.append(self.lbp(transition_prob, in_set))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds]), torch.cat([bounds.lower[1] for bounds in linear_bounds])),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds]), torch.cat([bounds.upper[1] for bounds in linear_bounds])),
        )

    def analytic_transition_prob(self, lower, upper):
        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)

        return self.factory.build(MinimizeGap(
            self.dynamics,
            GaussianProbability(loc, scale, lower, upper)
        )).to(self.device)

    def lbp(self, model, set):
        if self.lbp_method == 'crown':
            linear_bounds = model.crown(set, alpha=self.alpha)
        elif self.lbp_method == 'crown-ibp':
            linear_bounds = model.crown_ibp(set, alpha=self.alpha)
        elif self.lbp_method == 'ibp':
            linear_bounds = model.ibp(set)
        else:
            raise NotImplementedError(f'Supplied LBP method ({self.lbp_method}) does not exist')

        return linear_bounds.cpu()


