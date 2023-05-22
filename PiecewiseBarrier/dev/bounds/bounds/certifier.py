import logging

import torch
from torch import nn
from tqdm import tqdm

from bound_propagation import LinearBounds, HyperRectangle, ElementWiseLinear
from .bounds import GaussianProbability, MinimizeGap
from .dynamics import AdditiveGaussianDynamics

logger = logging.getLogger(__name__)


class GaussianCertifier(nn.Module):
    def __init__(self, dynamics, factory, partition, lbp_method='crown', alpha=False, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        # Assumptions:
        # 1. Partitions containing the boundary of the safe / unsafe set in its interior must not be included
        #    (assume B_i(x) = 1) to ensure corrects. For now, this is accomplished by assuming X_s is a hyperrectangle,
        #    so that no regions contain the boundary, but rather is aligned with.
        # 2. Regions are hyperrectangular and non-overlapping

        self.partition = partition.to(device)
        self.dynamics = dynamics.to(device)
        self.factory = factory

        self.alpha = alpha
        self.lbp_method = lbp_method
        self.device = device

    @torch.no_grad()
    def unsafe_probability_bounds(self):
        # Since T(X_u | x) = 1 - T(X_s | x), if we add a -1 * z + 1 layer to the end of the model for the transition
        # kernel T(X_s | x), we retrieve a transition kernel for T(X_u | x).
        ext = ElementWiseLinear(-1, 1)

        transition_kernel = self.analytic_transition_prob(self.dynamics.safe_set[0], self.dynamics.safe_set[1], ext)
        linear_bounds = self.lbp(transition_kernel, self.partition)

        return linear_bounds.cpu()

    @torch.no_grad()
    def regular_probability_bounds(self):
        return self.linear_bounds()

    def linear_bounds(self):
        in_batch_size = 1000
        in_set = self.partition

        lower, upper = in_set.lower.split(in_batch_size), in_set.upper.split(in_batch_size)

        linear_bounds = []
        for l, u in tqdm(list(zip(lower, upper)), desc='In'):
            linear_bounds.append(self.batch_linear_bounds(HyperRectangle(l, u)))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.lower[1] for bounds in linear_bounds], dim=1)),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.upper[1] for bounds in linear_bounds], dim=1)),
        )

    def batch_linear_bounds(self, in_set):
        out_batch_size = 5000
        out_set = self.partition

        lower, upper = out_set.lower.split(out_batch_size), out_set.upper.split(out_batch_size)
        transition_probs = [self.analytic_transition_prob(l.unsqueeze(1), u.unsqueeze(1)) for l, u in zip(lower, upper)]

        linear_bounds = []
        for transition_prob in tqdm(transition_probs, desc='Out'):
            linear_bounds.append(self.lbp(transition_prob, in_set))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds]), torch.cat([bounds.lower[1] for bounds in linear_bounds])),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds]), torch.cat([bounds.upper[1] for bounds in linear_bounds])),
        )

    def analytic_transition_prob(self, lower, upper, ext=None):
        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)

        transition_kernel = nn.Sequential(
            self.dynamics,
            GaussianProbability(loc, scale, lower, upper)
        )

        if ext is not None:
            transition_kernel.append(ext)

        return self.factory.build(transition_kernel).to(self.device)

    def lbp(self, model, set):
        if self.lbp_method == 'crown':
            linear_bounds = model.crown(set, alpha=self.alpha)
        elif self.lbp_method == 'crown-ibp':
            linear_bounds = model.crown_ibp(set, alpha=self.alpha)
        else:
            raise NotImplementedError(f'Supplied LBP method ({self.lbp_method}) does not exist')

        return linear_bounds.cpu()
