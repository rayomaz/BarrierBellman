import numpy as np
import torch
from bound_propagation import FixedLinear
from torch import nn, distributions

from abstract_barrier.dynamics import AdditiveGaussianDynamics


class Population(FixedLinear, AdditiveGaussianDynamics):
    @property
    def v(self):
        return (
            torch.tensor([0.0, 0.0]),
            torch.as_tensor(self.sigma)
        )

    # x[1] = juveniles
    # x[2] = adults

    def __init__(self, dynamics_config):
        super().__init__(torch.as_tensor([
            [0.0, dynamics_config['fertility_rate']],
            [dynamics_config['survival_juvenile'], dynamics_config['survival_adult']]
        ]))

        self.sigma = dynamics_config['sigma']

    def near_far_center(self, x, eps):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            near = torch.min(lower_x.abs(), upper_x.abs())
            far = torch.max(lower_x.abs(), upper_x.abs())
            center = (lower_x + upper_x) / 2
        else:
            near, far, center = x, x, x

        return near, far, center

    def initial(self, x, eps=None):
        near, far, center = self.near_far_center(x, eps)

        return (near.norm(dim=-1) <= 0.5) | (center.norm(dim=-1) <= 0.5)

    def sample_initial(self, num_particles):
        dist = distributions.Uniform(0, 1)
        r = 0.5 * dist.sample((num_particles,)).sqrt()
        theta = dist.sample((num_particles,)) * 2 * np.pi

        return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

    def safe(self, x, eps=None):
        near, far, center = self.near_far_center(x, eps)

        return far.norm(dim=-1) <= 2.0

    def unsafe(self, x, eps=None):
        near, far, center = self.near_far_center(x, eps)

        return far.norm(dim=-1) > 2.0

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -3.0) & (lower_x[..., 0] <= 3.0) & (upper_x[..., 1] >= -3.0) & (lower_x[..., 1] <= 3.0)

    @property
    def volume(self):
        return 6.0 ** 2
