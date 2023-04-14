import torch
from bound_propagation import ElementWiseLinear

from bounds.dynamics import AdditiveGaussianDynamics


class Linear(ElementWiseLinear, AdditiveGaussianDynamics):
    @property
    def v(self):
        return (
            torch.tensor([0.0]),
            torch.as_tensor(self.sigma)
        )

    def __init__(self, dynamics_config):
        super().__init__(torch.as_tensor([dynamics_config['rate']]))

        self.sigma = dynamics_config['sigma']

    def near_far(self, x, eps):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            near = torch.min(lower_x.abs(), upper_x.abs())
            far = torch.max(lower_x.abs(), upper_x.abs())
        else:
            near, far, center = x, x, x

        return near, far

    def initial(self, x, eps=None):
        near, far = self.near_far(x, eps)

        return near.norm(dim=-1) <= 0.2

    def safe(self, x, eps=None):
        near, far = self.near_far(x, eps)

        return far.norm(dim=-1) <= 1.0

    def unsafe(self, x, eps=None):
        near, far = self.near_far(x, eps)

        return far.norm(dim=-1) > 1.0

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -2.0) & (lower_x[..., 0] <= 2.0)

    @property
    def volume(self):
        return 6.0 ** 2
