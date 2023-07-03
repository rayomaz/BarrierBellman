import torch
from bound_propagation import ElementWiseLinear

from bounds.dynamics import AdditiveGaussianDynamics


class Linear(ElementWiseLinear, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        super().__init__(torch.as_tensor([dynamics_config['rate']]))

        self.sigma = dynamics_config['sigma']
        self.safe = torch.as_tensor(dynamics_config['safe_set'][0]), torch.as_tensor(dynamics_config['safe_set'][1])

    @property
    def v(self):
        return (
            torch.tensor([0.0]),
            torch.as_tensor(self.sigma)
        )

    @property
    def safe_set(self):
        return self.safe
