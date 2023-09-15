import torch
from torch import nn
import onnx2torch
from bound_propagation import FixedLinear, Parallel, Select, Cos, Sin, Cat, Mul, Div

from bounds.dynamics import AdditiveGaussianDynamics


class UnicycleController(FixedLinear):
    def __init__(self):
        # Gain using optimal LQR controller
        F = torch.as_tensor([
            [0.9907, 1.8584, 0.0, 0.0],
            [0.0, 0.0, 0.9907, 1.8584]
        ])

        # Negative feedback control
        super().__init__(-F)


class Unicycle(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        
        Ts = dynamics_config['Ts']

        # Choose ODE formulation z = (x, y, θ, v)
        # ẋ = z₄ cos z₃
        # ẏ = z₄ sin z₃
        # θ̇ = - [(sin z₃) / z₄ ] * u₁ + [(cos z₃) / z₄] * u₂
        # v̇ = cos z₃ * u₁ + sin z₃ * u₂

        # Discretize with Euler x(k + 1) = x(k) + Ts * dx(k)

        super().__init__(
            Parallel(
                nn.Identity(),  # (x, y, theta, v)
                nn.Sequential(
                    Select([2]),  # theta
                    Parallel(
                        Sin(),
                        Cos()
                    )  # (sin theta, cos theta)
                ),
                UnicycleController()  # (u1, u2)
            ),
            Cat(
                Mul(
                    Select([3, 3, 6, 6, 7, 7]),
                    Select([4, 5, 4, 5, 4, 5])
                )  # (v * sin theta, v * cos theta, u1 * sin theta, u1 * cos theta, u2 * sin theta, u2 * cos theta)
            ),
            Cat(
                Div(
                    Select([10, 13]),
                    Select([3, 3])
                )  # (u1 / v * sin theta, u2 / v * cos theta)
            ),
            FixedLinear(
                torch.as_tensor([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]) +
                Ts * torch.as_tensor([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
                ])
            )
        )

        self.sigma = dynamics_config['sigma']
        self.safe = torch.as_tensor(dynamics_config['safe_set'][0]), torch.as_tensor(dynamics_config['safe_set'][1])
        self._dim = dynamics_config['dim']

    @property
    def v(self):
        return torch.tensor([0.0]), torch.as_tensor(self.sigma)

    @property
    def safe_set(self):
        return self.safe

    @property
    def dim(self):
        return self._dim