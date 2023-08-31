import torch
from torch import nn
import onnx2torch
from bound_propagation import FixedLinear, Parallel, Select, Cos, Sin, Cat, Mul

from bounds.dynamics import AdditiveGaussianDynamics


class HarrierController(FixedLinear):
    def __init__(self):
        # Computed gain using discrete time LQR
        F = torch.as_tensor([
            [-8.891842293854333, -5.670193065350035e-11, 24.727163139540206, -7.281891826403686, -2.192161066390223e-12, 4.5596001105981685],
            [-4.185165149214767e-11, 30.99010210584173, 1.864243873114716e-10, -4.237308089280953e-11, 16.15334892779466, 1.326172799497476e-11]
        ])

        # Negative feedback control
        super().__init__(-F)


class Harrier(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        m = dynamics_config['m']
        g = dynamics_config['g']
        J = dynamics_config['J']
        r = dynamics_config['r']
        c = dynamics_config['c']
        Ts = dynamics_config['Ts']

        # Choose ODE formulation z = (x, y, θ, ẋ, ẏ, θ̇)
        # ẋ = z₄
        # ẏ = z₅
        # θ̇ = z₆
        # ẍ = (u₁ cos z₃ - u₂ sin z₃ - mg sin z₃ - cz₄) / m
        # ÿ = (u₁ sin z₃ + u₂ cos z₃ + mg (cos z₃ - 1) - cz₅) / m
        # θ̈ = r u₁ / J

        # Discretize with Euler x(k + 1) = x(k) + Ts * dx(k)

        super().__init__(
            Parallel(
                nn.Identity(),  # (x, y, theta, dx, dy, dtheta)
                nn.Sequential(
                    Select([2]),
                    Parallel(
                        Sin(),
                        Cos()
                    )
                ),  # (sin theta, cos theta)
                HarrierController()  # (u1, u2)
            ),
            Cat(
                Mul(
                    Select([8, 8, 9, 9]),
                    Select([6, 7, 6, 7])
                )  # (u1 * sin theta, u1 * cos theta, u2 * sin theta, u2 * cos theta)
            ),
            FixedLinear(
                torch.as_tensor([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ]) +
                Ts * torch.as_tensor([
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -c/m, 0, 0, -g, 0, 0, 0, 0, 1/m, -1/m, 0],
                    [0, 0, 0, 0, -c/m, 0, 0, g, 0, 0, 1/m, 0, 0, 1/m],
                    [0, 0, 0, 0, 0, 0, 0, 0, r/J, 0, 0, 0, 0, 0],
                ]),
                torch.as_tensor([0, 0, 0, 0, -g, 0])
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
