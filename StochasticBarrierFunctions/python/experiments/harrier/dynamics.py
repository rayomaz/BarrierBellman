import torch
from torch import nn
import onnx2torch
from bound_propagation import FixedLinear, Parallel, Select, Cos, Sin, Cat, Mul

from bounds.dynamics import AdditiveGaussianDynamics


class HarrierController(FixedLinear):
    def __init__(self):
        # Computed gain using discrete time LQR
        F = torch.as_tensor([
            [-0.0941953318523896, -1.6667981845027905e-13, 10.350863035117074, -0.43351164811730847, -2.009186606883043e-12, 2.214009158955452],
            [-4.494790399575651e-14, 0.09989432095302439, 7.549912965370178e-13, -3.3411843719527154e-13, 0.8454323004943662, 8.182689610523825e-14]
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
