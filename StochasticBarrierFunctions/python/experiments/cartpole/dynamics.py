import torch
from torch import nn
import onnx2torch
from bound_propagation import FixedLinear, Parallel, Select, Cos, Sin, Cat, Mul, Div, Pow

from bounds.dynamics import AdditiveGaussianDynamics


class CartpoleController(FixedLinear):
    def __init__(self, F):
        # Computed gain using discrete time LQR
        F = torch.as_tensor(F).unsqueeze(0)

        # Negative feedback control
        super().__init__(-F)


class Cartpole(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        masscart = dynamics_config['masscart']
        masspole = dynamics_config['masspole']
        total_mass = masspole + masscart
        centerpole = dynamics_config['length'] / 2.0
        polemass_length = masspole * centerpole

        g = dynamics_config['g']
        Ts = dynamics_config['Ts']
        F = dynamics_config['F']

        # Choose ODE formulation z = (x, ẋ, θ, θ̇)
        # Discretize with Euler x(k + 1) = x(k) + Ts * f(x(k))

        super().__init__(
            # (x, dx, theta, dtheta)
            Cat(
                nn.Sequential(
                    Select([2]),  # theta
                    Parallel(
                        Sin(),
                        Cos()
                    )  # (sin theta, cos theta)
                ),
            ),  # (x, dx, theta, dtheta, sin theta, cos theta)
            Cat(
                nn.Sequential(
                    Parallel(
                        Select([4]),  # sin theta
                        nn.Sequential(
                            Select([0, 1, 2, 3]),
                            CartpoleController(F)  # u1
                        ),
                        nn.Sequential(
                            Select([3]),  # dtheta
                            Pow(2)
                        ),  # (dtheta^2)
                    ),
                    Cat(
                        Mul(
                            Select([0]),  # sin theta
                            Select([2])  # dtheta^2
                        )
                    ),
                    FixedLinear(
                        torch.as_tensor([[1 / total_mass, 0, 0, polemass_length / total_mass]])
                    )
                )
            ),  # (x, dx, theta, dtheta, sin theta, cos theta, temp)
            Cat(
                Div(
                    nn.Sequential(
                        Cat(
                            Mul(
                                Select([5]),  # cos theta
                                Select([6])  # temp
                            )
                        ),  # (x, dx, theta, dtheta, sin theta, cos theta, temp, cos theta * temp)
                        FixedLinear(
                            torch.as_tensor([[0, 0, 0, 0, g, 0, 0, -1]])
                        )
                    ),  # (g * sin theta - cos theta * temp)
                    nn.Sequential(
                        Cat(
                            nn.Sequential(
                                Select([5]),  # cos theta
                                Pow(2)
                            ),  # (cos^2 theta)
                        ),  # (x, dx, theta, dtheta, sin theta, cos theta, temp, cos^2 theta)
                        FixedLinear(
                            torch.as_tensor([[0, 0, 0, 0, 0, 0, 0, centerpole * masspole / total_mass]]),
                            torch.as_tensor([centerpole * 4 / 3])
                        )
                    )  # centerpole * (4.0 / 3.0 - masspole * costheta**2 / self.total_mass)
                )
            ),  # (x, dx, theta, dtheta, sin theta, cos theta, temp, thetaacc)
            Cat(
                nn.Sequential(
                    Cat(
                        Mul(
                            Select([5]),  # cos theta
                            Select([7])  # thetaacc
                        )
                    ),  # (x, dx, theta, dtheta, sin theta, cos theta, temp, thetaacc, cos theta * thetaacc)
                    FixedLinear(
                        torch.as_tensor([[0, 0, 0, 0, 0, 0, 1, 0, -polemass_length / total_mass]])
                    )
                )
            ),  # (x, dx, theta, dtheta, sin theta, cos theta, temp, thetaacc, xacc)
            FixedLinear(
                torch.as_tensor([
                    [1, Ts, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, Ts],
                    [0, 0, 1, Ts, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, Ts, 0]
                ])
            )
        )

        self.sigma = dynamics_config['sigma']
        self.safe = torch.as_tensor(dynamics_config['safe_set'][0]), torch.as_tensor(dynamics_config['safe_set'][1])
        self._dim = dynamics_config['dim']

    @property
    def v(self):
        sigma = torch.as_tensor(self.sigma)
        mu = torch.zeros_like(sigma)
        return mu, sigma

    @property
    def safe_set(self):
        return self.safe

    @property
    def dim(self):
        return self._dim
