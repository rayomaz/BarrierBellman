import torch
from torch import nn
import onnx2torch
from bound_propagation import FixedLinear, Parallel, Select, Cos, Sin, Cat, Mul, Div

from bounds.dynamics import AdditiveGaussianDynamics

class NominalUnicycle(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        
        # Load control parameters
        Ts = dynamics_config['Ts']
        kp1 = dynamics_config['kp1']
        kd1 = dynamics_config['kd1']
        kp2 = dynamics_config['kp2']
        kd2 = dynamics_config['kd2']

        # Choose ODE formulation z = (x, y, θ, v)
        # ẋ = z₄ cos z₃
        # ẏ = z₄ sin z₃
        # θ̇ = - [(sin z₃) / z₄ ] * u₁ + [(cos z₃) / z₄] * u₂
        # v̇ = cos z₃ * u₁ + sin z₃ * u₂
        
        # Nominal Control Law for v > 5e-2 
        # u₁ = - kp1 * x - kd1 * v * cos theta
        # u₂ = - kp2 * y - kd2 * v * sin theta
        
        # Discretize with Euler x(k + 1) = x(k) + Ts * dx(k)

        super().__init__(
            Parallel(
                nn.Identity(),  # (x, y, theta, v) #(0, 1, 2, 3)
                nn.Sequential(
                    Select([2]),  # theta
                    Parallel(
                        Sin(),
                        Cos()
                    )  # (sin theta, cos theta) #(4, 5)
                ),
            ),
            Cat(
                Mul(
                    Select([3, 3]),
                    Select([4, 5])
                )  # (v * sin theta, v * cos theta) #(6, 7)
            ),
            Cat(
                Div(
                    Select([4, 5]),
                    Select([3, 3])
                )  # (sin theta / v, cos theta / v) #(8, 9)
            ),
            Cat(
                Mul(
                    Select([0, 1]),
                    Select([5, 4])
                )  # (x * cos theta, y * sin theta) #(10, 11)
            ),
            Cat(
                Mul(
                    Select([0, 1]),
                    Select([8, 9])
                )  # (x * sin theta / v, y * cos theta / v) #(12, 13)
            ),
            Cat(
                Mul(
                    Select([7, 6]),
                    Select([8, 9])
                )  # (v cos theta * sin theta / v, v sin theta * cos theta / v) #(14, 15)
            ),
            Cat(
                Mul(
                    Select([7, 6]),
                    Select([5, 4])
                )  # (v cos theta * cos theta, v sin theta * sin theta) #(16, 17)
            ),
            FixedLinear(
                torch.as_tensor([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]) +
                Ts * torch.as_tensor([
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kp1, -kp2, kd1, -kd2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kp1, -kp2, 0, 0, 0, 0, -kd1, -kd2]
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


class ZeroVelocityUnicycle(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        
        # Load control parameters
        Ts = dynamics_config['Ts']

        # Choose ODE formulation z = (x, y, θ, v)
        # ẋ = z₄ cos z₃
        # ẏ = z₄ sin z₃
        # θ̇ = u₁
        # v̇ = u₂
        
        # Nominal Control Law for v < 5e-2 
        # u₁ = - θ / dt
        # u₂ = - v / dt
        
        # Discretize with Euler x(k + 1) = x(k) + Ts * dx(k)

        super().__init__(
            Parallel(
                nn.Identity(),  # (x, y, theta, v) #(0, 1, 2, 3)
                nn.Sequential(
                    Select([2]),  # theta
                    Parallel(
                        Sin(),
                        Cos()
                    )  # (sin theta, cos theta) #(4, 5)
                ),
            ),
            Cat(
                Mul(
                    Select([3, 3]),
                    Select([4, 5])
                )  # (v * sin theta, v * cos theta) #(6, 7)
            ),
            FixedLinear(
                torch.as_tensor([
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0]
                ]) +
                Ts * torch.as_tensor([
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, -1/Ts, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1/Ts, 0, 0, 0, 0]
                ])
            )
        )

        self.sigma = dynamics_config['sigma']
        self.safe = torch.as_tensor(dynamics_config['safe_set_2'][0]), torch.as_tensor(dynamics_config['safe_set_2'][1])
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