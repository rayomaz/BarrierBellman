import logging
import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, distributions

from bound_propagation import BoundModule, IntervalBounds, LinearBounds, HyperRectangle, Select
from bound_propagation.activation import assert_bound_order, regimes

from abstract_barrier.discretization import Euler
from abstract_barrier.dynamics import AdditiveGaussianDynamics
from abstract_barrier.geometry import overlap_circle, overlap_rectangle, overlap_outside_circle, overlap_outside_rectangle


logger = logging.getLogger(__name__)


class PolynomialUpdate(nn.Module):
    def __init__(self):
        super().__init__()

    def x1_cubed(self, x):
        return (x[..., 0] ** 3) / 3.0

    def forward(self, x):
        x1 = x[..., 1]
        x2 = self.x1_cubed(x) - x.sum(dim=-1)

        x = torch.stack([x1, x2], dim=-1)
        return x


@torch.jit.script
def crown_backward_polynomial_jit(W_tilde: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda = torch.where(W_tilde[..., 1] < 0, alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-1))
    _delta = torch.where(W_tilde[..., 1] < 0, beta[0].unsqueeze(-1), beta[1].unsqueeze(-1))

    bias = W_tilde[..., 1] * _delta

    W_tilde1 = W_tilde[..., 1] * _lambda - W_tilde[..., 1]
    W_tilde2 = W_tilde[..., 0] - W_tilde[..., 1]

    W_tilde = torch.stack([W_tilde1, W_tilde2], dim=-1)

    return W_tilde, bias


class BoundPolynomialUpdate(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.input_bounds = None

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None

        self.bounded = False

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        self.input_bounds = None

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None

        self.bounded = False

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def func(self, x):
        return x**3 / 3.0

    def derivative(self, x):
        return x**2

    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower[..., 0], preactivation.upper[..., 0]
        zero_width, n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self.func(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self.func(upper[zero_width])

        lower_act, upper_act = self.func(lower), self.func(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y):
            alpha[mask] = a[mask]
            beta[mask] = y[mask] - a[mask] * x[mask]

        ###################
        # Negative regime #
        ###################
        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=n, a=d_prime, x=d, y=d_act)
        self.unstable_upper = n
        self.unstable_d_upper = d[n].detach().clone().requires_grad_()
        self.unstable_range_upper = lower[n], upper[n]

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=slope, x=lower, y=lower_act)

        ###################
        # Positive regime #
        ###################
        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=d_prime, x=d, y=d_act)
        self.unstable_lower = p
        self.unstable_d_lower = d[p].detach().clone().requires_grad_()
        self.unstable_range_lower = lower[p], upper[p]

        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=slope, x=upper, y=upper_act)

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct_upper = np & (slope >= lower_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_upper, a=slope, x=upper, y=upper_act)

        # Else use polynomial derivative to find upper bound on slope.
        implicit_upper = np & (slope < lower_prime)

        d = -upper / 2.0
        # Slope has to attach to (upper, upper^3)
        add_linear(self.alpha_upper, self.beta_upper, mask=implicit_upper, a=self.derivative(d), x=upper, y=upper_act)

        # Lower bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_lower = np & (slope >= upper_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_lower, a=slope, x=lower, y=lower_act)

        # Else use polynomial derivative to find upper bound on slope.
        implicit_lower = np & (slope < upper_prime)

        d = -lower / 2.0
        # Slope has to attach to (lower, lower^3)
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, a=self.derivative(d), x=lower, y=lower_act)

    @property
    def need_relaxation(self):
        return not self.bounded

    def set_relaxation(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        interval_bounds = IntervalBounds(
            linear_bounds.region,
            torch.max(interval_bounds.lower, self.input_bounds.lower),
            torch.min(interval_bounds.upper, self.input_bounds.upper)
        )

        self.alpha_beta(preactivation=interval_bounds)
        self.bounded = True

    def backward_relaxation(self, region):
        linear_bounds = self.initial_linear_bounds(region, 2)
        return linear_bounds, self

    def crown_backward(self, linear_bounds, optimize):
        assert self.bounded

        alpha_lower, alpha_upper = self.alpha_lower.detach().clone(), self.alpha_upper.detach().clone()
        beta_lower, beta_upper = self.beta_lower.detach().clone(), self.beta_upper.detach().clone()

        if optimize:
            alpha_lower, alpha_upper, beta_lower, beta_upper = \
                self.parameterize_alpha_beta(alpha_lower, alpha_upper, beta_lower, beta_upper)

        # NOTE: The order of alpha and beta are deliberately reversed - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha = alpha_upper, alpha_lower
            beta = beta_upper, beta_lower
            lower = crown_backward_polynomial_jit(linear_bounds.lower[0], alpha, beta)

            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = alpha_lower, alpha_upper
            beta = beta_lower, beta_upper
            upper = crown_backward_polynomial_jit(linear_bounds.upper[0], alpha, beta)
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward_x1_cubed(self, bounds):
        # x[..., 0] ** 3 is non-decreasing (and multiplying/dividing by a positive constant preserves this)
        return (bounds.lower[..., 0] ** 3) / 3.0, (bounds.upper[..., 0] ** 3) / 3.0

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        x1_lower = bounds.lower[..., 1]
        x1_upper = bounds.upper[..., 1]

        x1_cubed_lower, x1_cubed_upper = self.ibp_forward_x1_cubed(bounds)

        x2_lower = x1_cubed_lower - bounds.upper.sum(dim=-1)
        x2_upper = x1_cubed_upper - bounds.lower.sum(dim=-1)

        lower = torch.stack([x1_lower, x2_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 2

        return 2

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Polynomial bound not parameterized but expected to')

        def add_linear(alpha, beta, mask, x):
            a = self.derivative(x)
            y = self.func(x)

            alpha[mask] = a
            beta[mask] = y - a * x

        add_linear(alpha_lower, beta_lower, mask=self.unstable_lower, x=self.unstable_d_lower)
        add_linear(alpha_upper, beta_upper, mask=self.unstable_upper, x=self.unstable_d_upper)

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Polynomial bound not parameterized but expected to')

        yield self.unstable_d_lower
        yield self.unstable_d_upper

    def clip_params(self):
        self.unstable_d_lower.data.clamp_(min=self.unstable_range_lower[0], max=self.unstable_range_lower[1])
        self.unstable_d_upper.data.clamp_(min=self.unstable_range_upper[0], max=self.unstable_range_upper[1])


class Polynomial(Euler, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        super().__init__(
            PolynomialUpdate(), dynamics_config['dt']
        )

        self.dynamics_config = dynamics_config

    @property
    def v(self):
        return (
            torch.tensor([0.0, 0.0]),
            self.dynamics_config['dt'] * torch.as_tensor(self.dynamics_config['sigma'])
        )

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([1.5, 0.0], device=x.device), math.sqrt(0.25))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.8, -0.1], device=x.device), torch.tensor([-1.2, 0.1], device=x.device))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.4, -0.5], device=x.device), torch.tensor([-1.2, 0.1], device=x.device))

        return cond1 | cond2 | cond3

    def sample_initial(self, num_particles):
        return self.sample_initial_unsafe(
            num_particles,
            (torch.tensor([-1.8, -0.1]), torch.tensor([-1.4, 0.1])),
            (torch.tensor([-1.4, -0.5]), torch.tensor([-1.2, 0.1])),
            (torch.tensor([1.5, 0.0]), math.sqrt(0.25))
        )

    def sample_initial_unsafe(self, num_particles, rect1, rect2, circle):
        rect1_area = (rect1[1] - rect1[0]).prod()
        rect2_area = (rect2[1] - rect2[0]).prod()
        circle_area = circle[1] ** 2 * np.pi
        total_area = rect1_area + rect2_area + circle_area

        rect1_prob = rect1_area / total_area
        rect2_prob = rect2_area / total_area
        circle_prob = circle_area / total_area

        dist = distributions.Multinomial(total_count=num_particles, probs=torch.tensor([rect1_prob, rect2_prob, circle_prob]))
        count = dist.sample().int()

        dist = distributions.Uniform(rect1[0], rect1[1])
        rect1_samples = dist.sample((count[0],))

        dist = distributions.Uniform(rect2[0], rect2[1])
        rect2_samples = dist.sample((count[1],))

        dist = distributions.Uniform(0, 1)
        r = circle[1] * dist.sample((count[2],)).sqrt()
        theta = dist.sample((count[2],)) * 2 * np.pi
        circle_samples = circle[0] + torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

        return torch.cat([rect1_samples, rect2_samples, circle_samples], dim=0)

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0], device=x.device), math.sqrt(0.16))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.6, 0.5], device=x.device))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.8, 0.3], device=x.device))

        return cond1 | cond2 | cond3

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -3.5) & (lower_x[..., 0] <= 2.0) & \
               (upper_x[..., 1] >= -2.0) & (lower_x[..., 1] <= 1.0)

    @property
    def volume(self):
        return (2.0 - (-3.5)) * (1.0 - (-2.0))
