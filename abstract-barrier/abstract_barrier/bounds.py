import math

import numpy as np
import torch
from bound_propagation import BoundModelFactory, BoundModule, LinearBounds, IntervalBounds, VectorMul, \
    Parallel, ElementWiseLinear, Clamp, BoundSequential, Select
from bound_propagation.activation import assert_bound_order, bisection
from bound_propagation.probability import BoundBellCurve
from torch import nn
from torch.nn import Identity

from abstract_barrier.discretization import BoundButcherTableau, ButcherTableau


class SumAll(nn.Module):
    def forward(self, x):
        return x.sum(dim=-1, keepdim=True)


class BoundSumAll(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.in_size = None

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        assert self.in_size is not None

        if linear_bounds.lower is None:
            lower = None
        else:
            size = [1 for _ in range(linear_bounds.lower[0].dim() - 1)]
            lower = (linear_bounds.lower[0].repeat(*size, self.in_size), linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            size = [1 for _ in range(linear_bounds.upper[0].dim() - 1)]
            upper = (linear_bounds.upper[0].repeat(*size, self.in_size), linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        return IntervalBounds(bounds.region,
                              bounds.lower.sum(dim=-1, keepdim=True),
                              bounds.upper.sum(dim=-1, keepdim=True))

    def propagate_size(self, in_size):
        self.in_size = in_size
        return 1


class ProdAll(nn.Sequential):
    def __init__(self, ndim):
        modules = []
        iterations = math.ceil(math.log2(ndim))

        if iterations == 0:
            super().__init__(Identity())
        else:
            for _ in range(iterations):
                if ndim % 2 == 0:
                    modules.append(VectorMul())
                else:
                    modules.append(Parallel(VectorMul(), nn.Identity(), split_size=[ndim - 1, 1]))

                ndim = ndim // 2 + ndim % 2

            super().__init__(
                *modules
            )


class Repeat(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.n = n

    def forward(self, x):
        return torch.cat([x for _ in range(self.n)], dim=-1)


class BoundRepeat(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = torch.stack(linear_bounds.lower[0].split(self.module.n, dim=-1), dim=-1).sum(dim=-1)
            lower = (lowerA, linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = torch.stack(linear_bounds.upper[0].split(self.module.n, dim=-1), dim=-1).sum(dim=-1)
            upper = (upperA, linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        return IntervalBounds(bounds.region,
                              torch.cat([bounds.lower for _ in range(self.module.n)], dim=-1),
                              torch.cat([bounds.upper for _ in range(self.module.n)], dim=-1))

    def propagate_size(self, in_size):
        return in_size * self.module.n


class GaussianProbability(nn.Sequential):
    def __init__(self, loc, scale, lower, upper):
        dim = lower.size(-1)

        super().__init__(
            ErfDiff(loc, scale, lower, upper),
            Clamp(min=0.0, max=1.0),
            ProdAll(dim),
            Clamp(min=0.0, max=1.0)
        )


class ErfDiff(nn.Module):
    def __init__(self, loc, scale, lower, upper):
        super().__init__()

        self.loc, self.scale = loc, scale
        self.lower, self.upper = lower.unsqueeze(1), upper.unsqueeze(1)

    def forward(self, x):
        return erf_diff(self.loc, self.scale, self.lower, self.upper, x)


def erf_diff(loc, scale, lower, upper, x, mask=None):
    def one_side(x, side, loc, scale):
        return 0.5 * torch.special.erf((side - loc - x) / (np.sqrt(2) * scale))

    lower, upper = lower, upper
    loc, scale = torch.as_tensor(loc), torch.as_tensor(scale)
    if mask is not None:
        lower, upper = lower.expand_as(mask)[mask], upper.expand_as(mask)[mask]
        loc, scale = loc.expand_as(mask)[mask], scale.expand_as(mask)[mask]

    return one_side(x, upper, loc, scale) - one_side(x, lower, loc, scale)


class BoundErfDiff(BoundBellCurve):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.midpoint = self.find_midpoint()
        self.lower_inflection, self.upper_inflection = kwargs.get('erf_diff_inflection', self.find_inflection_points())

    @torch.no_grad()
    def double_derivative(self, x):
        return erf_diff_double_derivative(self.module.loc, self.module.scale, self.module.lower, self.module.upper, x)

    def derivative(self, x, mask=None):
        return erf_diff_derivative(self.module.loc, self.module.scale, self.module.lower, self.module.upper, x, mask=mask)

    def func(self, x, mask=None):
        return erf_diff(self.module.loc, self.module.scale, self.module.lower, self.module.upper, x, mask=mask)

    @torch.no_grad()
    def find_inflection_points(self):
        return erf_diff_find_inflection_points(self.module.loc, self.module.scale, self.module.lower, self.module.upper, self.double_derivative)

    @torch.no_grad()
    def find_midpoint(self):
        return (self.module.lower + self.module.upper) / 2 - self.module.loc


@torch.no_grad()
def erf_diff_find_inflection_points(loc, scale, lower, upper, double_derivative):
    # Left inflection point
    def func(x):
        return -double_derivative(x)

    mid = (lower + upper) / 2 - loc

    bisection_lower = mid - 2 * scale
    bisection_upper = mid - scale
    lower_inflection = bisection(bisection_lower, bisection_upper, func, num_iter=100)

    # Right inflection point
    upper_inflection = 2 * mid - lower_inflection[1], 2 * mid - lower_inflection[0]

    return lower_inflection, upper_inflection


@torch.no_grad()
def erf_diff_double_derivative(loc, scale, lower, upper, x):
    def one_side(x, side):
        return (side - loc - x) * torch.exp(-torch.pow(side - loc - x, 2) / (2 * scale ** 2)) / (np.sqrt(2 * np.pi) * scale ** 3)

    return one_side(x, lower) - one_side(x, upper)


def erf_diff_derivative(loc, scale, lower, upper, x, mask=None):
    def one_side(x, side, loc, scale):
        return torch.exp(-torch.pow(side - loc - x, 2) / (2 * scale ** 2)) / (np.sqrt(2 * np.pi) * scale)

    lower, upper = lower, upper
    loc, scale = torch.as_tensor(loc), torch.as_tensor(scale)
    if mask is not None:
        lower, upper = lower.expand_as(mask)[mask], upper.expand_as(mask)[mask]
        loc, scale = loc.expand_as(mask)[mask], scale.expand_as(mask)[mask]

    return one_side(x, lower, loc, scale) - one_side(x, upper, loc, scale)


class MinimizeGap(nn.Sequential):
    pass


class BoundMinimizeGap(BoundSequential):
    def alpha_loss(self, linear_bounds, bound_lower, bound_upper):
        linear_bounds = LinearBounds(linear_bounds.region, None,
                                     (linear_bounds.upper[0] - linear_bounds.lower[0], linear_bounds.upper[1] - linear_bounds.lower[1]))
        interval_bounds = linear_bounds.concretize()

        return interval_bounds.upper.sum()


class BarrierBoundModelFactory(BoundModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register(SumAll, BoundSumAll)
        self.register(Repeat, BoundRepeat)
        self.register(ButcherTableau, BoundButcherTableau)
        self.register(ErfDiff, BoundErfDiff)
        self.register(MinimizeGap, BoundMinimizeGap)
