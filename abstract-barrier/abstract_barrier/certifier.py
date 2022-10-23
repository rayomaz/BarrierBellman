import logging

import cvxpy as cp
import torch
from bound_propagation import LinearBounds, HyperRectangle
from torch import nn
from tqdm import tqdm, trange

from .bounds import GaussianProbability, MinimizeGap, erf_diff_double_derivative, erf_diff_find_inflection_points
from .dynamics import AdditiveGaussianDynamics
from .geometry import overlap_rectangle

logger = logging.getLogger(__name__)


class AdditiveGaussianCertifierV1(nn.Module):
    """
    - Worst case over the summation of linear bounds
    - Grid-based
    """
    def __init__(self, dynamics, factory, partition, horizon, lbp_method='crown', alpha=False, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        # Assumptions:
        # 1. Partitions containing the boundary of the safe / unsafe set belong to the unsafe subpartition
        #    to ensure correctness
        # 2. Regions are hyperrectangular and non-overlapping
        self.partition = partition.to(device)
        self.dynamics = dynamics.to(device)

        factory.kwargs['alpha_iterations'] = 40
        self.factory = factory

        self.horizon = horizon
        self.alpha = alpha
        self.lbp_method = lbp_method
        self.device = device

    def analytic_transition_prob(self, lower, upper):
        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)

        return self.factory.build(MinimizeGap(
            self.dynamics,
            GaussianProbability(loc, scale, lower, upper)
        )).to(self.device)

    @torch.no_grad()
    def certify(self):
        linear_bounds = self.linear_bounds()
        prob_gap = self.prob_gap(linear_bounds)

        gamma = self.time_propagate(linear_bounds)
        unsafe_prob = self.max_initial_set(gamma)

        return unsafe_prob.item()

    def linear_bounds(self):
        in_batch_size = 1000
        in_set = self.partition.safe

        lower, upper = in_set.lower.split(in_batch_size), in_set.upper.split(in_batch_size)
        linear_bounds = []
        for l, u in tqdm(list(zip(lower, upper)), desc='In'):
            linear_bounds.append(self.batch_linear_bounds(HyperRectangle(l, u)))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.lower[1] for bounds in linear_bounds], dim=1)),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.upper[1] for bounds in linear_bounds], dim=1)),
        )

    def batch_linear_bounds(self, in_set):
        out_batch_size = 5000
        out_set = self.partition.safe

        lower, upper = out_set.lower.split(out_batch_size), out_set.upper.split(out_batch_size)
        transition_probs = [self.analytic_transition_prob(l, u) for l, u in zip(lower, upper)]

        linear_bounds = []
        for transition_prob in tqdm(transition_probs, desc='Out'):
            linear_bounds.append(self.lbp(transition_prob, in_set))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds]), torch.cat([bounds.lower[1] for bounds in linear_bounds])),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds]), torch.cat([bounds.upper[1] for bounds in linear_bounds])),
        )

    def lbp(self, model, set):
        if self.lbp_method == 'crown':
            linear_bounds = model.crown(set, alpha=self.alpha)
        elif self.lbp_method == 'crown-ibp':
            linear_bounds = model.crown_ibp(set, alpha=self.alpha)
        else:
            raise NotImplementedError(f'Supplied LBP method ({self.lbp_method}) does not exist')

        # linear_bounds = self.ibp_improved_bounds(model, linear_bounds)

        return linear_bounds.cpu()

    def ibp_improved_bounds(self, model, linear_bounds):
        interval_bounds = model.ibp(linear_bounds.region)
        inverse_linear_bounds = LinearBounds(linear_bounds.region, linear_bounds.upper, linear_bounds.lower).concretize()

        ibp_better_lower = interval_bounds.lower > inverse_linear_bounds.upper
        ibp_better_upper = interval_bounds.upper < inverse_linear_bounds.lower

        linear_bounds.lower[0][ibp_better_lower] = 0.0
        linear_bounds.lower[1][ibp_better_lower] = interval_bounds.lower[ibp_better_lower]

        linear_bounds.upper[0][ibp_better_upper] = 0.0
        linear_bounds.upper[1][ibp_better_upper] = interval_bounds.upper[ibp_better_upper]

        return linear_bounds

    def prob_gap(self, linear_bounds):
        gapA = linear_bounds.upper[0].sum(dim=0) - linear_bounds.lower[0].sum(dim=0)
        gap_bias = linear_bounds.upper[1].sum(dim=0) - linear_bounds.lower[1].sum(dim=0)
        return LinearBounds(linear_bounds.region, None, (gapA, gap_bias)).concretize().upper

    def time_propagate(self, linear_bounds):
        gamma = torch.zeros((len(self.partition.safe),))

        for i in trange(self.horizon, desc='Time propagation'):
            gammaA = (linear_bounds.upper[0] * gamma.view(gamma.size(0), 1, 1, -1)).sum(dim=0) - linear_bounds.lower[0].sum(dim=0)
            gammab = (linear_bounds.upper[1] * gamma.view(gamma.size(0), 1, -1)).sum(dim=0) + 1 - linear_bounds.lower[1].sum(dim=0)
            gamma_transition = LinearBounds(linear_bounds.region, None, (gammaA, gammab))
            gamma_transition = gamma_transition.concretize()

            gamma = gamma_transition.upper[:, 0].clamp(min=0.0, max=1.0)
            logger.info(f'Gamma for step {i + 1}: {self.max_initial_set(gamma).item()}')

        return gamma

    def max_initial_set(self, gamma):
        initial = self.dynamics.initial(self.partition.safe.center, self.partition.safe.width / 2)

        gamma_initial = gamma[initial]
        return gamma_initial.max()


class AdditiveGaussianCertifierV2(nn.Module):
    """
    - LP with linear bounds as constraints
    - Grid-based
    """
    def __init__(self, dynamics, factory, partition, horizon, lbp_method='crown', alpha=False, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        # Assumptions:
        # 1. Partitions containing the boundary of the safe / unsafe set belong to the unsafe subpartition
        #    to ensure correctness
        # 2. Regions are hyperrectangular and non-overlapping
        self.partition = partition.to(device)
        self.dynamics = dynamics.to(device)

        factory.kwargs['alpha_iterations'] = 40
        # factory.kwargs['alpha_optimizer_params'] = {'lr': 1e-1, 'betas': (0.5, 0.99)}
        self.factory = factory

        self.horizon = horizon
        self.alpha = alpha
        self.lbp_method = lbp_method
        self.device = device

    def analytic_transition_prob(self, lower, upper):
        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)

        return self.factory.build(MinimizeGap(
            self.dynamics,
            GaussianProbability(loc, scale, lower, upper)
        )).to(self.device)

    @torch.no_grad()
    def certify(self):
        linear_bounds = self.linear_bounds()
        prob_gap = self.prob_gap(linear_bounds)

        gamma = self.time_propagate(linear_bounds)
        unsafe_prob = self.max_initial_set(gamma)

        return unsafe_prob.item()

    def linear_bounds(self):
        in_batch_size = 1000
        in_set = self.partition.safe

        lower, upper = in_set.lower.split(in_batch_size), in_set.upper.split(in_batch_size)
        linear_bounds = []
        for l, u in tqdm(list(zip(lower, upper)), desc='In'):
            linear_bounds.append(self.batch_linear_bounds(HyperRectangle(l, u)))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.lower[1] for bounds in linear_bounds], dim=1)),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds], dim=1), torch.cat([bounds.upper[1] for bounds in linear_bounds], dim=1)),
        )

    def batch_linear_bounds(self, in_set):
        out_batch_size = 5000
        out_set = self.partition.safe

        lower, upper = out_set.lower.split(out_batch_size), out_set.upper.split(out_batch_size)
        transition_probs = [self.analytic_transition_prob(l, u) for l, u in zip(lower, upper)]

        linear_bounds = []
        for transition_prob in tqdm(transition_probs, desc='Out'):
            linear_bounds.append(self.lbp(transition_prob, in_set))

        return LinearBounds(
            in_set.cpu(),
            (torch.cat([bounds.lower[0] for bounds in linear_bounds]), torch.cat([bounds.lower[1] for bounds in linear_bounds])),
            (torch.cat([bounds.upper[0] for bounds in linear_bounds]), torch.cat([bounds.upper[1] for bounds in linear_bounds])),
        )

    def lbp(self, model, set):
        if self.lbp_method == 'crown':
            linear_bounds = model.crown(set, alpha=self.alpha)
        elif self.lbp_method == 'crown-ibp':
            linear_bounds = model.crown_ibp(set, alpha=self.alpha)
        else:
            raise NotImplementedError(f'Supplied LBP method ({self.lbp_method}) does not exist')

        # linear_bounds = self.ibp_improved_bounds(model, linear_bounds)

        return linear_bounds.cpu()

    def ibp_improved_bounds(self, model, linear_bounds):
        interval_bounds = model.ibp(linear_bounds.region)
        inverse_linear_bounds = LinearBounds(linear_bounds.region, linear_bounds.upper, linear_bounds.lower).concretize()

        ibp_better_lower = interval_bounds.lower > inverse_linear_bounds.upper
        ibp_better_upper = interval_bounds.upper < inverse_linear_bounds.lower

        linear_bounds.lower[0][ibp_better_lower] = 0.0
        linear_bounds.lower[1][ibp_better_lower] = interval_bounds.lower[ibp_better_lower]

        linear_bounds.upper[0][ibp_better_upper] = 0.0
        linear_bounds.upper[1][ibp_better_upper] = interval_bounds.upper[ibp_better_upper]

        return linear_bounds

    def prob_gap(self, linear_bounds):
        gapA = linear_bounds.upper[0].sum(dim=0) - linear_bounds.lower[0].sum(dim=0)
        gap_bias = linear_bounds.upper[1].sum(dim=0) - linear_bounds.lower[1].sum(dim=0)
        return LinearBounds(linear_bounds.region, None, (gapA, gap_bias)).concretize().upper

    def time_propagate(self, linear_bounds):
        gamma = torch.zeros((len(self.partition.safe),))

        for i in trange(self.horizon, desc='Time propagation'):
            new_gamma = []

            for j in trange(len(self.partition.safe), desc='LP', mininterval=100, maxinterval=150):
                lower = linear_bounds.lower[0][:, j], linear_bounds.lower[1][:, j]
                upper = linear_bounds.upper[0][:, j], linear_bounds.upper[1][:, j]
                g = self.step_single_partition(gamma, lower, upper, linear_bounds.region[j])

                new_gamma.append(g)

            gamma = torch.as_tensor(new_gamma).clamp(min=0.0, max=1.0)
            logger.info(f'Gamma for step {i + 1}: {self.max_initial_set(gamma).item()}')

        return gamma

    def step_single_partition(self, gamma, lower, upper, region):
        can_remove = (upper[0][:, 0] == 0).all(dim=1) & (upper[1][:, 0] == 0)
        cant_remove = ~can_remove

        t = cp.Variable(cant_remove.int().sum().item())
        t_unsafe = cp.Variable()
        x = cp.Variable(region.lower.size(0))

        prob = cp.Problem(cp.Maximize(gamma[cant_remove] @ t + t_unsafe),
                          [
                              t >= 0.0,
                              t <= 1.0,
                              t_unsafe >= 0.0,
                              t_unsafe <= 1.0,
                              cp.sum(t) + t_unsafe == 1,
                              lower[0][cant_remove, 0] @ x + lower[1][cant_remove, 0] <= t,
                              t <= upper[0][cant_remove, 0] @ x + upper[1][cant_remove, 0],
                              region.lower <= x,
                              x <= region.upper
                          ])

        prob.solve()

        return prob.value

    def max_initial_set(self, gamma):
        initial = self.dynamics.initial(self.partition.safe.center, self.partition.safe.width / 2)

        gamma_initial = gamma[initial]
        return gamma_initial.max()


class AdditiveGaussianCertifierV3(nn.Module):
    """
    - Branch-and-bound for worst case over the summation of linear bounds
    - Grid-based for gamma partitions
    """
    def __init__(self, dynamics, factory, partition, horizon, split_gap_stop_treshold=1e-6, max_set_size=5000,
                 time_propagate_batch_size=10, split_batch_size=1000,
                 lbp_method='crown', alpha=False, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        # Assumptions:
        # 1. Partitions containing the boundary of the safe / unsafe set belong to the unsafe subpartition
        #    to ensure correctness
        # 2. Regions are hyperrectangular and non-overlapping
        self.partition = partition.to(device)
        self.dynamics = dynamics.to(device)

        factory.kwargs['alpha_iterations'] = 40
        self.factory = factory

        self.horizon = horizon

        self.split_gap_stop_treshold = split_gap_stop_treshold
        self.max_set_size = max_set_size

        self.time_propagate_batch_size = time_propagate_batch_size  # NOT IN USE YET
        self.split_batch_size = split_batch_size

        self.lbp_method = lbp_method
        self.alpha = alpha
        self.device = device

    @torch.no_grad()
    def certify(self):
        transition_prob = self.analytic_transition_prob(self.partition.safe)
        gamma = self.time_propagate(transition_prob)

        unsafe_prob = self.max_initial_set(gamma)

        return unsafe_prob.item()

    def analytic_transition_prob(self, set):
        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)

        return self.factory.build(MinimizeGap(
            self.dynamics,
            GaussianProbability(loc, scale, set.lower, set.upper)
        )).to(self.device)

    def time_propagate(self, transition_prob):
        gamma = torch.zeros((len(self.partition.safe),), device=self.device)

        for i in trange(self.horizon, desc='Time propagation'):
            new_gamma = []

            for region in tqdm(self.partition.safe, desc='LBP'):
                g = self.step_single_partition(transition_prob, gamma, region)

                new_gamma.append(g)

            gamma = torch.as_tensor(new_gamma, device=self.device).clamp(min=0.0, max=1.0)
            logger.info(f'Gamma for step {i + 1}: {self.max_initial_set(gamma).item()}')

        return gamma

    def step_single_partition(self, transition_prob, gamma, region):
        set = HyperRectangle(region.lower.unsqueeze(0), region.upper.unsqueeze(0))
        linear_bounds = self.unsafety_prob(transition_prob, gamma, set)
        min, max = self.min_max(linear_bounds)
        last_gap = [torch.finfo(min.dtype).max for _ in range(49)] + [(max.max() - min.max()).item()]

        while not self.should_stop(linear_bounds, min, max, last_gap):
            linear_bounds, keep, prune_all = self.prune(linear_bounds, min, max)

            if prune_all:
                logger.warning(f'Pruning all: {min}, {max}, last gap: {last_gap[-1]}')
                break

            split_index, other = self.pick_for_splitting(linear_bounds, min[keep], max[keep])

            split_linear_bounds = linear_bounds[split_index]
            linear_bounds = linear_bounds[other]

            split_set = self.split(split_linear_bounds)
            split_linear_bounds = self.unsafety_prob(transition_prob, gamma, split_set)

            linear_bounds = self.cat(linear_bounds, split_linear_bounds)
            min, max = self.min_max(linear_bounds)

            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

        return max.max().clamp(min=0.0, max=1.0)

    def unsafety_prob(self, transition_prob, gamma, set):
        linear_bounds = self.lbp(transition_prob, set)
        linear_bounds = LinearBounds(
            set,
            ((linear_bounds.lower[0] * gamma.view(-1, 1, 1, 1)).sum(dim=0) - linear_bounds.upper[0].sum(dim=0), (linear_bounds.lower[1] * gamma.view(-1, 1, 1)).sum(dim=0) + 1 - linear_bounds.upper[1].sum(dim=0)),
            ((linear_bounds.upper[0] * gamma.view(-1, 1, 1, 1)).sum(dim=0) - linear_bounds.lower[0].sum(dim=0), (linear_bounds.upper[1] * gamma.view(-1, 1, 1)).sum(dim=0) + 1 - linear_bounds.lower[1].sum(dim=0))
        )

        return linear_bounds

    def lbp(self, model, set):
        if self.lbp_method == 'crown':
            linear_bounds = model.crown(set, alpha=self.alpha)
        elif self.lbp_method == 'crown-ibp':
            linear_bounds = model.crown_ibp(set, alpha=self.alpha)
        else:
            raise NotImplementedError(f'Supplied LBP method ({self.lbp_method}) does not exist')

        # linear_bounds = self.ibp_improved_bounds(model, linear_bounds)

        return linear_bounds

    def ibp_improved_bounds(self, model, linear_bounds):
        interval_bounds = model.ibp(linear_bounds.region)
        inverse_linear_bounds = LinearBounds(linear_bounds.region, linear_bounds.upper, linear_bounds.lower).concretize()

        ibp_better_lower = interval_bounds.lower > inverse_linear_bounds.upper
        ibp_better_upper = interval_bounds.upper < inverse_linear_bounds.lower

        linear_bounds.lower[0][ibp_better_lower] = 0.0
        linear_bounds.lower[1][ibp_better_lower] = interval_bounds.lower[ibp_better_lower]

        linear_bounds.upper[0][ibp_better_upper] = 0.0
        linear_bounds.upper[1][ibp_better_upper] = interval_bounds.upper[ibp_better_upper]

        return linear_bounds

    def min_max(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        min, max = interval_bounds.lower.clamp(min=0), interval_bounds.upper.clamp(max=1)

        return min.view(-1), max.view(-1)

    def should_stop(self, set, min, max, last_gap, max_set_size=None):
        gap = last_gap[-1]
        abs_max = max.max().item()

        improvement_threshold = torch.tensor(last_gap).max() * 0.999

        max_set_size = max_set_size or self.max_set_size

        cond1 = len(set) > max_set_size  # Active set is larger than threshold
        cond2 = gap <= self.split_gap_stop_treshold  # Too small a gap, can't improve much more
        cond3 = torch.tensor(last_gap[-10:]).min() >= improvement_threshold  # No progress has been made for the last n steps

        should_stop = cond1 or cond2 or cond3

        if should_stop:
            logger.info(f'Gap: {gap}, set size: {len(set)}, upper bound: {abs_max}')
        else:
            logger.debug(f'Gap: {gap}, set size: {len(set)}, upper bound: {abs_max}')

        return should_stop

    def prune(self, linear_bounds, min, max):
        largest_lower_bound = min.max()

        prune = max <= largest_lower_bound
        keep = ~prune

        if torch.all(prune):
            return linear_bounds, keep, True

        linear_bounds = linear_bounds[keep]
        return linear_bounds, keep, False

    def pick_for_splitting(self, linear_bounds, min_, max_):
        split_size = min(self.split_batch_size, len(linear_bounds))

        split_indices = max_.topk(split_size).indices

        other_indices = torch.full((len(linear_bounds),), True, dtype=torch.bool, device=max_.device)
        other_indices[split_indices] = False

        return split_indices, other_indices

    def split(self, linear_bounds):
        set = linear_bounds.region

        relative_width = set.width / set.width.min(dim=-1, keepdim=True).values

        split_dim = ((linear_bounds.upper[0] - linear_bounds.lower[0]).abs()[:, 0] * relative_width).argmax(dim=-1)
        partition_indices = torch.arange(0, len(set), device=set.lower.device)
        split_dim = (partition_indices, split_dim)

        lower, upper = set.lower, set.upper
        mid = set.center

        p1_lower = lower.clone()
        p1_upper = upper.clone()
        p1_upper[split_dim] = mid[split_dim]

        p2_lower = lower.clone()
        p2_upper = upper.clone()
        p2_lower[split_dim] = mid[split_dim]

        lower, upper = torch.cat([p1_lower, p2_lower]), torch.cat([p1_upper, p2_upper])

        assert torch.all(lower <= upper)

        return HyperRectangle(lower, upper)

    def cat(self, lb1, lb2):
        return LinearBounds(
            HyperRectangle(torch.cat([lb1.region.lower, lb2.region.lower]), torch.cat([lb1.region.upper, lb2.region.upper])),
            (torch.cat([lb1.lower[0], lb2.lower[0]]), torch.cat([lb1.lower[1], lb2.lower[1]])),
            (torch.cat([lb1.upper[0], lb2.upper[0]]), torch.cat([lb1.upper[1], lb2.upper[1]]))
        )

    def max_initial_set(self, gamma):
        initial = self.dynamics.initial(self.partition.safe.center, self.partition.safe.width / 2)

        gamma_initial = gamma[initial]
        return gamma_initial.max()


class AdditiveGaussianIMDPCertifier(nn.Module):
    def __init__(self, dynamics, factory, partition, horizon, sigma_neighborhood=6, prune_threshold=1e-6,
                 lbp_method='crown', alpha=False, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        # Assumptions:
        # 1. Partitions containing the boundary of the safe / unsafe set belong to the unsafe subpartition
        #    to ensure correctness
        # 2. Regions are hyperrectangular and non-overlapping
        self.safe_set = partition.safe.to(device)
        self.dynamics = dynamics.to(device)

        factory.kwargs['alpha_iterations'] = 40
        self.factory = factory
        self.nominal_model = factory.build(self.dynamics).to(device)

        self.horizon = horizon
        self.sigma_neighborhood = sigma_neighborhood
        self.prune_threshold = prune_threshold

        self.alpha = alpha
        self.lbp_method = lbp_method
        self.device = device

        lower, upper = self.safe_set.lower.unsqueeze(1), self.safe_set.upper.unsqueeze(1)

        def d2dx2(x):
            return erf_diff_double_derivative(lower, upper, x)

        self.inflection_points = erf_diff_find_inflection_points(lower, upper, d2dx2)

    @torch.no_grad()
    def certify(self):
        neighborhoods = self.nominal_neighborhood()
        bounds = self.bounds(neighborhoods)

        gamma = self.time_propagate(bounds)
        unsafe_prob = self.max_initial_set(gamma)

        return unsafe_prob.item()

    def nominal_neighborhood(self):
        linear_bounds = self.lbp(self.nominal_model, self.safe_set, bound_upper=True)
        interval_bounds = linear_bounds.concretize()

        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)
        lower = interval_bounds.lower + loc - scale * self.sigma_neighborhood
        upper = interval_bounds.upper + loc + scale * self.sigma_neighborhood

        return zip(lower, upper)

    def bounds(self, neighborhoods):
        bounds = []
        for l, u, neighborhood in tqdm(list(zip(self.safe_set.lower, self.safe_set.upper, neighborhoods)),
                                       desc='In'):
            bounds.append(self.partition_bounds(l, u, neighborhood))

        return bounds

    def partition_bounds(self, l, u, neighborhood):
        overlap = overlap_rectangle(neighborhood[0].unsqueeze(0), neighborhood[1].unsqueeze(0), self.safe_set.lower,
                                    self.safe_set.upper)
        neighbors = torch.where(overlap)[0]

        transition_prob = self.analytic_transition_prob(neighbors)
        linear_bounds = self.lbp(transition_prob, HyperRectangle(l.unsqueeze(0), u.unsqueeze(0)))
        interval_bounds = linear_bounds.concretize()

        return neighbors.cpu(), interval_bounds.to(torch.device('cpu'))

    def analytic_transition_prob(self, neighbors):
        out_set = self.safe_set[neighbors]
        lower, upper = out_set.lower, out_set.upper

        loc, scale = self.dynamics.v
        loc, scale = loc.to(self.device), scale.to(self.device)

        self.factory.kwargs['erf_diff_inflection'] = \
            (self.inflection_points[0][0][neighbors], self.inflection_points[0][1][neighbors]), \
            (self.inflection_points[1][0][neighbors], self.inflection_points[1][1][neighbors])

        return self.factory.build(MinimizeGap(
            self.dynamics,
            GaussianProbability(loc, scale, lower, upper)
        )).to(self.device)

    def lbp(self, model, set, bound_upper=False):
        if self.lbp_method == 'crown':
            linear_bounds = model.crown(set, alpha=self.alpha, bound_upper=bound_upper)
        elif self.lbp_method == 'crown-ibp':
            linear_bounds = model.crown_ibp(set, alpha=self.alpha, bound_upper=bound_upper)
        else:
            raise NotImplementedError(f'Supplied LBP method ({self.lbp_method}) does not exist')

        return linear_bounds.cpu()

    def time_propagate(self, linear_bounds):
        gamma = torch.zeros((len(self.safe_set),))

        for i in trange(self.horizon, desc='Time propagation'):
            new_gamma = torch.zeros((len(self.safe_set),))

            for j, (neighbors, bound) in enumerate(linear_bounds):
                gamma_max = (gamma[neighbors] * bound.lower).sum() + (1 - bound.lower.sum())
                new_gamma[j] = gamma_max.clamp(min=0.0, max=1.0)

            gamma = new_gamma
            logger.debug(f'Gamma for step {i + 1}: {self.max_initial_set(gamma).item()}')

        return gamma

    def max_initial_set(self, gamma):
        initial = self.dynamics.initial(self.safe_set.center, self.safe_set.width / 2)

        gamma_initial = gamma[initial]
        return gamma_initial.max()
