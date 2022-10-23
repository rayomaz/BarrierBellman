import logging

import torch
from torch.distributions import Normal

from .dynamics import AdditiveGaussianDynamics

logger = logging.getLogger(__name__)


def monte_carlo_simulation(args, config, dynamics):
    assert isinstance(dynamics, AdditiveGaussianDynamics)

    num_initial_states = config['monte_carlo']['num_initial_states']
    num_particles = config['monte_carlo']['num_particles']
    horizon = config['dynamics']['horizon']

    x = dynamics.sample_initial(num_initial_states).to(args.device).unsqueeze(1).expand((num_initial_states, num_particles, -1))
    traj = [x]

    unsafe = torch.full((num_initial_states, num_particles), False, device=args.device)

    loc, scale = dynamics.v
    dist = Normal(loc.to(args.device), scale.to(args.device))

    for _ in range(horizon):
        noise = dist.sample((num_initial_states, num_particles))
        x = dynamics(x) + noise
        traj.append(x)

        unsafe |= dynamics.unsafe(x) & ~unsafe

    max_unsafe = unsafe.float().sum(dim=1).max()

    logger.info(f'Num initial states: {num_initial_states}/num particles: {num_particles}/unsafe: {max_unsafe}, ratio unsafe: {max_unsafe / num_particles}')

    return x, torch.stack(traj), unsafe
