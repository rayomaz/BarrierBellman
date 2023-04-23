import torch
from bounds.partitioning import Partition


def linear_grid_partition(args, config, dynamics):
    partitioning_config = config['partitioning']

    x_lower = partitioning_config['state_space'][0]
    x_upper = partitioning_config['state_space'][1]

    x1_space = torch.linspace(x_lower, x_upper, partitioning_config['num_slices'][0] + 1)

    cell_width = (x1_space[1:] - x1_space[:-1]) / 2
    cell_centers = (x1_space[:-1] + x1_space[1:]) / 2

    cell_width, cell_centers = cell_width.unsqueeze(-1), cell_centers.unsqueeze(-1)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    initial_mask = dynamics.initial(cell_centers, cell_width)

    if args.space == 'equivalent_space':
        safe_mask = dynamics.state_space(cell_centers, cell_width)
    elif args.space == 'modified_space':
        safe_mask = dynamics.safe(cell_centers, cell_width)
        
    unsafe_mask = dynamics.unsafe(cell_centers, cell_width)

    partition = Partition(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    return partition