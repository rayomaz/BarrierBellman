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


def safe_linear_grid_partition(args, config, dynamics):
    partitioning_config = config['partitioning']

    x_lower = partitioning_config['state_space'][0]
    x_upper = partitioning_config['state_space'][1]

    x1_space = torch.linspace(x_lower, x_upper, partitioning_config['num_slices'][0] + 1)

    cell_width = (x1_space[1:] - x1_space[:-1]) / 2
    cell_centers = (x1_space[:-1] + x1_space[1:]) / 2

    cell_width, cell_centers = cell_width.unsqueeze(-1), cell_centers.unsqueeze(-1)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    x_lower = lower_x[0]
    x_upper = upper_x[-1]

    idx = config.get('index')

    # Cat: current partition defined by idx, and full state space
    lower_x = torch.cat((lower_x[idx].unsqueeze(0), x_lower.unsqueeze(0)), dim=0)
    upper_x = torch.cat((upper_x[idx].unsqueeze(0), x_upper.unsqueeze(0)), dim=0)

    if args.space == 'equivalent_space':
        safe_mask = dynamics.state_space(cell_centers, cell_width)
    elif args.space == 'modified_space':
        safe_mask = dynamics.safe(cell_centers, cell_width)
    
    #! To do: remove _mask (not needed for this work: defined in bounds/partitioning)
    safe_mask = safe_mask[0:2]
    unsafe_mask = dynamics.unsafe(cell_centers, cell_width)
    unsafe_mask = unsafe_mask[0:2]

    initial_mask = unsafe_mask

    partition = Partition(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    return partition
