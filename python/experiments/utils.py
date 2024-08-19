import json

import torch
from bounds.partitioning import Partition

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def grid_partition(args, config):

    partitions = []
    
    # Number of dynamics laws
    num_controllers = config['dynamics']['num_controllers']

    for j in range(0, num_controllers):

        x_lower = config['dynamics']['safe_set'][j][0]
        x_upper = config['dynamics']['safe_set'][j][1]

        cell_widths = []
        cell_centers = []

        for i in range(config['dynamics']['dim']):
            x_space = torch.linspace(x_lower[i], x_upper[i], config['partitioning']['num_slices'][i] + 1)

            cell_width = (x_space[1:] - x_space[:-1]) / 2
            cell_center = (x_space[:-1] + x_space[1:]) / 2

            cell_widths.append(cell_width)
            cell_centers.append(cell_center)

        width, center = torch.cartesian_prod(*cell_widths), torch.cartesian_prod(*cell_centers)
        lower_x, upper_x = center - width, center + width

        partition = Partition((lower_x, upper_x))

        partitions.append(partition)

    return partitions

