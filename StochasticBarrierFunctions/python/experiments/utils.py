import json

import torch
from bounds.partitioning import Partition


def load_config(config_path):

    if type(config_path) is list:

        config_dict = {}  # Create an empty dictionary to store the data

        for idx, path in enumerate(config_path):
            try:
                with open(path, 'r') as f:
                    config_dict[idx] = json.load(f)
            except FileNotFoundError:
                print(f"File not found: {config_path}")

        return config_dict      # Return dict of multiple .json dicts

    elif type(config_path) is str:
        with open(config_path, 'r') as f:
            return json.load(f)


def grid_partition(args, config):
    x_lower = config['dynamics']['safe_set'][0]
    x_upper = config['dynamics']['safe_set'][1]

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

    return partition
