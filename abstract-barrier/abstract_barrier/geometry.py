import torch


def overlap_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    # Separating axis theorem
    return (partition_upper >= rect_lower).all(dim=-1) & (partition_lower <= rect_upper).all(dim=-1)


def overlap_circle(partition_lower, partition_upper, center, radius):
    closest_point = torch.max(partition_lower, torch.min(partition_upper, center))
    distance = (closest_point - center).norm(dim=-1)
    return distance <= radius


def overlap_outside_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    return (partition_upper >= rect_lower).any(dim=-1) | (partition_lower <= rect_upper).any(dim=-1)


def overlap_outside_circle(partition_lower, partition_upper, center, radius):
    farthest_point = torch.where((partition_lower - center).abs() > (partition_upper - center).abs(), partition_lower, partition_upper)
    distance = (farthest_point - center).norm(dim=-1)
    return distance >= radius