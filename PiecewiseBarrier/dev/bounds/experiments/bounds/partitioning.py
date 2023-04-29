import torch
from torch import nn


class SubPartition(nn.Module):
    # This is nn.Module to allow ".to(device)"
    def __init__(self, bounds):
        super().__init__()

        lower, upper = bounds

        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)

    @property
    def volume(self):
        return self.volumes.sum()

    @property
    def volumes(self):
        return self.width.prod(dim=-1)

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def __getitem__(self, idx):
        return SubPartition(
            (self.lower[idx], self.upper[idx])
        )

    def __len__(self):
        return self.lower.size(0)


class Partition(nn.Module):
    def __init__(self, safe, state_space=None):
        super().__init__()

        self.safe = self.convert(safe)
        self.state_space = self.convert(state_space)

    @staticmethod
    def convert(partitions):
        if partitions is None:
            return None
        elif isinstance(partitions, SubPartition):
            return partitions
        else:
            return SubPartition(partitions)

    def __len__(self):
         return len(self.safe) + \
               (len(self.state_space) if self.state_space else 0)

    def __getitem__(self, idx):
        safe_idx, state_space_idx = idx

        return Partition(
            self.safe[safe_idx],
            self.state_space[state_space_idx] if self.state_space else None,
        )
