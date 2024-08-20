from torch import nn


class Partition(nn.Module):
    """
    A data class (not @dataclass) to represent the partitioning of the safe set.
    This class is nn.Module subclass to allow ".to(device)"

    Assumptions:
    - Each region is a hyperrectangle

    """

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
        return Partition(
            (self.lower[idx], self.upper[idx])
        )

    def __len__(self):
        return self.lower.size(0)
