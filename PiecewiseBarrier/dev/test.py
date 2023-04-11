import math

import torch
from bound_propagation import VectorMul, Parallel, Add, Mul, FixedLinear, ElementWiseLinear, Pow, Exp, VectorSub
from torch import nn
from torch.nn import Identity

import numpy as np

from abstract_barrier.bounds import ErfDiff


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

k = 1
z_upper, z_lower = torch.tensor([]), torch.tensor([])

# Mean is some function of x.
mean = nn.Sequential(np.dot(0.95, x))

erfdiff_term = nn.Sequential(
    mean,
    Parallel(
        ErfDiff(0.0, 1.0, z_lower[:2], z_upper[2:]),
        nn.Sequential(
            FixedLinear(torch.tensor([[-1], [-1]]), torch.stack([z_lower[1], z_upper[1]])),
            Pow(2),
            ElementWiseLinear(-1/2),
            Exp(),
            VectorSub()
        ),
        ErfDiff(0.0, 1.0, z_lower[3:], z_upper[3:]),
        split_size=[1, 1, k - 2]
    ),
    ProdAll(k),
    ElementWiseLinear(1/(2**(k - 1) * math.sqrt(2 * math.pi)))
)

transition = nn.Sequential()
transition_term = Mul(mean, transition)

model = Add(
        erfdiff_term,
        transition_term
    )


# ### New approach
# erfdiff = ErfDiff(0.0, 1.0, z_lower, z_upper)
# sqrexp = nn.Sequential(
#             FixedLinear(torch.tensor([[-1], [-1]]), torch.stack([z_lower, z_upper])),
#             Pow(2),
#             ElementWiseLinear(-1/2),
#             Exp(),
#             VectorSub()
#         )


# stacked = nn.Sequential(
#     dynamics,
#     Parallel(erfdiff, sqrexp)
# )

# erfdiff_term = nn.Sequential(
#     stacked,
#     Parallel(
#         nn.Sequential(
#             Select([k+1, 2, 3, 4, 5]),
#             ProdAll(k),
#             ElementWiseLinear(1/(2**(k - 1) * math.sqrt(2 * math.pi)))
#         ),
#         nn.Sequential(
#             Select([1, k+2, 3, 4, 5]),
#             ProdAll(k),
#             ElementWiseLinear(1/(2**(k - 1) * math.sqrt(2 * math.pi)))
#         ),
#         nn.Sequential(
#             Select([1, 2, k+3, 4, 5]),
#             ProdAll(k),
#             ElementWiseLinear(1/(2**(k - 1) * math.sqrt(2 * math.pi)))
#         ),
#         nn.Sequential(
#             Select([1, 2, 3, k+4, 5]),
#             ProdAll(k),
#             ElementWiseLinear(1/(2**(k - 1) * math.sqrt(2 * math.pi)))
#         ),
#         nn.Sequential(
#             Select([1, 2, 3, 4, k+5]),
#             ProdAll(k),
#             ElementWiseLinear(1/(2**(k - 1) * math.sqrt(2 * math.pi)))
#         )
#     )
# )

