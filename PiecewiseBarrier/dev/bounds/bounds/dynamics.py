import abc


class AdditiveGaussianDynamics(abc.ABC):
    """
    Represent dynamics with additive Gaussian noise.

    Assumptions:
    - The safe set is a hyperrectangle

    """
    @property
    @abc.abstractmethod
    def safe_set(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplementedError()
