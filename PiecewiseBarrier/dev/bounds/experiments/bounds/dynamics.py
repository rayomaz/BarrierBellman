import abc


class StochasticDynamics(abc.ABC):
    @abc.abstractmethod
    def initial(self, x, eps=None):
        raise NotImplementedError()

    def safe(self, x, eps=None):
        return ~self.unsafe(x, eps=eps)

    def unsafe(self, x, eps=None):
        return ~self.safe(x, eps=eps)

    @abc.abstractmethod
    def state_space(self, x, eps=None):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def volume(self):
        raise NotImplementedError()


class AdditiveNoiseDynamics(StochasticDynamics, abc.ABC):
    pass


class AdditiveGaussianDynamics(AdditiveNoiseDynamics, abc.ABC):
    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplementedError()
