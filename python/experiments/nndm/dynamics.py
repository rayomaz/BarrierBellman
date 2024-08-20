import torch
from torch import nn
import onnx2torch

from bounds.dynamics import AdditiveGaussianDynamics


class NNDM(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        # Load model from ONNX
        onnx_model_path = dynamics_config['nn_model']
        graph_model = onnx2torch.convert(onnx_model_path)

        # Assume that the model is structured as a Sequential
        modules = graph_model.children()
        super().__init__(*modules)

        self.sigma = dynamics_config['sigma']
        self.safe = torch.as_tensor(dynamics_config['safe_set'][0][0]), torch.as_tensor(dynamics_config['safe_set'][0][1])
        self._dim = dynamics_config['dim']

    @property
    def v(self):
        return torch.tensor([0.0]), torch.as_tensor(self.sigma)

    @property
    def safe_set(self):
        return self.safe

    @property
    def dim(self):
        return self._dim
