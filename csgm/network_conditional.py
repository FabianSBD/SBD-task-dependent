# https://github.com/tanelp/tiny-diffusion

import torch
from torch import nn

from .fourier_neural_operator import FourierNeuralOperator2D




class ConditionalScoreModel2D(nn.Module):

    def __init__(self,
                 modes: int = 5,
                 hidden_dim: int = 128,
                 nlayers: int = 5,
                 nt = 1,
                 norm_layersize = None):
        super().__init__()

        self.nt = nt
        self.network = FourierNeuralOperator2D(modes, hidden_dim, 2, 1,
                                               nlayers, norm_layersize = norm_layersize)

    def forward(self, x, t):
        t = t.reshape(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2],
                                          1) / self.nt * 2.0 - 1.0
        z = torch.cat((x, t), dim=-1)
        z = self.network(z)
        return z

class ConditionalScoreModel2Dy(nn.Module):

    def __init__(self,
                 modes: int = 5,
                 hidden_dim: int = 128,
                 nlayers: int = 5,
                 nt: int = 500):
        super().__init__()

        self.nt = nt
        self.network = FourierNeuralOperator2D(modes, hidden_dim, 3, 1,
                                               nlayers)

    def forward(self, x, y, t):
        t = t.reshape(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2],
                                          1) / self.nt * 2.0 - 1.0

        if x.shape[2] - y.shape[2] > 0:
            y = torch.nn.functional.pad(y, (0, 0, x.shape[2] - y.shape[2], 0))

       
        
        z = torch.cat((x, y, t), dim=-1)
        z = self.network(z)
        return z
