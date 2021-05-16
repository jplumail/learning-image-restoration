import torch
import torch.nn as nn
import torch.nn.functional as F



class DiffusionNetStep(nn.Module):
    def __init__(self, rbf, n_channels=3, n_filters=5, filter_size=5):
        super(DiffusionNetStep, self).__init__()
        self.padding = filter_size // 2 + (filter_size - 2 * (filter_size // 2)) - 1
        self.conv = nn.Conv2d(n_channels, n_filters, filter_size, padding=self.padding, bias=False, padding_mode="replicate")
        self.means = rbf["means"]
        self.std = rbf["std"]
        n_rbf = self.means.shape[0]
        self.weights_rbf = nn.Parameter(torch.ones(n_filters, n_rbf) / n_rbf)
        self.reaction_weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
    
    def compute_nonlinearity(self, u):
        s = torch.zeros(u.shape, dtype=torch.float32).to(u.device)
        for i in range(len(self.means)):
            v = (u - self.means[i]) / self.std[i]
            v = torch.exp(-torch.pow(v, 2))
            v = self.weights_rbf[None,:,i,None,None] * v
            s += v
        return s
    
    def compute_diffusion(self, u):
        # First conv
        u = self.conv(u)

        # Non linearity
        u = self.compute_nonlinearity(u)

        # Rotated conv
        rotated_kernels_weights = torch.rot90(self.conv.weight, 2, [2, 3]) # rotate 2 times conv1 weights
        rotated_kernels_weights = torch.transpose(rotated_kernels_weights, dim0=0, dim1=1)
        u = nn.ReplicationPad2d(self.padding)(u)
        u = F.conv2d(u, rotated_kernels_weights)
        return u
    
    def forward(self, u, f):
        u = u - (self.compute_diffusion(u) + self.reaction_weight * (u - f))
        return u


class DiffusionNet(nn.Module):
    def __init__(self, n_rbf=63, T=5, **dnet_args):
        super(DiffusionNet, self).__init__()
        means = nn.Parameter(torch.randn(n_rbf, dtype=torch.float32))
        std = nn.Parameter(torch.randn(n_rbf, dtype=torch.float32) + 1.0)
        rbfs = {"means": means, "std": std}
        self.dnets = nn.ModuleList([DiffusionNetStep(rbfs, **dnet_args) for i in range(T)])
    
    def step(self, u, f, i):
        return self.dnets[i].forward(u, f)
    
    def forward(self, u, f):
        for i in range(len(self.dnets)):
            u = self.step(u, f, i)
        return u