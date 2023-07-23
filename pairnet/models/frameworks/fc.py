import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS, NORM_LAYERS


@NORM_LAYERS.register_module()
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@ACTIVATION_LAYERS.register_module()
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FC(nn.Module):
    """
    7 layer fc, 0.2M parameters
    """

    def __init__(self, input_dim=100, hidden_dim=128, output_dim=100):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = FC()
    input = torch.rand((2, 100, 100))
    output = model(input)
    print(output.shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([torch.numel(p) for p in model_parameters]))
