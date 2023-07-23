import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    """
    2 transformer, 0.3M parameters
    """

    def __init__(self):
        super(Attn, self).__init__()
        self.in_fc1 = nn.Linear(100, 128)
        self.in_fc2 = nn.Linear(100, 128)
        self.transformer1 = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, activation=nn.GELU(), dim_feedforward=256
        )
        self.transformer2 = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, activation=nn.GELU(), dim_feedforward=256
        )
        self.out_fc1 = nn.Linear(128, 100)
        self.out_fc2 = nn.Linear(128, 100)

    def forward(self, x):
        x1 = self.in_fc1(x)
        x1 = self.transformer1(x1)
        x1 = self.out_fc1(x1)
        x2 = self.in_fc2(x.transpose(1, 2))
        x2 = self.transformer2(x2)
        x2 = self.out_fc1(x2)
        return x1 + x2


if __name__ == "__main__":
    model = Attn()
    input = torch.rand((2, 100, 100))
    output = model(input)
    print(output.shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([torch.numel(p) for p in model_parameters]))
