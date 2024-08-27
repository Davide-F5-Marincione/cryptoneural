import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        
        nn.init.xavier_normal_(self.out[0].weight)
        nn.init.zeros_(self.out[0].bias)
        nn.init.zeros_(self.out[2].weight)
        nn.init.zeros_(self.out[2].bias)
        
    def forward(self, x):
        return x + self.out(x)

class CryptFuncA(nn.Module):
    def __init__(self, dim, bits, n_layers=5):
        super().__init__()

        self.in_layer = nn.Linear(bits, dim)
        nn.init.xavier_normal_(self.in_layer.weight)
        nn.init.zeros_(self.in_layer.bias)

        self.layers = nn.Sequential(*[ResBlock(dim) for _ in range(n_layers)])
        
        self.out = nn.Linear(dim, bits, bias=False)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        return self.out(x)