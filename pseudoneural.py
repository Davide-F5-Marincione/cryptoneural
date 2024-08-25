import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm

BITS = 32


def pseudo_random(n, c, i):
    return ((i<<n)%2**(BITS-1)) + c


class MyModel(nn.Module):
    def __init__(self, layers):
        super().__init__()

        modules = []
        prev_size = BITS
        for size in layers:
            linear = nn.Linear(prev_size, size)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='linear')
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            modules.append(nn.GELU())
            modules.append(nn.LayerNorm(size))

            prev_size = size

        linear = nn.Linear(prev_size, BITS, bias=False)
        nn.init.kaiming_normal_(linear.weight, nonlinearity='linear')

        modules.append(linear)

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)
    
def our_loss1(y_pred, y_true):
    return torch.relu(torch.where(y_true <= 0, y_pred, -y_pred)).mean()

def binarization(numba):
    return [int(bit) for bit in bin(numba)[2:].zfill(BITS)]

model = MyModel([64, 64])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.L1Loss()

N = 10
C = 101
BATCH_SIZE = 32
TRAINING_SEED = 42
STEPS = 100

random.seed(TRAINING_SEED)

for i in (progressbar:=tqdm.trange(STEPS)):
    x = [random.randint(0, 2**BITS - 1) for _ in range(BATCH_SIZE)]
    y = [pseudo_random(N, C, numba) for numba in x]

    x = torch.as_tensor([binarization(numba) for numba in x])
    y = torch.as_tensor([binarization(numba) for numba in y])

    y_pred = model(x.float())
    loss = loss_fn(y_pred, (y.float() * 4 - 2))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    correct_bits = ((y_pred > 0) == (y > 0)).float().mean().item()
    progressbar.set_description(f"Loss: {loss.item():.3f}, Correct bits: {correct_bits:.3f}")