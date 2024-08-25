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
    def __init__(self, bits):
        super().__init__()

        self.bits = bits

        self.module = nn.Sequential(
            nn.Linear(bits, 2*bits),
            nn.LeakyReLU(.1),
            nn.Linear(2*bits, 2*bits),
            nn.LeakyReLU(.1),
            nn.Linear(2*bits, bits, bias=False)
        )

        nn.init.kaiming_normal_(self.module[0].weight, mode="fan_out", a=.1, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.module[0].weight, mode="fan_out", a=.1, nonlinearity='leaky_relu')
        nn.init.xavier_normal_(self.module[4].weight)
        nn.init.zeros_(self.module[0].bias)
        nn.init.zeros_(self.module[2].bias)
        # nn.init.zeros_(self.module[4].bias)

    def forward(self, x):
        return self.module(x)
    
def our_loss1(y_pred, y_true):
    return torch.relu(torch.where(y_true <= 0, y_pred, -y_pred)).mean()

def binarization(numba):
    return [int(bit) for bit in bin(numba)[2:].zfill(BITS)]

model = MyModel(BITS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

N = 10
C = 101
BATCH_SIZE = 32
TRAINING_SEED = 42
STEPS = 500

random.seed(TRAINING_SEED)

for i in (progressbar:=tqdm.trange(STEPS)):
    x = [random.randint(0, 2**BITS - 1) for _ in range(BATCH_SIZE)]
    y = [pseudo_random(N, C, numba) for numba in x]

    x = torch.as_tensor([binarization(numba) for numba in x])
    y = torch.as_tensor([binarization(numba) for numba in y])

    y_pred = model(x.float())
    loss = loss_fn(y_pred, (y.float() * 2 - 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    correct_bits = ((y_pred > 0) == (y > 0)).float().mean().item()
    progressbar.set_description(f"Loss: {loss.item():.3f}, Correct bits: {correct_bits:.3f}")