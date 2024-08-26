import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm

from matplotlib import pyplot as plt

EARLY_STOPPING_THRESH = .9999
BITS = 32
N = 10
C = 101
BATCH_SIZE = 8196
SEED = 42
STEPS = 400000
UPDATE_EVERY = 10
LOOPS = 2
S_BOX = np.asarray([
	0xa3, 0xd7, 0x09, 0x83, 0xf8, 0x48, 0xf6, 0xf4, 
	0xb3, 0x21, 0x15, 0x78, 0x99, 0xb1, 0xaf, 0xf9, 
	0xe7, 0x2d, 0x4d, 0x8a, 0xce, 0x4c, 0xca, 0x2e, 
	0x52, 0x95, 0xd9, 0x1e, 0x4e, 0x38, 0x44, 0x28, 
	0x0a, 0xdf, 0x02, 0xa0, 0x17, 0xf1, 0x60, 0x68, 
	0x12, 0xb7, 0x7a, 0xc3, 0xe9, 0xfa, 0x3d, 0x53, 
	0x96, 0x84, 0x6b, 0xba, 0xf2, 0x63, 0x9a, 0x19, 
	0x7c, 0xae, 0xe5, 0xf5, 0xf7, 0x16, 0x6a, 0xa2, 
	0x39, 0xb6, 0x7b, 0x0f, 0xc1, 0x93, 0x81, 0x1b, 
	0xee, 0xb4, 0x1a, 0xea, 0xd0, 0x91, 0x2f, 0xb8, 
	0x55, 0xb9, 0xda, 0x85, 0x3f, 0x41, 0xbf, 0xe0, 
	0x5a, 0x58, 0x80, 0x5f, 0x66, 0x0b, 0xd8, 0x90, 
	0x35, 0xd5, 0xc0, 0xa7, 0x33, 0x06, 0x65, 0x69, 
	0x45, 0x00, 0x94, 0x56, 0x6d, 0x98, 0x9b, 0x76, 
	0x97, 0xfc, 0xb2, 0xc2, 0xb0, 0xfe, 0xdb, 0x20, 
	0xe1, 0xeb, 0xd6, 0xe4, 0xdd, 0x47, 0x4a, 0x1d, 
	0x42, 0xed, 0x9e, 0x6e, 0x49, 0x3c, 0xcd, 0x43, 
	0x27, 0xd2, 0x07, 0xd4, 0xde, 0xc7, 0x67, 0x18, 
	0x89, 0xcb, 0x30, 0x1f, 0x8d, 0xc6, 0x8f, 0xaa, 
	0xc8, 0x74, 0xdc, 0xc9, 0x5d, 0x5c, 0x31, 0xa4, 
	0x70, 0x88, 0x61, 0x2c, 0x9f, 0x0d, 0x2b, 0x87, 
	0x50, 0x82, 0x54, 0x64, 0x26, 0x7d, 0x03, 0x40, 
	0x34, 0x4b, 0x1c, 0x73, 0xd1, 0xc4, 0xfd, 0x3b, 
	0xcc, 0xfb, 0x7f, 0xab, 0xe6, 0x3e, 0x5b, 0xa5, 
	0xad, 0x04, 0x23, 0x9c, 0x14, 0x51, 0x22, 0xf0, 
	0x29, 0x79, 0x71, 0x7e, 0xff, 0x8c, 0x0e, 0xe2, 
	0x0c, 0xef, 0xbc, 0x72, 0x75, 0x6f, 0x37, 0xa1, 
	0xec, 0xd3, 0x8e, 0x62, 0x8b, 0x86, 0x10, 0xe8, 
	0x08, 0x77, 0x11, 0xbe, 0x92, 0x4f, 0x24, 0xc5, 
	0x32, 0x36, 0x9d, 0xcf, 0xf3, 0xa6, 0xbb, 0xac, 
	0x5e, 0x6c, 0xa9, 0x13, 0x57, 0x25, 0xb5, 0xe3, 
	0xbd, 0xa8, 0x3a, 0x01, 0x05, 0x59, 0x2a, 0x46], dtype=np.uint8)

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

PERMUTATION_LIST = np.random.permutation(BITS)
PERMUTATION_MATRIX = np.zeros((BITS, BITS), dtype=np.uint8)

for i in range(BITS):
    PERMUTATION_MATRIX[i, PERMUTATION_LIST[i]] = 1


def pseudo_random(i):
    return ((i<<N)%2**(BITS-1)) + C

def f(i):
    k0=0b10101010
    k1=0b11110101
    k2=0b01111101
    k3=0b01100010
    
    words = np.empty((i.shape[0],4), dtype=np.uint8)

    #S-BOX
    i0= np.bitwise_and(i,0x000000ff)
    i0= np.bitwise_xor(i0,k0)
    words[:, 0]= S_BOX[i0]

    i1= np.right_shift(np.bitwise_and(i, 0x0000ff00),8)
    i1= np.bitwise_xor(i1,k1)
    words[:, 1]= S_BOX[i1]

    i2= np.right_shift(np.bitwise_and(i,0x00ff0000),16)
    i2= np.bitwise_xor(i2,k2)
    words[:, 2]= S_BOX[i2]

    i3= np.right_shift(np.bitwise_and(i, 0xff000000),24)
    i3= np.bitwise_xor(i3,k3)
    words[:, 3]= S_BOX[i3]

    #P-BOX
    bits = np.unpackbits(words).reshape(-1, BITS) @ PERMUTATION_MATRIX
    y = np.packbits(bits).view(np.uint32)

    return y

def pseudo_random2(i):
    for _ in range(LOOPS):
        i = f(i)
    return i
class ResBlock(nn.Module):
    def __init__(self, outer_size, inner_size):
        super().__init__()
        
        self.out = nn.Sequential(
            nn.Linear(outer_size, outer_size),
            nn.LeakyReLU(.1),
            nn.BatchNorm1d(outer_size)
        )
        
        nn.init.kaiming_normal_(self.out[0].weight, a=.1, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.zeros_(self.out[0].bias)
        nn.init.zeros_(self.out[2].weight)
        nn.init.zeros_(self.out[2].bias)
        
    def forward(self, x):
        return x + self.out(x)


class CryptFuncA(nn.Module):
    def __init__(self, outer_size, inner_size, n_layers=5):
        super().__init__()

        self.in_layer = nn.Linear(BITS, outer_size)
        nn.init.xavier_normal_(self.in_layer.weight)
        nn.init.zeros_(self.in_layer.bias)

        self.layers = nn.Sequential(*[ResBlock(outer_size, inner_size) for _ in range(n_layers)])
        
        self.out = nn.Linear(outer_size, BITS, bias=False)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        return self.out(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

def model_looper(model, x):
    for _ in range(LOOPS):
        x = torch.sigmoid(model(x))
    return x

def binarization(numba):
    return np.unpackbits(np.asarray([numba], dtype=np.uint32).view(np.uint8))

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

model = CryptFuncA(1024, 1024, n_layers=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
loss_fn = nn.BCEWithLogitsLoss()

losses = []
avg_acc = []
accuracies = [[] for _ in range(BITS)]
correct_bits = []

scaler = torch.amp.GradScaler(device)

for step_index in (pbar:=tqdm.trange(STEPS)):
    x = np.random.randint(2**BITS - 1, dtype=np.uint32, size=(BATCH_SIZE,))
    y = pseudo_random2(x)

    x = torch.from_numpy(np.unpackbits(x.view(np.uint8)).reshape(-1, BITS)).to(dtype=torch.float32, device=device)
    y = torch.from_numpy(np.unpackbits(y.view(np.uint8)).reshape(-1, BITS)).to(dtype=torch.float32, device=device)
    
    with torch.autocast(device_type=device, dtype=torch.float16):
        y_pred = model(x)
        loss = loss_fn(y_pred.flatten(), y.flatten())
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    correct_bits.append(((y_pred > 0) == (y > 0)).cpu().float())
    
    if step_index % UPDATE_EVERY == 0:
        correct_bits = torch.cat(correct_bits, 0)
        
        for i,v in enumerate(correct_bits.mean(0).tolist()):
            accuracies[i].append(v)
            
        avg_acc.append(correct_bits.mean().item())
        
        pbar.set_description(f"Loss: {loss.item():.3f}, Correct bits: {avg_acc[-1]:.3f}", refresh=False)
        correct_bits = []
        
        if avg_acc[-1] > EARLY_STOPPING_THRESH:
            break
        
torch.save(model.state_dict(), "checkpoint.pt")