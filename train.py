import torch
import numpy as np
import tqdm
import argparse
import matplotlib.pyplot as plt

import model
import functions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Cryptographic functions parser"
    )

    parser.add_argument("--bits", type=int, default=32, help="Number of bits to use")
    parser.add_argument("--depth", type=int, default=3, help="Number of rounds to use")

    parser.add_argument("--n_layers", type=int, default=10, help="Number of layers in the NN to use")
    parser.add_argument("--dim", type=int, default=1024, help="Inner size of the NN to use")

    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate to use")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size to use")
    parser.add_argument("--grad_acc", type=int, default=1, help="How many gradients to accumulate before running an optimization step")
    
    parser.add_argument("--steps", type=int, default=30000, help="Number of steps during training")
    parser.add_argument("--update_freq", type=int, default=10, help="How often to update the progress bar")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--threshold",type=float, default=.999, help="Threshold to use for early stopping")
    parser.add_argument("--ckpt_name", type=str, default="checkpoint.pt", help="Filename for the network's weights")

    args = parser.parse_args()
    print(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available") 
    
    crypt = functions.BasicCrypto(args.depth, args.seed, args.bits)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = model.CryptFuncA(args.dim, args.bits, n_layers=args.n_layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    avg_acc = []
    accuracies = [[] for _ in range(args.bits)]
    correct_bits = []

    # scaler = torch.amp.GradScaler(args.device)

    for step_index in (pbar:=tqdm.trange(args.steps)):
        x = np.random.randint(2**args.bits, dtype=crypt.type, size=(args.batch_size,))
        y = crypt.sample(x)

        x = torch.from_numpy(np.unpackbits(x.view(np.uint8)).reshape(-1, args.bits)).to(dtype=torch.float32, device=args.device)
        y = torch.from_numpy(np.unpackbits(y.view(np.uint8)).reshape(-1, args.bits)).to(dtype=torch.float32, device=args.device)
        
        # with torch.autocast(device_type=args.device, dtype=torch.float16):
        y_pred = model(x)
        loss = torch.relu((1 - y*2) * y_pred + 1).mean() / args.grad_acc
            
        loss.backward()
        
        if step_index % args.grad_acc == 0:
            optimizer.step()
            # scaler.update()
            optimizer.zero_grad()

        correct_bits.append(((y_pred > 0) == (y > 0)).cpu().float())
        
        if step_index % args.update_freq == 0:
            correct_bits = torch.cat(correct_bits, 0)
            
            for i,v in enumerate(correct_bits.mean(0).tolist()):
                accuracies[i].append(v)
                
            avg_acc.append(correct_bits.mean().item())
            
            pbar.set_description(f"Loss: {loss.item():.3f}, Correct bits: {avg_acc[-1]:.3f}", refresh=False)
            correct_bits = []
            
            if avg_acc[-1] > args.threshold:
                break
            
    torch.save(model.state_dict(), args.ckpt_name)