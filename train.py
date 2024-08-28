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

    parser.add_argument("--bytes", type=int, default=8, help="Number of bytes to use")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds to use")

    parser.add_argument("--n_layers", type=int, default=10, help="Number of layers in the NN to use")
    parser.add_argument("--dim", type=int, default=1024, help="Inner size of the NN to use")

    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate to use")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size to use")
    parser.add_argument("--grad_acc", type=int, default=1, help="How many gradients to accumulate before running an optimization step")
    
    parser.add_argument("--sequential", type=int, default=0, help="Number of bits to use")
    parser.add_argument("--steps", type=int, default=30000, help="Number of steps during training")
    parser.add_argument("--update_freq", type=int, default=10, help="How often to update the progress bar")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    parser.add_argument("--cuda", action="store_true", help="If use CUDA")
    parser.add_argument("--threshold",type=float, default=.999, help="Threshold to use for early stopping")
    parser.add_argument("--ckpt_name", type=str, default="checkpoint.pt", help="Filename for the network's weights")
    parser.add_argument("--crypto_func", type=str, default="des", help="Which cryptographic function to use")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the results")

    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    
    match args.crypto_func:
        case "aes128":
            print("using AES128")
            crypt = functions.AES(args.rounds, args.seed, args.bytes, 16)
        case "aes192":
            print("using AES192")
            crypt = functions.AES(args.rounds, args.seed, args.bytes, 24)
        case "aes256":
            print("using AES256")
            crypt = functions.AES(args.rounds, args.seed, args.bytes, 32)
        case "des":
            print("using DES")
            crypt = functions.DES(args.rounds, args.seed, args.bytes)
        case "base":
            print("using base")
            crypt = functions.BasicCrypto(args.rounds, args.seed, args.bytes)
        case _:
            raise ValueError("Cryptographic function was not correctly defined")


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = model.CryptFuncA(args.dim, args.bytes * 8, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    avg_acc = []
    accuracies = [[] for _ in range(args.bytes * 8)]
    correct_bits = []

    # scaler = torch.amp.GradScaler(args.device)

    for step_index in (pbar:=tqdm.trange(args.steps)):
        if args.sequential > 0:
            x = np.random.randint(256, dtype=np.uint8, size=(args.batch_size//args.sequential, crypt.nbytes))
            x = x[:, None].repeat(args.sequential, 1)
            randtake = np.random.randint(crypt.nbytes, size=(x.shape[0],))
            x[np.arange(x.shape[0]), :, randtake] += np.arange(args.sequential, dtype=np.uint8)[None]
            x = x.reshape(-1, crypt.nbytes)
        else:
            x = np.random.randint(256, dtype=np.uint8, size=(args.batch_size, crypt.nbytes))

        y = crypt.sample(x)

        x = torch.from_numpy(np.unpackbits(x).reshape(-1, args.bytes * 8)).to(dtype=torch.float32, device=device)
        y = torch.from_numpy(np.unpackbits(y).reshape(-1, args.bytes * 8)).to(dtype=torch.float32, device=device)
        
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

    if args.plot:
        plt.plot(avg_acc)
        plt.title("Average accuracy")
        plt.show()

        cmap = plt.get_cmap('rainbow')

        K = 128
        N = args.bytes * K + args.bytes * 8

        colors = cmap(np.linspace(0,1,N))

        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(args.bytes * 8 ):
            ax.plot(accuracies[i], label=f"bit{i:02d}", c=colors[i + K * (i // 8)])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=4)
        plt.title("Bits' accuracies")
        plt.show()

        print(avg_acc)
        print(accuracies)