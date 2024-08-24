import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import AdamW

from tqdm import tqdm
import argparse
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training. (DEFAULT = 64)")
    parser.add_argument('--g_lr', type=float, default=0.0001, help="Initial learning rate for the Generator. (DEFAULT = 0.0001)")
    parser.add_argument('--d_lr', type=float, default=0.0001, help="Initial learning rate for the Discriminator. (DEFAULT = 0.0001)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train. (DEFAULT = 100)")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of random values to sample. (DEFAULT = 100)")
    parser.add_argument('--seed', type=int, default=9, help="Seed for reproducibility. (DEFAULT = 9)")  # Add seed argument
    return parser.parse_args()

class Generator(nn.Module):
    def __init__(self, sample_size):
        super().__init__()

        self.sample_size = sample_size

        self.linear1 = nn.Linear(self.sample_size, 128)
        self.leakyrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_size):
        z = torch.randn(batch_size, self.sample_size)

        out = self.linear1(z)
        out = self.leakyrelu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        generated_images = out.reshape(batch_size, 1, 28, 28)

        return generated_images
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 128)
        self.leakyrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, 1)

    def forward(self, images, targets):
        out = self.linear1(images.reshape(-1, 784))
        out = self.leakyrelu(out)
        out = self.linear2(out)

        loss = nn.functional.binary_cross_entropy_with_logits(out, targets)

        return loss
    
def save_image_grid(epoch, images, ncol):
    image_grid = make_grid(images, ncol)     # Images in a grid
    image_grid = image_grid.permute(1, 2, 0) # Move channel last
    image_grid = image_grid.cpu().numpy()    # To Numpy

    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'generated_{epoch:03d}.jpg')
    plt.close()

def main():
    args = parse_args()

    real_targets = torch.ones(args.batch_size, 1)
    fake_targets = torch.zeros(args.batch_size, 1)

    generator = Generator(sample_size=args.sample_size)
    discriminator = Discriminator()

    g_optim = AdamW(generator.parameters(), lr=args.g_lr)
    d_optim = AdamW(discriminator.parameters(), lr=args.d_lr)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train = True, download = True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    for epoch in range(args.epochs):
        d_losses = []
        g_losses = []

        for images, labels in tqdm(dataloader):
            discriminator.train()
            d_loss = discriminator(images, real_targets)

            generator.eval()
            with torch.no_grad():
                generated_images = generator(args.batch_size)

            d_loss += discriminator(generated_images, fake_targets)

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            generator.train()
            generated_images = generator(args.batch_size)

            discriminator.eval()
            g_loss = discriminator(generated_images, real_targets)

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        print(epoch, np.mean(d_losses), np.mean(g_losses))

        save_image_grid(epoch, generator(args.batch_size), ncol=8)

if __name__ == "__main__":
    main()
