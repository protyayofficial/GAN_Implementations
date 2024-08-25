import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import Adam

from tqdm import tqdm
import argparse
import random
import shutil
from PIL import Image, ImageSequence

def set_seed(seed):
    """
    Set the seed for random number generators to ensure reproducibility.
    
    Parameters:
    - seed (int): Seed value to use for random number generation.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parse command-line arguments for training the GAN model.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the GAN model")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--g_lr', type=float, default=0.0002, help="Learning rate for the Generator.")
    parser.add_argument('--d_lr', type=float, default=0.0002, help="Learning rate for the Discriminator.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument('--img_size', type=int, default=64, help="Size of the Image.")
    parser.add_argument('--features_d', type=int, default=64, help="Features for the Discriminator.")
    parser.add_argument('--features_g', type=int, default=64, help="Features for the Generator.")
    parser.add_argument('--img_channels', type=int, default=1, help="Number of channels in the input image.")
    parser.add_argument('--sample_size', type=int, default=100, help="Size of the random noise vector (latent space).")
    parser.add_argument('--seed', type=int, default=9, help="Seed for reproducibility.")
    parser.add_argument('--output_dir', type=str, default='results/DCGAN', help="Directory to save the generated GIF and images.")
    return parser.parse_args()

class Generator(nn.Module):
    """
    Generator model for the GAN. Takes random noise as input and generates images.
    """
    def __init__(self, noise, img_channels, features_g):
        """
        Initialize the Generator model.

        Parameters:
        - sample_size (int): Size of the input noise vector.
        """
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self._block(in_channels=noise, out_channels=features_g * 16, kernel_size=4, stride=1, padding=0),
            self._block(in_channels=features_g * 16, out_channels= features_g * 8, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_g * 8, out_channels= features_g * 4, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_g * 4, out_channels= features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=features_g * 2, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, z):
        """
        Forward pass through the Generator.

        Parameters:
        - z (Tensor): Input noise vector.

        Returns:
        - Tensor: Generated image.
        """
        return self.gen(z)

class Discriminator(nn.Module):
    """
    Discriminator model for the GAN. Takes images as input and outputs the probability of them being real.
    """
    def __init__(self, img_channels, features_d):
        """
        Initialize the Discriminator model.
        """
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            self._block(in_channels=features_d, out_channels=features_d * 2, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_d * 2, out_channels=features_d * 4, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_d * 4, out_channels=features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=features_d * 8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        

    def forward(self, x):
        """
        Forward pass through the Discriminator.

        Parameters:
        - x (Tensor): Input image.

        Returns:
        - Tensor: Probability of the image being real.
        """
        return self.dis(x)
    
def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

def save_image_grid(epoch, images, ncol):
    """
    Save a grid of generated images to a file in the 'results' folder.

    Parameters:
    - epoch (int): Current epoch number.
    - images (Tensor): Generated images.
    - ncol (int): Number of columns in the image grid.
    """
    # Create the results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate the image grid
    image_grid = make_grid(images, ncol)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()
    
    # Plot and save the image grid
    plt.imshow(image_grid, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'results/DCGAN_{epoch:03d}.jpg')
    plt.close()

def create_gif_and_move(output_dir):
    """
    Create a GIF from the saved images and move all images and GIF to the specified output directory.
    
    Parameters:
    - output_dir (str): Directory where the GIF and images should be moved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list to hold the images for the GIF
    images = []
    
    # Iterate over the image files in the 'results' directory
    for filename in sorted(os.listdir('results')):
        # Only process .jpg or .png files
        if 'DCGAN' in filename and filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join('results', filename))
            images.append(img)
    
    # Create the GIF and save it in the output directory
    gif_path = os.path.join(output_dir, 'DCGAN_MNIST.gif')
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
    
    # Now move all the image files and the GIF into the output directory
    for img_file in os.listdir('results'):
        # Ensure we're only moving files and not directories or invalid paths
        src_path = os.path.join('results', img_file)
        dest_path = os.path.join(output_dir, img_file)
        
        if os.path.isfile(src_path):  # Check if it's a file, not a directory
            shutil.move(src_path, dest_path)

def main():
    """
    Main function to set up and train the GAN model.
    """
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create models
    generator = Generator(noise=args.sample_size, img_channels=args.img_channels, features_g=args.features_g).to(device)
    discriminator = Discriminator(img_channels=args.img_channels, features_d=args.features_d).to(device)

    init_weights(generator)
    init_weights(discriminator)

    # Optimizers
    g_optim = Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optim = Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()

    # Transform and dataset
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(args.img_channels)], 
            [0.5 for _ in range(args.img_channels)]
        )
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        d_losses = []
        g_losses = []

        for real_imgs, _ in tqdm(dataloader):
            real_imgs = real_imgs.to(device)

            # =======================
            # Train Discriminator
            # =======================
            discriminator.train()

            # Real images
            d_loss_real = discriminator(real_imgs).reshape(-1)
            real_loss = criterion(d_loss_real, torch.ones_like(d_loss_real))

            # Generated images
            noise = torch.randn(args.batch_size, args.sample_size, 1, 1).to(device)
            fake_images = generator(noise)
            d_loss_fake = discriminator(fake_images.detach()).reshape(-1)
            fake_loss = criterion(d_loss_fake, torch.zeros_like(d_loss_fake))

            # Total Discriminator loss
            d_loss = (real_loss + fake_loss) / 2.
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # =======================
            # Train Generator
            # =======================
            generator.train()
            g_loss = discriminator(fake_images).reshape(-1)  # Generated images again
            g_loss = criterion(g_loss, torch.ones_like(g_loss))

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        print(f"Epoch [{epoch + 1}/{args.epochs}], D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")

        # Save generated images every epoch
        generator.eval()
        save_image_grid(epoch, fake_images, ncol=8)

    create_gif_and_move(args.output_dir)

if __name__ == "__main__":
    main()
