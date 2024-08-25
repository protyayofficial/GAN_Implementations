import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import Adam

from tqdm import tqdm
import argparse
import random
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
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--g_lr', type=float, default=0.0001, help="Learning rate for the Generator.")
    parser.add_argument('--d_lr', type=float, default=0.0001, help="Learning rate for the Discriminator.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument('--sample_size', type=int, default=100, help="Size of the random noise vector (latent space).")
    parser.add_argument('--seed', type=int, default=9, help="Seed for reproducibility.")
    parser.add_argument('--output_dir', type=str, default='results/AdversarialNets', help="Directory to save the generated GIF and images.")
    return parser.parse_args()

class Generator(nn.Module):
    """
    Generator model for the GAN. Takes random noise as input and generates images.
    """
    def __init__(self, sample_size):
        """
        Initialize the Generator model.

        Parameters:
        - sample_size (int): Size of the input noise vector.
        """
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(sample_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 784)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        """
        Forward pass through the Generator.

        Parameters:
        - z (Tensor): Input noise vector.

        Returns:
        - Tensor: Generated image.
        """
        z = self.leakyrelu(self.bn1(self.fc1(z)))
        z = self.leakyrelu(self.bn2(self.fc2(z)))
        z = self.leakyrelu(self.bn3(self.fc3(z)))
        z = self.tanh(self.fc4(z))
        return z.view(-1, 1, 28, 28)  # Reshape to match image dimensions (1, 28, 28)

class Discriminator(nn.Module):
    """
    Discriminator model for the GAN. Takes images as input and outputs the probability of them being real.
    """
    def __init__(self):
        """
        Initialize the Discriminator model.
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the Discriminator.

        Parameters:
        - x (Tensor): Input image.

        Returns:
        - Tensor: Probability of the image being real.
        """
        x = x.view(-1, 784)  # Flatten the images
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # Final layer gives a probability
        return x

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
    plt.savefig(f'results/AdversarialNets_{epoch:03d}.jpg')
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
        if 'AdversarialNets' in filename and filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join('results', filename))
            images.append(img)
    
    # Create the GIF and save it in the output directory
    gif_path = os.path.join(output_dir, 'AdversarialNets_MNIST.gif')
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

    # Create real and fake labels with label smoothing
    real_label_value = 1.0
    fake_label_value = 0.0
    
    # Create models
    generator = Generator(sample_size=args.sample_size).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optim = Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optim = Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()

    # Transform and dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(args.epochs):
        d_losses = []
        g_losses = []

        for images, _ in tqdm(dataloader):
            images = images.to(device)
            batch_size = images.size(0)

            # Create real and fake labels
            real_labels = torch.full((batch_size, 1), real_label_value).to(device)
            fake_labels = torch.full((batch_size, 1), fake_label_value).to(device)

            # =======================
            # Train Discriminator
            # =======================
            discriminator.train()

            # Real images
            d_loss_real = discriminator(images)
            real_loss = criterion(d_loss_real, real_labels)

            # Generated images
            noise = torch.randn(batch_size, args.sample_size).to(device)
            fake_images = generator(noise)
            d_loss_fake = discriminator(fake_images.detach())
            fake_loss = criterion(d_loss_fake, fake_labels)

            # Total Discriminator loss
            d_loss = real_loss + fake_loss
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # =======================
            # Train Generator
            # =======================
            generator.train()
            g_loss = discriminator(fake_images)  # Generated images again
            g_loss = criterion(g_loss, real_labels)

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
