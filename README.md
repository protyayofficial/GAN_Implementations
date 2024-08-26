# GAN Architectures Implementation

Welcome to the GAN Architectures Implementation repository! This project provides implementations of various Generative Adversarial Networks (GANs) using PyTorch. GANs are a class of deep learning models that consist of a generator and a discriminator network that compete with each other, resulting in high-quality generated data.

## Project Overview

This repository includes implementations of several GAN architectures. Each architecture has been implemented in a modular and extensible manner, allowing for easy experimentation and comparison.

## Included Architectures

- **[Vanilla GAN](./models/AdversarialNets.py)**: A basic GAN model as introduced by Goodfellow et al. in their original paper.
- **[Deep Convolutional GAN (DCGAN)](./models/DCGAN.py)**: An improvement over the vanilla GAN by using convolutional layers for both generators and discriminators, which helps in generating more realistic images.
<!-- - **Conditional GAN (cGAN)**: A GAN variant where the generator and discriminator receive additional information (e.g., class labels) to condition the generation process.
- **Wasserstein GAN (WGAN)**: An alternative to the traditional GANs that uses the Wasserstein distance to improve training stability and the quality of generated samples.
- **Wasserstein GAN with Gradient Penalty (WGAN-GP)**: An enhanced version of WGAN that includes a gradient penalty term to further stabilize the training process.
- **Least Squares GAN (LSGAN)**: A GAN variant that uses least squares loss instead of binary cross-entropy, aiming to address some of the issues with vanishing gradients. -->
- More architectures to be included soon...

## Running the GAN Training Script

To streamline the process of training the GAN model, we have included a shell script, `train_gan.sh`, in the repository. This script is designed to execute the GAN training with pre-configured default settings for parameters like batch size, learning rate, number of epochs, and more.

### Usage

You can use the `train_{gan}.sh` script to start training the GAN model with a simple command. Follow these steps:

1. **Make the script executable** (if you haven't already):
    ```bash
    chmod +x scripts/train_{gan}.sh
    ```

2. **Run the script**:
    ```bash
    ./scripts/train_{gan}.sh
    ```

## Motivation
In this project, I explore fundamental concepts in computer vision by meticulously reconstructing well-known Generative Adversarial Network (GAN) models from scratch using PyTorch. This hands-on approach deepens my understanding of these intricate architectures and allows me to thoroughly examine the details of each layer and function.

As a passionate learner in the field of Computer Vision, I am fascinated by the potential of enabling machines to see, interpret, and create visual content. This capability has the power to drive innovation across various sectors, from advancing healthcare diagnostics to expanding the possibilities of digital entertainment.

My interest in Computer Vision began with a curiosity about how algorithms can analyze complex visual data and generate realistic images from nothing. This curiosity has led me to delve into different aspects of the field, including image classification, object detection, and now the captivating world of GANs. Each project has enhanced my understanding of the delicate interplay between data, algorithms, and creativity.

This project marks a significant step in my journey toward becoming a Computer Vision Researcher. It provides an opportunity to deepen my expertise in advanced generative models, sharpen my technical skills, and contribute to the ongoing research in image generation. I look forward to continuing my exploration of innovative solutions in Computer Vision and contributing to technologies that will redefine our interaction with visual data.

## Basics of GANs
**Generative Adversarial Networks (GANs)** are a class of machine learning frameworks introduced by Ian Goodfellow and his colleagues in 2014. GANs consist of two neural networks: the generator and the discriminator, which are trained simultaneously through adversarial processes. The key idea is to have these two networks compete with each other, leading to improved performance and generation of high-quality data.

## How GANs Work
- **Generator**: The generator network's goal is to create synthetic data that is indistinguishable from real data. It takes a random noise vector as input and transforms it into a data sample (e.g., an image). Initially, the generated data is not very realistic, but it improves as the generator learns to create more convincing samples.

- **Discriminator**: The discriminator network's task is to differentiate between real data (from the training set) and fake data (generated by the generator). It outputs a probability that the input data is real or fake. The discriminator is trained to maximize its accuracy in distinguishing real from fake data.

## Training Process
During training, the generator and discriminator are engaged in a two-player minimax game:

- **Generator’s Objective**: The generator aims to fool the discriminator into believing that its generated data is real. It tries to minimize the discriminator's ability to correctly classify generated data as fake.
 
- **Discriminator’s Objective**: The discriminator aims to correctly classify real and fake data. It tries to maximize its accuracy in identifying which data is real and which is generated.

The training process involves:

- **Step 1**: The discriminator is trained with real data and fake data from the generator. It learns to differentiate between the two.

- **Step 2**: The generator is trained to produce data that the discriminator classifies as real. It improves by receiving feedback from the discriminator’s judgments.

Over time, both networks improve: the generator produces increasingly realistic data, and the discriminator becomes better at identifying the nuances between real and fake data. This adversarial process continues until the generator produces data that is almost indistinguishable from real data, and the discriminator cannot reliably tell the difference.

## Applications
GANs are used in various applications, including:

- Image generation and enhancement
- Style transfer
- Data augmentation
- Super-resolution
- Text-to-image synthesis

GANs have become a foundational technology in generative models and continue to be an active area of research and development.

## Contributing
Contributions are welcome! If you would like to contribute to this repository, please fork the repository and submit a pull request with your proposed changes. Make sure to follow the coding standards and provide clear descriptions of your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


## Acknowledgements

- [PyTorch](https://pytorch.org/): An open-source machine learning library used for the implementations.
- [Generative Adversarial Nets by Goodfellow _et al._](https://arxiv.org/pdf/1406.2661) The original paper that introduced GAN to the whole world in 2014.
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks by Radford _et al._](https://arxiv.org/pdf/1511.06434) - DCGAN

## Contact
If you have any questions or suggestions, feel free to reach out!

Protyay Dey
- Email: [protyayofficial@gmail.com](mailto:protyayofficial.gmail.com)
- LinkedIn: [protyaydey](https:www.linkedin.com/in/protyaydey)
- GitHub: [protyayofficial](https://www.github.com/protyayofficial)
- Website: [protyayofficial.github.io](https://protyayofficial.github.io)