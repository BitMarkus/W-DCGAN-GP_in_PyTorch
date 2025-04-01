# Wasserstein DCGAN with Gradient Penalty (WGAN-GP)

A PyTorch implementation of a Deep Convolutional GAN (DCGAN) with Wasserstein loss and Gradient Penalty, optimized for high-resolution grayscale/RGB image generation.

## Key Features
- **Wasserstein Loss (WGAN)**: Stable training via Earth-Mover distance
- **Gradient Penalty (GP)**: Replaces weight clipping for better convergence
- **512×512 px Support**: Designed for high-res grayscale/RGB images
- **Training Monitoring**: Tracks losses, gradients, and sample quality

## Why WGAN-GP?
Traditional GANs suffer from mode collapse and unstable training. This implementation addresses these issues by:
- **Wasserstein Loss**: Provides meaningful loss metrics correlating with generation quality.
- **Gradient Penalty**: Ensures the critic (discriminator) stays within Lipschitz constraints, avoiding vanishing/exploding gradients.
- **Architecture**: Pure convolutional networks (no fully connected layers) for spatial feature preservation.

## Reference: 
Improved Training of Wasserstein GANs (Gulrajani et al., 2017).

## Quick Start
1. **Prepare Data:**<br> 
Place images in data/your_dataset/ (subfolder required, e.g., data/fibroblasts/wt/)
2. **Configure Settings:**<br> 
Edit settings.py

## Directory Structure
All necessary program folders will be created once the program is started for the first time.
<pre>
project-root/
├── checkpoints/       <em># Generator snapshots</em>
├── data/              <em># Training images</em>
│   └── fibroblasts/   <em># Example dataset</em>
│       └── wt/        <em># Image subfolder</em>
├── plots/             <em># Loss curves</em>
└── samples/           <em># Generated images</em>
</pre>

## Technical Notes
- **Gradient Penalty**: λ=10 (default) enforces 1-Lipschitz constraint. Gradient penalty should stay between 0-10, better between 0-1.
- **LR Scheduling**: Cosine annealing with restarts (T_0=10 epochs)
- **Monitor**: Gradient norms should stabilize near 1.0

## Training and image generation
Currently there is no console menu included. Uncomment the lines "train.train()" or "samples(device)" for training or image generation based on an existing checkpoint.








