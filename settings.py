####################
# Program settings #
####################

setting = {

    ############
    # TRAINING #
    ############

    # GPU and GPU
    "num_workers": 2,                   # Number of workers for dataloader, Default: 2
    "num_gpu": 1,                       # Number of GPUs available. Use 0 for CPU mode, Default: 1  
 
    # Hyperparameter
    "batch_size": 32,                 
    "num_epochs": 5000,              

    # Generator
    "gen_learning_rate": 0.0002,
    "gen_dropout": 0.25,

    # Discriminator
    "disc_learning_rate": 0.0001,
    "disc_dropout": 0.25,

    # Input/output dims
    "img_channels": 1,                  # Number of channels in the training images. For color images this is 3  
    "img_size": 512,                    # Spatial size of training images. All images will be resized to this size using a transformer
    "num_classes": 1,  

    # Misc
    "opt_beta_1": 0.5,                  # Beta 1 parameter for ADAM optimizer, Default for GAN: 0.5
    "lrelu_alpha": 0.2,                 # Alpha value of leaky ReLU activation function
    "latent_vector_size": 2048,         # Size of z latent vector (i.e. size of generator input)

    # Sample and plot generation
    "generate_samples": True,
    "num_samples": 2,                   # Number of images, which will be saved during training as examples
    "generate_samples_epochs": 10,      # Save sample images every x epochs in samples folder
    "generate_checkpoints_epochs": 1250,  # Save generator every x epochs in checkpoints folder
    "generate_plot_epochs": 500,         # Save loss plot every x epochs

    # Paths
    "pth_data": "data/fibroblasts/",    # Root directory for dataset
    "pth_samples": "samples/",          # Directory for generated samples during training
    "pth_plots": "plots/",              # Directory for saving plots
    "pth_checkpoints": "checkpoints/",  # Directory for saving checkpoints
}