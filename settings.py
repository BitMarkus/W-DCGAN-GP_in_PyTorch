####################
# Program settings #
####################

setting = {

    # GPU and GPU
    "num_workers": 6,                       # Number of workers for dataloader, Default: 2
    "num_gpu": 1,                           # Number of GPUs available. Use 0 for CPU mode, Default: 1  
 
    # Hyperparameter
    "batch_size": 300,                       # 300 for W-DCGAN-GC              
    "num_epochs": 1000,              

    # Generator
    "gen_learning_rate": 0.00005,            # 0.0002 for vanilla GAN, 0.00005 = 5e-5 for WGAN
    "gen_dropout": 0.0,

    # Discriminator
    "disc_learning_rate": 0.00005,           # 0.0002 for vanilla GAN, 0.00005 = 5e-5 for WGAN
    "disc_dropout": 0.0,

    # Input/output dims
    "img_channels": 1,                      # Number of channels in the training images. For color images this is 3  
    "img_size": 512,                        # Spatial size of training images. All images will be resized to this size using a transformer

    # Conv parameters
    "conv_kernel_size": 4,                  # Option 1: 5, 2, 2, (1) -> Exploding losses after epoch 500, image quality better than ks 3
    "conv_stride": 2,                       # Option 2: 3, 2, 1, (1) -> Seems to only pick up local details, but not the bigger picture
    "conv_padding": 1,                      # Option 3: 4, 2, 1, (0) -> Exploding losses after epoch 1200, image quality worse than ks 5
    "conv_out_padding": 0,  

    # Misc
    "opt_beta_1": 0.5,                      # Beta 1 parameter for ADAM optimizer, Default for GAN: 0.5
    "lrelu_alpha": 0.2,                     # Alpha value of leaky ReLU activation function
    "latent_vector_size": 512,              # Size of z latent vector (i.e. size of generator input)
    "num_max_feature_maps": 512, 
    "size_min_feature_maps": 4,             # 4 = 4x4 pixels is the minimum size, from where an image is scaled up

    # Sample and plot generation
    "generate_samples": True,
    "num_samples": 2,                       # Number of images, which will be saved during training as examples
    "generate_samples_epochs": 1,           # Save sample images every x epochs in samples folder
    "generate_checkpoints_epochs": 100,     # Save generator every x epochs in checkpoints folder
    "generate_plot_epochs": 25,             # Save loss plot every x epochs

    # Training generator several times per epoch
    "max_gen_loss_1": 0.8,
    "max_gen_loss_2": 1.0,
    "max_gen_loss_3": 1.3,
    "max_gen_loss_4": 1.6,
    "max_gen_loss_5": 2.0,
    "max_gen_loss_6": 2.5,
    "max_gen_loss_7": 3.0,
    "max_gen_loss_8": 3.5,
    "max_gen_loss_9": 4.0,

    # WGAN
    # Training discriminator more that generator
    "num_disc_training": 5,

    # Paths
    "pth_data_root": "data/",               # Root directory for all datasets
    "pth_data": "data/fibroblasts/",      # Root directory for the current dataset
    # "pth_data": "data/faces/",              # Root directory for the current dataset
    "pth_samples": "samples/",              # Directory for generated samples during training
    "pth_plots": "plots/",                  # Directory for saving plots
    "pth_checkpoints": "checkpoints/",      # Directory for saving checkpoints
}