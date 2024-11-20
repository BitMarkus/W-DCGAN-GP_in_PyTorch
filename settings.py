####################
# Program settings #
####################

setting = {

    # GPU and GPU
    "num_workers": 2,                       # Number of workers for dataloader, Default: 2
    "num_gpu": 1,                           # Number of GPUs available. Use 0 for CPU mode, Default: 1  
 
    # Hyperparameter
    "batch_size": 16,                 
    "num_epochs": 2000,              

    # Generator
    "gen_learning_rate": 0.00005,            # 0.0002 for vanilla GAN, 0.00005 = 5e-5 for WGAN
    "gen_dropout": 0.0,

    # Discriminator
    "disc_learning_rate": 0.00005,           # 0.0002 for vanilla GAN, 0.00005 = 5e-5 for WGAN
    "disc_dropout": 0.0,

    # Input/output dims
    "img_channels": 1,                      # Number of channels in the training images. For color images this is 3  
    "img_size": 512,                        # Spatial size of training images. All images will be resized to this size using a transformer
    "num_classes": 1,                       # Number of classes. This will be important for creating a conditional GAN

    # Misc
    "opt_beta_1": 0.5,                      # Beta 1 parameter for ADAM optimizer, Default for GAN: 0.5
    "lrelu_alpha": 0.2,                     # Alpha value of leaky ReLU activation function
    "latent_vector_size": 1024,             # Size of z latent vector (i.e. size of generator input)

    # Sample and plot generation
    "generate_samples": True,
    "num_samples": 2,                       # Number of images, which will be saved during training as examples
    "generate_samples_epochs": 10,          # Save sample images every x epochs in samples folder
    "generate_checkpoints_epochs": 100,     # Save generator every x epochs in checkpoints folder
    "generate_plot_epochs": 100,            # Save loss plot every x epochs

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
    "pth_data": "data/fibroblasts/",        # Root directory for the current dataset
    "pth_samples": "samples/",              # Directory for generated samples during training
    "pth_plots": "plots/",                  # Directory for saving plots
    "pth_checkpoints": "checkpoints/",      # Directory for saving checkpoints
}