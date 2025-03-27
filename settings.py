####################
# Program settings #
####################

setting = {

    # GPU and GPU
    "num_workers": 1,                       # Number of workers for dataloader, Default: 2, HERE: The less the better???
    "num_gpu": 1,                           # Number of GPUs available. Use 0 for CPU mode, Default: 1  
 
    # Training parameters
    "batch_size": 40,                       # Strongly depends on the number of filters!      
    "num_epochs": 500, 

    # Learning rate (scheduler) parameters for CosineAnnealingWarmRestarts
    # Generator
    "gen_learning_rate": 0.0001,            # 0.0001
    "gen_lrs_eta_min": 0.000001,            # Minimum LR to avoid stalling, 1e-6
    "gen_lrs_t_0": 10,                      # Epochs in the first cycle. Smaller values = more frequent restarts
    "gen_lrs_t_mult": 2,                    # Cycle length grows exponentially (T_0, T_0*2, T_0*4, ...). Set to 1 for fixed-length cycles
    # Critic
    "crit_learning_rate": 0.00005,          # 0.00005
    "crit_lrs_eta_min": 0.000001,           # 1e-6
    "crit_lrs_t_0": 10,                     # 10
    "crit_lrs_t_mult": 2,                   # 2

    # Input/output dims
    "img_channels": 1,                      # Number of channels in the training images. For color images this is 3  
    "img_size": 512,                        # Spatial size of training images. All images will be resized to this size using a transformer

    # Conv parameters
    "conv_kernel_size": 5,                  # Option 1: 5, 2, 2, (1) -> Seems to work best
    "conv_stride": 2,                       # Option 2: 3, 2, 1, (1) -> Most common one! Should be used
    "conv_padding": 2,                      # Option 3: 4, 2, 1, (0) -> Not so common. Causes asymetries
    "conv_out_padding": 1, 
    "latent_vector_size": 512,              # Size of z latent vector (i.e. size of generator input)
    "size_min_feature_maps": 4,             # 4 = 4x4 pixels is the minimum size, from where an image is scaled up 
    "gen_chan_per_layer": [512, 256, 256, 128, 128, 64, 64],
    # Alternatives: [512, 256, 128, 64, 32, 16, 8][1024, 512, 256, 128, 64, 32, 16][512, 512, 256, 256, 128, 128, 64] 
    "crit_chan_per_layer": [64, 64, 128, 128, 256, 256, 512],  
    # Alternatives: [8, 16, 32, 64, 128, 256, 512][16, 32, 64, 128, 256, 512, 1024][64, 128, 128, 256, 256, 512, 512]

    # Misc
    "adam_beta_1": 0.0,                     # Beta 1 and 2 parameter for ADAM optimizer
    "adam_beta_2": 0.9,
    "lrelu_alpha": 0.2,                     # Alpha value of leaky ReLU activation function

    # Sample and plot generation
    "generate_samples": True,
    "num_samples": 2,                       # Number of images, which will be saved during training as examples
    "generate_samples_epochs": 1,           # Save sample images every x epochs in samples folder
    "generate_checkpoints_epochs": 50,      # Save generator every x epochs in checkpoints folder
    "generate_plot_epochs": 10,             # Save loss plot every x epochs

    # WGAN
    # Training critic more that generator
    "num_crit_training": 1,                 # 2
    "gradient_penalty_weight": 10,          # 10

    # Paths
    "pth_data_root": "data/",               # Root directory for all datasets
    "pth_data": "data/fibroblasts/",        # Root directory for the current dataset
    "pth_samples": "samples/",              # Directory for generated samples during training
    "pth_plots": "plots/",                  # Directory for saving plots
    "pth_checkpoints": "checkpoints/",      # Directory for saving checkpoints
}