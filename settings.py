####################
# Program settings #
####################

setting = {

    # GPU and GPU
    "num_workers": 1,                       # Number of workers for dataloader, Default: 2, HERE: The less the better???
    "num_gpu": 1,                           # Number of GPUs available. Use 0 for CPU mode, Default: 1  
 
    # Training parameters
    "batch_size": 32,                       # Strongly depends on the number of filters! around 32      
    "num_epochs": 300, 

    # Learning rate (scheduler) parameters
    "use_lr_scheduler": True,
    "use_cosine_ann": True,                 # If use_cosine_ann and use_cosine_ann_wr are both set to True, cosine_ann will be taken
    "use_cosine_ann_wr": False,             # If use_cosine_ann and use_cosine_ann_wr are both set to False, it is the same as setting use_lr_scheduler to False

    # GENERATOR:
    "gen_learning_rate": 0.00005,            # 2e-4 = 0.0002
    # CosineAnnealingWarmRestarts:
    "gen_lrs_eta_min": 0.000005,            # Minimum LR to avoid stalling, 0.000001 = 1e-6
    "gen_lrs_t_0": 10,                      # Epochs in the first cycle. Smaller values = more frequent restarts
    "gen_lrs_t_mult": 2,                    # Cycle length grows exponentially (T_0, T_0*2, T_0*4, ...). Set to 1 for fixed-length cycles

    # CRITIC:
    "crit_learning_rate": 0.0001,           # 0.00002
    # CosineAnnealingWarmRestarts:
    "crit_lrs_eta_min": 0.00001,            # 0.00001 = 1e-5 (10% of LR)
    "crit_lrs_t_0": 10,                     # 10
    "crit_lrs_t_mult": 2,                   # 2

    # Input/output dims
    "img_channels": 1,                      # Number of channels in the training images. For color images this is 3  
    "img_size": 512,                        # Spatial size of training images. All images will be resized to this size using a transformer

    # Conv parameters
    "conv_kernel_size": 3,                  # Option 1: 5, 2, 2, (1) -> Seems to work best
    "conv_stride": 2,                       # Option 2: 3, 2, 1, (1) -> Most common one! Should be used
    "conv_padding": 1,                      # Option 3: 4, 2, 1, (0) -> Not so common. Causes asymetries
    "conv_out_padding": 1, 
    "latent_vector_size": 512,              # Size of z latent vector (i.e. size of generator input)
    "size_min_feature_maps": 4,             # 4 = 4x4 pixels is the minimum size, from where an image is scaled up 
    "gen_chan_per_layer": [512, 256, 128, 64, 32, 16, 8],
    # Alternatives: 
    # [1024, 512, 256, 128, 64, 32, 16]
    # [512, 512, 256, 256, 128, 128, 64]
    # [512, 256, 256, 128, 128, 64, 64]
    # [512, 256, 128, 64, 32, 16, 8] 
    # [512, 256, 128, 128, 64, 64, 32]
    # [384, 192, 96, 96, 48, 48, 24]
    # The Critic should have 1.2–2.5x the generator’s parameters
    "crit_chan_per_layer": [64, 128, 256, 256, 512, 512, 512],   # before: [64, 64, 128, 128, 256, 256, 512]
    # Alternatives: 
    # [16, 32, 64, 128, 256, 512, 1024]
    # [64, 128, 128, 256, 256, 512, 512]
    # [64, 64, 128, 128, 256, 256, 512]
    # [8, 16, 32, 64, 128, 256, 512]
    # [32, 64, 64, 128, 128, 256, 512]
    # [32, 64, 128, 256, 256, 512, 512]

    #################
    # AUGMENTATIONS #
    #################

    # Use augmentations
    "train_use_augment": True, 

    # INTENSITY AUGMENTATIONS:
    "aug_intense_prob": 0.5,
    "aug_brightness": 0.2,
    "aug_contrast": 0.2,
    "aug_saturation": 0.2, # only for RGB images
    # Gamma correction
    # Gamma = 1: No change. The image looks "natural" (linear brightness)
    # Gamma < 1 (e.g., 0.5): Dark areas get brighter, bright areas stay mostly the same
    # Gamma > 1 (e.g., 2.0): Bright areas get darker, dark areas stay mostly the same
    "aug_gamma_prob": 0.4,
    "aug_gamma_min": 0.7,
    "aug_gamma_max": 1.3,
    # OPTICAL AUGMENTATIONS:
    # Gaussian Blur Parameters
    # Probability
    "aug_gauss_prob": 0.3,
    # Kernel size
    "aug_gauss_kernel_size": 5,
    # Sigma: ontrols the "spread" of the blur (how intense/smooth it is)
    "aug_gauss_sigma_min": 0.1,
    "aug_gauss_sigma_max": 0.5,
    # Poisson noise
    # Probability
    "aug_poiss_prob": 0.4, 
    # Controls how much the noise depends on image brightness
    # Suggested range: 0.01-0.1 (higher = more noise)
    "aug_poiss_scaling": 0.05,  # 5% of pixel value
    # Noise Strength: Final noise intensity multiplier
    "aug_poiss_noise_strength": 0.1,

    # Misc
    "gen_adam_beta_1": 0.5,                 # Beta 1 and 2 parameter for ADAM optimizer
    "gen_adam_beta_2": 0.9,
    "crit_adam_beta_1": 0.0,                # NO momentum for critic!
    "crit_adam_beta_2": 0.9,
    "lrelu_alpha": 0.2,                     # Alpha value of leaky ReLU activation function
    "crit_dropout": 0.2,                    # Dropout for critic

    # Noise injection for critic training
    "use_noise_injection": True,            # Toggle noise injection
    "max_noise_std": 0.05,                  # Initial noise level
    "min_noise_std": 0.01,                  # Minimum noise level

    # Sample and plot generation
    # Training samples:
    "generate_samples": True,
    "num_sample_images": 9,                 # Number of images, which will be saved during training as examples. Can be less as the last batch can contain less images!
    "num_rows_sample_images": 3,            # Number of rows for sample image display
    "generate_samples_epochs": 1,           # Save sample images every x epochs in samples folder
    # Checkpoints:
    "generate_checkpoints": True,
    "generate_checkpoints_epochs": 25,      # Save generator every x epochs in checkpoints folder
    # Metrics plot
    "generate_plots": True,
    "generate_plot_epochs": 10,             # Save loss plot every x epochs

    # WGAN
    # Training critic more that generator
    "num_crit_training": 5,                 # 2
    "gradient_penalty_weight": 10,          # 10

    # Paths 
    "pth_data_root": "data/",               # Root directory for all datasets
    "pth_data": "data/fibroblasts/",        # Root directory for the current dataset
    "pth_samples": "samples/",              # Directory for generated samples during training
    "pth_plots": "plots/",                  # Directory for saving plots
    "pth_checkpoints": "checkpoints/",      # Directory for saving checkpoints
}