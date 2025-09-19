####################
# Dataset handling #
####################

import torch
from torchvision.transforms import transforms
import numpy as np
import random 
import os
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Own modules
from settings import setting

class Dataset():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Root directory for dataset
        self.dataroot = setting["pth_data"]
        # Batch size for training 
        self.batch_size = setting["batch_size"]
        # Number of channels of training images 
        self.input_channels = setting["img_channels"]
        # Image width and height for training
        self.img_size = setting["img_size"]
        # Number of workers for dataloader
        self.num_workers = setting["num_workers"]

        ################
        # Augentations #
        ################

        # Use augmentations
        self.train_use_augment = setting["train_use_augment"]
        # INTENSITY AUGMENTATIONS:
        self.intense_prob = setting["aug_intense_prob"]
        self.brightness = setting["aug_brightness"]
        self.contrast = setting["aug_contrast"]
        self.saturation = setting["aug_saturation"] # only for RGB images
        # Gamma correction
        # Gamma = 1: No change. The image looks "natural" (linear brightness)
        # Gamma < 1 (e.g., 0.5): Dark areas get brighter, bright areas stay mostly the same
        # Gamma > 1 (e.g., 2.0): Bright areas get darker, dark areas stay mostly the same
        self.gamma_prob = setting["aug_gamma_prob"]
        self.gamma_min = setting["aug_gamma_min"]
        self.gamma_max = setting["aug_gamma_max"]
        # OPTICAL AUGMENTATIONS:
        # Gaussian Blur Parameters
        # Probability
        self.gauss_prob = setting["aug_gauss_prob"]
        # Kernel size
        self.gauss_kernel_size = setting["aug_gauss_kernel_size"]
        # Sigma: ontrols the "spread" of the blur (how intense/smooth it is)
        self.gauss_sigma_min = setting["aug_gauss_sigma_min"]
        self.gauss_sigma_max = setting["aug_gauss_sigma_max"]
        # Poisson noise
        # Probability
        self.poiss_prob = setting["aug_poiss_prob"]
        # Controls how much the noise depends on image brightness
        # Suggested range: 0.01-0.1 (higher = more noise)
        self.poiss_scaling = setting["aug_poiss_scaling"]
        # Noise Strength: Final noise intensity multiplier
        self.poiss_noise_strength = setting["aug_poiss_noise_strength"]
        # Flag if dataset is already loaded or not
        self.is_dataset_loaded = False

    #############################################################################################################
    # METHODS:

    # Helper Methods
    def _gamma_correction(self, x):
        return (x + 1e-6) ** random.uniform(self.gamma_min, self.gamma_max)
    def _add_poisson_noise(self, x):
        return torch.clamp(x + torch.poisson(x * self.poiss_scaling) * self.poiss_noise_strength, 0, 1)

    # Transformer without augmentations
    def _get_transformer(self):
        # Transformer for Grayscale images
        # https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
        if(self.input_channels == 1):
            transformer = transforms.Compose([
                # transforms.Resize(self.img_size),
                # transforms.CenterCrop(self.img_size),

                # Convert to grayscale
                # Without the next line, the image will have 3 channels instead of 1
                transforms.Grayscale(num_output_channels=1), # <- Grayscale

                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(), 

                # Normalization of the image in the range [-1,1]
                # https://github.com/pytorch/vision/issues/288
                transforms.Normalize((0.5, ), (0.5, )),
            ])
            return transformer

        # Transformer for RGB images
        # https://pytorch.org/hub/pytorch_vision_resnet/
        elif(self.input_channels == 3):
            transformer = transforms.Compose([
                # transforms.Resize(self.img_size),
                # transforms.CenterCrop(self.img_size),

                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(),

                # Normalization of the image in the range [-1,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # <- RGB
            ])
            return transformer

        else:
            return False

    # Transformer with augmentations
    def _get_transformer_with_augment(self):
        # Transformer for Grayscale images
        # https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
        if(self.input_channels == 1):
            transformer = transforms.Compose([
                # transforms.Resize(self.img_size),
                # transforms.CenterCrop(self.img_size),

                # Convert to grayscale
                # Without the next line, the image will have 3 channels instead of 1
                transforms.Grayscale(num_output_channels=1), # <- Grayscale

                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(), 

                #################
                # AUGMENTATIONS #
                #################

                # Intensity augmentations (tensor-level):
                # Randomly adjust brightness and contrast by up to ±20% to
                # simulate variations in lighting/staining intensity across samples
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast)],
                    p=self.intense_prob
                ),
                # Gamma correction: Mimics nonlinear microscope/camera responses
                # Gamma = 1: No change. The image looks "natural" (linear brightness)
                # Gamma < 1 (e.g., 0.5): Dark areas get brighter, bright areas stay mostly the same
                # Gamma > 1 (e.g., 2.0): Bright areas get darker, dark areas stay mostly the same
                transforms.RandomApply(
                    [transforms.Lambda(self._gamma_correction)],  # <- Use method reference
                    p=self.gamma_prob
                ),

                # Optical augmentations (tensor-level)
                # Apply mild Gaussian blur
                # Simulates slight defocus or motion blur in microscopy
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=self.gauss_kernel_size, sigma=(self.gauss_sigma_min, self.gauss_sigma_max))], 
                    p=self.gauss_prob
                ),
                # Adds Poisson noise (a type of noise common in microscopy/imaging) scaled to 5% of pixel values
                transforms.RandomApply(
                    [transforms.Lambda(self._add_poisson_noise)],  # <- Use method reference
                    p=self.poiss_prob
                ),

                # Normalization of the image in the range [-1,1]
                # https://github.com/pytorch/vision/issues/288
                transforms.Normalize((0.5, ), (0.5, )),
            ])
            return transformer

        # Transformer for RGB images
        # https://pytorch.org/hub/pytorch_vision_resnet/
        elif(self.input_channels == 3):
            transformer = transforms.Compose([
                # transforms.Resize(self.img_size),
                # transforms.CenterCrop(self.img_size),

                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(),

                #################
                # AUGMENTATIONS #
                ################# 

                # Intensity augmentations
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation)],
                    p=self.intense_prob
                ),
                # Gamma correction
                transforms.RandomApply(
                    [transforms.Lambda(self._gamma_correction)],
                    p=self.gamma_prob
                ),
                
                # Optical augmentations
                # Gaussian blurr
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=self.gauss_kernel_size, sigma=(self.gauss_sigma_min, self.gauss_sigma_max))], 
                    p=self.gauss_prob
                ),
                # Poisson noise 
                transforms.RandomApply(
                    [transforms.Lambda(self._add_poisson_noise)],
                    p=self.poiss_prob
                ),

                # Normalization of the image in the range [-1,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # <- RGB
            ])
            return transformer

        else:
            return False
        
    # Print dataset examples
    def print_ds_examples(self, dataloader, device):
        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.tight_layout()
        plt.show()

    # Load dataset
    def load_training_dataset(self):
        
        # Check if dataroot exists
        if not os.path.exists(self.dataroot):
            print(f"Warning: Dataset directory '{self.dataroot}' does not exist!")
            return False
        
        # Check if dataroot is a directory
        if not os.path.isdir(self.dataroot):
            print(f"Warning: '{self.dataroot}' is not a directory!")
            return False
        
        # Check if there are any subdirectories (classes) in the dataroot
        subdirs = [d for d in os.listdir(self.dataroot) 
                if os.path.isdir(os.path.join(self.dataroot, d))]
        
        if not subdirs:
            print(f"Warning: No class subdirectories found in '{self.dataroot}'!")
            return False
        
        # Check if ALL subdirectories contain images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        empty_dirs = []
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.dataroot, subdir)
            has_images = False
            
            for file in os.listdir(subdir_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    has_images = True
                    break
            
            if not has_images:
                empty_dirs.append(subdir)
        
        if empty_dirs:
            print(f"Warning: The following directories contain no images: {', '.join(empty_dirs)}")
            return False
        
        # Load transformer with or without augmentations
        if self.train_use_augment:
            transformer = self._get_transformer_with_augment()
        else:
            transformer = self._get_transformer()
            
        if not transformer:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False
        
        dataset = dset.ImageFolder(self.dataroot, transform=transformer)
        # Create dataloader object
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True, 
        )

        # Set flag
        self.is_dataset_loaded = True

        return data_loader
