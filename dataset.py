####################
# Dataset handling #
####################

from torchvision.transforms import transforms
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils

# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
# https://stackoverflow.com/questions/66541812/kochat-in-use-runtimeerror-main-thread-is-not-in-main-loop
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
# Or try:
import matplotlib.pyplot as plt
plt.switch_backend('agg')
"""

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

    #############################################################################################################
    # METHODS:

    # Transforer for image resizing and normalization (and augmentation)
    def _get_transformer(self):
        # Transformer for Grayscale images
        # https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
        if(self.input_channels == 1):
            transformer = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(), 
                # Normalization of the image in the range [-1,1]
                # https://github.com/pytorch/vision/issues/288
                transforms.Normalize((0.5, ), (0.5, )),
                # Convert to grayscale
                # Without the next line, the image will have 3 channels instead of four
                transforms.Grayscale(num_output_channels=1), # <- Grayscale
            ])
            return transformer

        # Transformer for RGB images
        # https://pytorch.org/hub/pytorch_vision_resnet/
        elif(self.input_channels == 3):
            transformer = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                # 0-255 to 0-1, numpy to tensors:
                transforms.ToTensor(), 
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

    # Loads dataset for prediction
    def load_training_dataset(self):
        transformer = self._get_transformer()
        # Check if training images have either one or three channels
        if(transformer):
            dataset = dset.ImageFolder(self.dataroot, transform=transformer)
            # Create dataloader object
            data_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=self.num_workers
            )
            return data_loader  
        else:
            print("Loading of dataset failed! Input images must have either one (grayscale) or three (RGB) channels.")
            return False
        
