# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
# https://stackoverflow.com/questions/66541812/kochat-in-use-runtimeerror-main-thread-is-not-in-main-loop
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from prettytable import PrettyTable
import torch
import torchvision.transforms as T
from PIL import Image
# Own modules
import functions as fn
from settings import setting
from os import listdir
from os.path import isfile, join

class Generate():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, train_object):

        # Path for saving checkpoints
        self.pth_checkpoints = setting["pth_checkpoints"]
        # Path for samples
        self.pth_samples = setting["pth_samples"]
        # Train object for Generator network
        self.train = train_object 

    #############################################################################################################
    # METHODS:    

    def print_checkpoints_table(self):
        # List of all model files in the checkpoint directory with the .model extension
        checkpoints = [file for file in os.listdir(self.pth_checkpoints) if file.endswith('.model')]
        # Create a tuple from list with indices (no idea how to do that in one line)
        checkpoints = list(enumerate(checkpoints))
        # print(checkpoints)
        table = PrettyTable(["ID", "Checkpoint"])
        for id, name in checkpoints:
            table.add_row([id+1, name])
        print()
        print(table)
        # print(f"Number of checkpoints: {len(checkpoints)}")
        return checkpoints
    
    def select_checkpoint(self, checkpoints, prompt):
        # Check input
        max_id = len(checkpoints)
        while(True):
            nr = input(prompt)
            if not(fn.check_int(nr)):
                print("Input is not an integer number! Try again...")
            else:
                nr = int(nr)
                if not(fn.check_int_range(nr, 1, max_id)):
                    print("Index out of range! Try again...")
                else:
                        return checkpoints[nr-1][1] 
                
    # Load a generator checkpoint/weights
    def load_checkpoint(self, model, chckpt_name, device):
        model.load_state_dict(torch.load(self.pth_checkpoints + chckpt_name))
        model.to(device)
        print(f'Weights from checkpoint {chckpt_name} successfully loaded.')

    # Function to select number of sample images
    def select_num_samples(self, prompt):
        while(True):
            nr = input(prompt)
            if not(fn.check_int(nr)):
                print("Input is not an integer number! Try again...")
            else:
                nr = int(nr)
                if not(fn.check_int_range(nr, 1, 1000)):
                    print("Number of images must be between 1 and 1000! Try again...")
                else:
                    return nr 
                
    # Function plots images from tensors and saves them as PIL images
    def save_sample_images(self, samples_tensors, img_channels):
        # Set image channels
        # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
        if(img_channels == 1):
            mode = 'L'      # 8-bit pixels, grayscale
        elif(img_channels == 3):
            mode = 'RGB'    # 8-bit RGB
        else:
            return False
        # define a transform to convert a tensor to PIL image
        transform = T.ToPILImage(mode=mode)
        # Iterate over image tensors
        for i, tensor in enumerate(samples_tensors):
            # Normalize the image tensor to [0, 1]
            tensor = (tensor + 1) / 2
            # Detach the tensor from its computation graph and move it to the CPU
            image_unflat = tensor.detach().cpu()
            # convert the tensor to PIL image using above transform
            img = transform(image_unflat)
            # Image name and path
            img_pth = f"{self.pth_samples}test_img_{i}.png"
            # Save the PIL image as .png fole
            img.save(img_pth)

    #############################################################################################################
    # CALL:

    def __call__(self, device):
        # Print a table with index for all checkpoints in the checkpoints folder
        checkpoints = self.print_checkpoints_table()
        # If list is not empty
        if(checkpoints):
            # Select a checkpoint by index
            checkpoint_name = self.select_checkpoint(checkpoints, "Select a checkpoint: ")
            # Load weights into the generator network
            self.load_checkpoint(self.train.netG, checkpoint_name, device)
            # Select a number of generated sample images
            num_samples = self.select_num_samples("Choose a number of sample images: ")
            # Get image tensors
            samples_tensors = self.train.create_generator_samples(num_samples)
            # Plot and save all images as image single files (PIL Image)
            self.save_sample_images(samples_tensors, setting["img_channels"])        
            print(f"{num_samples} sample images were successfully created and saved to folder {self.pth_samples}.")   
            return True
        else:
            print("The checkpoint folder is empty!")     

