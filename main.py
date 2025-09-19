# Literature
# - https://neptune.ai/blog/gan-failure-modes

import random
import torch
# Own modules
from settings import setting
from dataset import Dataset
import functions as fn
from train import Train
from generate import Generate

########
# MAIN #
########

def main():

    # Show system information and select device (cpu or gpu)
    device = fn.show_cuda_and_versions()
    # Create program folders if they don't exist already
    fn.create_prg_folders()

    ###############
    # Random Seed #
    ###############

    # Set random seed for reproducibility
    # manualSeed = 369
    manualSeed = random.randint(1, 10000)
    print("Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    ###########
    # Objects #
    ###########

    # Create a dataset object
    ds = Dataset()

    #############
    # Main Menu #
    #############

    while(True):  
        print("\n:MAIN MENU:")
        print("1) Load Training Dataset")
        print("2) Train GAN")
        print("3) Generate samples")
        print("4) Exit Program")
        menu1 = int(fn.input_int("Please choose: "))

        ###########
        # Dataset #
        ###########

        if(menu1 == 1):       
            print("\n:LOAD DATASET:")  
            # Load training images
            dataloader = ds.load_training_dataset()
            if(dataloader):
                print("Dataset successfully loaded.")
                # Print dataset info:
                # Get total number of training images
                print(f"Training on {ds.total_training_images} images")
                # Get counts per folder/class
                for folder, count in ds.dataset_image_counts.items():
                    print(f"Class '{folder}': {count} images")
                # Check if it's a conditional dataset (multiple classes)
                is_conditional = len(ds.dataset_image_counts) > 1
                print(f"Is conditional dataset: {is_conditional}")
                # Get the list of class names (useful for conditional GAN)
                class_names = list(ds.dataset_image_counts.keys())
                print(f"Available classes: {class_names}")

        #########
        # Train #
        #########

        elif(menu1 == 2):       
            print("\n:TRAIN GAN:")  

            if(ds.is_dataset_loaded):
                # Create train object
                train = Train(device, dataloader)

                # Number of trainable parameters for generator and critic
                print("Number of trainable parameters:")
                num_param_crit = fn.count_parameters(train.netC)
                print(f"Critic: {num_param_crit}")
                num_param_gen = fn.count_parameters(train.netG)
                print(f"Generator: {num_param_gen}")
                print(f"Ratio critic/generator: {num_param_crit/num_param_gen:.2f} (recommended: 1.2-2.0)")

                # Train on dataset
                train.train()
            else:
                print("Warning: No dataset loaded yet!")

        ############################
        # Create Generator Samples #
        ############################

        elif(menu1 == 3):       
            print("\n:GENERATE SAMPLES:")  
            dataloader = ds.load_training_dataset()
            # Create train object (necessary for generator network)
            train = Train(device, dataloader)
            samples = Generate(train)
            samples(device)

        ################
        # Exit Program #
        ################

        elif(menu1 == 4):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")    

if __name__ == "__main__":
    main()