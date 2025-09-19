# Literature
# - https://neptune.ai/blog/gan-failure-modes

import random
import torch
import torch.nn.parallel
import torch.utils.data
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

        #########
        # Train #
        #########

        elif(menu1 == 2):       
            print("\n:TRAIN GAN:")  

            if(ds.is_dataset_loaded):
                # Create train object
                train = Train(device, dataloader)
                # Number of trainable parameters for generator and critic
                print("\nNumber of parameters for critic:")
                fn.count_parameters(train.netC)
                print("\nNumber of parameters for generator:")
                fn.count_parameters(train.netG)
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
            # Create train object
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