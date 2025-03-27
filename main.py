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
    # Needed for reproducible results -> doesn't work with my modified training loop
    # torch.use_deterministic_algorithms(True) 

    ################
    # Load Dataset #
    ################

    # Create a dataset object and load training images
    ds = Dataset()
    dataloader = ds.load_training_dataset()

    ######################
    # Networks and Train #
    ######################

    # Train on dataset
    train = Train(device, dataloader)
    # Number of trainable parameters for generator and critic
    print("\nNumber of parameters for critic:")
    fn.count_parameters(train.netC)
    print("\nNumber of parameters for generator:")
    fn.count_parameters(train.netG)
    # Train on dataset
    # train.train()

    ############################
    # Create Generator Samples #
    ############################

    # Create a generate object
    samples = Generate(train)
    samples(device)


if __name__ == "__main__":
    main()