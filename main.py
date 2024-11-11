import random
import torch
import torch.nn.parallel
import torch.utils.data
# Own modules
from settings import setting
from dataset import Dataset
import functions as fn
from train import Train

########
# MAIN #
########

def main():

    ##########
    # DEVICE #
    ##########

    device = fn.show_cuda_and_versions()

    ########
    # SEED #
    ########

    # Set random seed for reproducibility
    # manualSeed = 369
    manualSeed = random.randint(1, 10000)
    print("Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # torch.use_deterministic_algorithms(True) # Needed for reproducible results

    ########
    # DATA #
    ########

    # Create a dataset object and load training images
    ds = Dataset()
    dataloader = ds.load_training_dataset()

    ############
    # TRAINING #
    ############

    train_object = Train(device, dataloader)
    train_object()

if __name__ == "__main__":
    main()