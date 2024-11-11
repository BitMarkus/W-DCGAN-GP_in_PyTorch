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

    # Show system information and select device (cpu or gpu)
    device = fn.show_cuda_and_versions()

    # Create program folders if they don't exist already
    fn.create_prg_folders()

    # Set random seed for reproducibility
    # manualSeed = 369
    manualSeed = random.randint(1, 10000)
    print("Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # torch.use_deterministic_algorithms(True) # Needed for reproducible results

    # Create a dataset object and load training images
    ds = Dataset()
    dataloader = ds.load_training_dataset()

    # Train on dataset
    train_object = Train(device, dataloader)
    train_object()

if __name__ == "__main__":
    main()