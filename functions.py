#############
# FUNCTIONS #
#############

import sys
from pathlib import Path
import torch
from torch import nn
from prettytable import PrettyTable
# Own modules
from settings import setting

# Function to show if CUDA is working and software versions
def show_cuda_and_versions():
    print("\n>> DEVICE:")
    device = torch.device('cuda:0' if (torch.cuda.is_available() and setting["num_gpu"] > 0) else 'cpu')
    print("Using Device:", device)
    print(">> VERSIONS:")
    print("Python: ", sys.version, "")
    print("Pytorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    return device

# Function creates all working folders in the root directory of the program
# If they do not exist yet!
def create_prg_folders():
    # https://kodify.net/python/pathlib-path-mkdir-method/
    Path(setting["pth_data_root"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_samples"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_plots"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_checkpoints"]).mkdir(parents=True, exist_ok=True)

# Function to get a table of all tarinable parameters in a model and the total number
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
# Requires PrettyTable module
def count_parameters(model, print_table=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if(print_table):
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params

# Check variable for int
# Returns True if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_int(var):
    try:
        val = int(var)
        return True
    except ValueError:
        return False
    
# Checks if int parameters are within a certain range
def check_int_range(var, min, max):
    if(var >= min and var <= max):
        return True
    else:
        return False
    
# Creates an input with prompt
# which is checked, if the input is an integer number
# If not, the loop will continue until a valid number is entered
def input_int(prompt):
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again...")
        else:
            return int(nr)  
    