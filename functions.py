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
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
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
    
# Weights initialization
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
# https://github.com/martinarjovsky/WassersteinGAN/blob/f7a01e82007ea408647c451b9e1c8f1932a3db67/main.py#L108
# https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
# Initialization function according to DeepSeek:
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        # Xavier/Glorot initialization for Conv2d, ConvTranspose2d, and Linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # Initialize biases to zero
            nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        # Initialize scale (weight) to 1 and shift (bias) to 0 for normalization layers
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)   