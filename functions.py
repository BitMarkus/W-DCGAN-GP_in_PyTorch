#############
# FUNCTIONS #
#############

import sys
from pathlib import Path
import torch
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
