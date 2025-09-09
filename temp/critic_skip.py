##########
# CRITIC #
##########

from torch import nn
import torch.nn.functional as F
# Own modules
from settings import setting

class Critic(nn.Module):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):

        super(Critic, self).__init__()

        # Settings parameter
        self.ngpu = setting["num_gpu"]
        self.lrelu_alpha = setting["lrelu_alpha"]
        self.batch_size = setting["batch_size"]
        self.img_width = setting["img_size"]
        self.img_height = setting["img_size"]
        self.img_channels = setting["img_channels"]
        self.dropout = setting["crit_dropout"]

        # Conv patameters
        self.kernel_size = setting["conv_kernel_size"]
        self.stride = setting["conv_stride"]
        self.padding = setting["conv_padding"]
        self.size_min_feature_maps = setting["size_min_feature_maps"]
        self.crit_chan_per_layer = setting["crit_chan_per_layer"]

        # Network input:
        # Define convolutional blocks
        # Use identity mapping if input channels match first layer, else use projection
        self.initial_skip = nn.Identity()
        if self.img_channels != self.crit_chan_per_layer[0]:
            self.initial_skip = nn.Sequential(
                nn.Conv2d(self.img_channels, self.crit_chan_per_layer[0], kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(self.crit_chan_per_layer[0], affine=False),
            )
        self.conv_block_1 = self._residual_conv_block(self.img_channels, self.crit_chan_per_layer[0], downsample=True)  # Out: [batch_size, ch, 256, 256]
        self.conv_block_2 = self._residual_conv_block(self.crit_chan_per_layer[0], self.crit_chan_per_layer[1], downsample=True)  # Out: [batch_size, ch, 128, 128]
        self.conv_block_3 = self._residual_conv_block(self.crit_chan_per_layer[1], self.crit_chan_per_layer[2], downsample=True)  # Out: [batch_size, ch, 64, 64]
        self.conv_block_4 = self._residual_conv_block(self.crit_chan_per_layer[2], self.crit_chan_per_layer[3], downsample=True)  # Out: [batch_size, ch, 32, 32]
        self.conv_block_5 = self._residual_conv_block(self.crit_chan_per_layer[3], self.crit_chan_per_layer[4], downsample=True)  # Out: [batch_size, ch, 16, 16]
        self.conv_block_6 = self._residual_conv_block(self.crit_chan_per_layer[4], self.crit_chan_per_layer[5], downsample=True)  # Out: [batch_size, ch, 8, 8]
        self.conv_block_7 = self._residual_conv_block(self.crit_chan_per_layer[5], self.crit_chan_per_layer[-1], downsample=True) # Out: [batch_size, max_ch=512, size_min_feature_maps=4, size_min_feature_maps=4]

        # Define decoder
        self.decoder = self._decoder()  # Out: [batch_size, 1]

    #############################################################################################################
    # METHODS:

    # Creates a residual block for the critic with optional downsampling
    # Returns an nn.ModuleDict containing the 'main' and 'skip' paths
    def _residual_conv_block(self, in_channels, out_channels, downsample=True):

        # Calculate stride for the main convolutional path
        main_stride = 2 if downsample else 1

        main_path = nn.Sequential(
            # First convolutional layer (with downsampling if specified)
            nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=main_stride,
                    padding=self.padding,
                    bias=False),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
            nn.Dropout2d(self.dropout),

            # Second convolutional layer (no change in spatial size)
            nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
            nn.InstanceNorm2d(out_channels, affine=False),
        )

        # Skip connection: Handles downsampling and channel size change
        skip_path = nn.Sequential()
        if in_channels != out_channels or downsample:
            # Use a convolutional layer to adjust channels and/or stride
            skip_path = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=2 if downsample else 1, # Use stride=2 for downsampling in skip
                    padding=0,
                    bias=False),
                nn.InstanceNorm2d(out_channels, affine=False),
            )
        else:
            skip_path = nn.Identity()

        return nn.ModuleDict({"main": main_path, "skip": skip_path})


    # Helper function to compute the forward pass for a residual block
    # Applies the main path, skip path, adds them, and applies activation
    def _forward_res_block(self, x, block):

        identity = x
        x = block["main"](x)    # Main path
        identity = block["skip"](identity)  # Skip path
        x = x + identity        # Add main and skip paths
        x = F.leaky_relu(x, self.lrelu_alpha, inplace=True) # Apply activation after addition
        return x

    # Improved version according to DeepSeek:
    def _decoder(self):
        return nn.Sequential(
            # Adaptive Pooling: Unlike standard pooling (e.g., AvgPool2d), you specify the output size (e.g., (H, W), here: (1, 1)),
            # and PyTorch automatically computes the required kernel size and stride to achieve that size
            # In: [batch_size, max_feature_maps=512, size_min_feature_maps=4, size_min_feature_maps=4]
            nn.AdaptiveAvgPool2d(1),  # Out: [batch_size, max_feature_maps, 1, 1]
            nn.Flatten(),             # Out: [batch_size, max_feature_maps]
            nn.Linear(self.crit_chan_per_layer[-1], 1),  # Out: [batch_size, 1]
        )

    #############################################################################################################
    # FORWARD:

    def forward(self, x):
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.img_channels and
                x.shape[2] == self.img_height and
                x.shape[3] == self.img_width)

        # Handle initial projection if needed
        identity_init = x
        identity_init = self.initial_skip(identity_init)

        # Convolutional blocks:
        # First block is special because it takes the raw image input
        x = self._forward_res_block(x, self.conv_block_1)
        x = x + identity_init # Add initial skip connection after the first block
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[0] and
                x.shape[2] == 256 and
                x.shape[3] == 256)
        # Out: [batch_size, ch, 256, 256]

        x = self._forward_res_block(x, self.conv_block_2)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[1] and
                x.shape[2] == 128 and
                x.shape[3] == 128)
        # Out: [batch_size, ch, 128, 128]

        x = self._forward_res_block(x, self.conv_block_3)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[2] and
                x.shape[2] == 64 and
                x.shape[3] == 64)
        # Out: [batch_size, ch, 64, 64]

        x = self._forward_res_block(x, self.conv_block_4)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[3] and
                x.shape[2] == 32 and
                x.shape[3] == 32)
        # Out: [batch_size, ch, 32, 32]

        x = self._forward_res_block(x, self.conv_block_5)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[4] and
                x.shape[2] == 16 and
                x.shape[3] == 16)
        # Out: [batch_size, ch, 16, 16]

        x = self._forward_res_block(x, self.conv_block_6)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[5] and
                x.shape[2] == 8 and
                x.shape[3] == 8)
        # Out: [batch_size, ch, 8, 8]

        x = self._forward_res_block(x, self.conv_block_7)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == self.crit_chan_per_layer[-1] and
                x.shape[2] == self.size_min_feature_maps and
                x.shape[3] == self.size_min_feature_maps)
        # Out: [batch_size, max_ch=512, size_min_feature_maps=4, size_min_feature_maps=4]

        # DECODER
        x = self.decoder(x)
        # Out: [batch_size, 1]
        # print(x.shape)
        assert (x.shape[0] <= self.batch_size and
                x.shape[1] == 1)

        return x