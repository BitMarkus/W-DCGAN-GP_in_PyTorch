#################
# DISCRIMINATOR #
#################

from torch import nn
# Own modules
from settings import setting

class Discriminator(nn.Module):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):

        super(Discriminator, self).__init__()

        # Settings parameter
        self.ngpu = setting["num_gpu"]
        self.lrelu_alpha = setting["lrelu_alpha"]
        self.batch_size = setting["batch_size"]
        self.img_width = setting["img_size"]
        self.img_height = setting["img_size"]
        self.img_channels = setting["img_channels"]

        # Conv patameters
        self.kernel_size = setting["conv_kernel_size"]
        self.stride = setting["conv_stride"]
        self.padding = setting["conv_padding"]
        self.size_min_feature_maps = setting["size_min_feature_maps"]
        self.disc_chan_per_layer = setting["disc_chan_per_layer"]

        # Network input:
        # Define convolutional blocks
        # Channels per layer: [8, 16, 32, 64, 128, 256]
        self.conv_block_1 = self._conv_block(self.img_channels, self.disc_chan_per_layer[0])            # Out: [batch_size, 8, 256, 256]
        self.conv_block_2 = self._conv_block(self.disc_chan_per_layer[0], self.disc_chan_per_layer[1])  # Out: [batch_size, 16, 128, 128]
        self.conv_block_3 = self._conv_block(self.disc_chan_per_layer[1], self.disc_chan_per_layer[2])  # Out: [batch_size, 32, 64, 64]
        self.conv_block_4 = self._conv_block(self.disc_chan_per_layer[2], self.disc_chan_per_layer[3])  # Out: [batch_size, 64, 32, 32]
        self.conv_block_5 = self._conv_block(self.disc_chan_per_layer[3], self.disc_chan_per_layer[4])  # Out: [batch_size, 128, 16, 16]
        self.conv_block_6 = self._conv_block(self.disc_chan_per_layer[4], self.disc_chan_per_layer[5])  # Out: [batch_size, 256, 8, 8]
        self.conv_block_7 = self._conv_block(self.disc_chan_per_layer[5], self.disc_chan_per_layer[-1]) # Out: [batch_size, self.num_max_feature_maps=512, self.size_min_feature_maps=4, self.size_min_feature_maps=4]

        # Define decoder
        self.decoder = self._decoder(pool_size=self.size_min_feature_maps)  # Out: [batch_size, 1]

    #############################################################################################################
    # METHODS:  

    def _conv_block(self, in_channels, out_channels):
            return nn.Sequential(
                 
                # Strided convolutional layer
                nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size=self.kernel_size, 
                        stride=self.stride, 
                        padding=self.padding,),
                # NO batch normalization when using a Wasserstein GAN with gradient penalty!
                nn.LeakyReLU(self.lrelu_alpha, inplace=True),

                # Extra convolutional layer with no change of image size or channel number
                nn.Conv2d(out_channels, 
                        out_channels, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1),
                # NO batch normalization when using a Wasserstein GAN with gradient penalty!
                nn.LeakyReLU(self.lrelu_alpha, inplace=True),
                ) 

    def _decoder(self, pool_size,):  
        return nn.Sequential(
            # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/4
            # https://blog.paperspace.com/global-pooling-in-convolutional-neural-networks/
            # Global average pooling: 4 as the size of the last feature maps is 4x4 
            # In: [batch_size, num_max_feature_maps = 512, 4, 4]
            nn.AvgPool2d(pool_size),
            # Out: [batch_size, num_max_feature_maps = 512, 1, 1]
            # Flatten
            nn.Flatten(),
            # Out: [batch_size, num_max_feature_maps = 512]
            # Dense layer to model complex relationships
            nn.Linear(self.disc_chan_per_layer[-1], 128),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
            # Out: [batch_size, 128]
            # Output layer (no activation for WGAN-GP)
            nn.Linear(128, 1),
            # Out: [batch_size, 1]
        )

    #############################################################################################################
    # FORWARD:

    def forward(self, x):
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.img_channels and 
                x.shape[2] == self.img_height and 
                x.shape[3] == self.img_width)
             
        # Convolutional blocks:
        x = self.conv_block_1(x)
        # print(x.shape)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[0] and 
                x.shape[2] == 256 and 
                x.shape[3] == 256)
        # Out: [batch_size, 8, 256, 256]

        x = self.conv_block_2(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[1] and 
                x.shape[2] == 128 and 
                x.shape[3] == 128)
        # Out: [batch_size, 16, 128, 128]

        x = self.conv_block_3(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[2] and 
                x.shape[2] == 64 and 
                x.shape[3] == 64)
        # Out: [batch_size, 32, 64, 64]

        x = self.conv_block_4(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[3] and 
                x.shape[2] == 32 and 
                x.shape[3] == 32)
        # Out: [batch_size, 64, 32, 32]

        x = self.conv_block_5(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[4] and 
                x.shape[2] == 16 and 
                x.shape[3] == 16)
        # Out: [batch_size, 128, 16, 16]

        x = self.conv_block_6(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[5] and 
                x.shape[2] == 8 and 
                x.shape[3] == 8)
        # Out: [batch_size, 256, 8, 8]

        x = self.conv_block_7(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.disc_chan_per_layer[-1] and 
                x.shape[2] == self.size_min_feature_maps and 
                x.shape[3] == self.size_min_feature_maps)
        # Out: [batch_size, num_max_feature_maps=512, size_min_feature_maps=4, size_min_feature_maps=4]
        
        # DECODER
        x = self.decoder(x) 
        # Out: [batch_size, 1]
        # print(x.shape)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 1)

        return x

