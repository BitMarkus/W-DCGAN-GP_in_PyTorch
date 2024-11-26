#############
# GENERATOR #
#############

from torch import nn
import torch
# Own modules
from settings import setting

class Generator(nn.Module):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):

        super(Generator, self).__init__()

        # Settings parameter
        # Size of z latent vector (i.e. size of generator input)
        self.latent_vector_size = setting["latent_vector_size"]
        # Number of channels in the training images. For color images this is 3
        self.image_channels = setting["img_channels"]
        self.gen_dropout = setting['gen_dropout']
        self.lrelu_alpha = setting["lrelu_alpha"]

        # Define deconvolutional blocks
        self.deconv_block_1 = self._generator_block(self.latent_vector_size, 512)
        self.deconv_block_2 = self._generator_block(512, 256)
        self.deconv_block_3 = self._generator_block(256, 256)
        self.deconv_block_4 = self._generator_block(256, 128)
        self.deconv_block_5 = self._generator_block(128, 128)
        self.deconv_block_6 = self._generator_block(128, 64)
        self.deconv_block_7 = self._generator_block(64, 32)
        self.deconv_block_8 = self._generator_block(32, 16)
        # Exit out of the network
        self.out_block = self._out_block(16, self.image_channels)

    #############################################################################################################
    # METHODS:

    def _generator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1
        ):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
            # nn.Dropout2d(self.gen_dropout)
        )

    def _out_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1
        ):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.Tanh(),
        )

    #############################################################################################################
    # FORWARD:

    def forward(self, x):
        # print(x.shape)

        # Convolutional upscaling:
        # 1x1 -> 2x2 
        x = self.deconv_block_1(x)
        assert (x.shape[1] == 512 and
                x.shape[2] == 2 and
                x.shape[3] == 2)  
        # 2x2 -> 4x4 
        x = self.deconv_block_2(x)
        assert (x.shape[1] == 256 and
                x.shape[2] == 4 and
                x.shape[3] == 4)   
        # 4x4 -> 8x8 
        x = self.deconv_block_3(x)
        assert (x.shape[1] == 256 and
                x.shape[2] == 8 and
                x.shape[3] == 8)   
        # 8x8 -> 16x16 
        x = self.deconv_block_4(x)
        assert (x.shape[1] == 128 and
                x.shape[2] == 16 and
                x.shape[3] == 16)    
        # 16x16 -> 32x32 
        x = self.deconv_block_5(x)
        assert (x.shape[1] == 128 and
                x.shape[2] == 32 and
                x.shape[3] == 32)    
        # 32x32 -> 64x64
        x = self.deconv_block_6(x)
        assert (x.shape[1] == 64 and
                x.shape[2] == 64 and
                x.shape[3] == 64)    
        # 64x64 -> 128x128
        x = self.deconv_block_7(x)
        assert (x.shape[1] == 32 and
                x.shape[2] == 128 and
                x.shape[3] == 128)    
        # 128x128 -> 256x256
        x = self.deconv_block_8(x)
        assert (x.shape[1] == 16 and
                x.shape[2] == 256 and
                x.shape[3] == 256)   
        
        # Output block:
        # 256x256 -> 512x512
        x = self.out_block(x)
        assert (x.shape[1] == self.image_channels and
                x.shape[2] == 512 and
                x.shape[3] == 512)   

        return x
