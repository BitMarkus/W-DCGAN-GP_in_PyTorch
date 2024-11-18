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
        self.batch_size = setting["batch_size"]
        self.lrelu_alpha = setting["lrelu_alpha"]

        # Entry into the network
        self.in_block = self._in_block(self.latent_vector_size, 8*8*512)
        # Define deconvolutional blocks
        self.deconv_block_1 = self._generator_block(512, 256)
        self.deconv_block_2 = self._generator_block(256, 256)
        self.deconv_block_3 = self._generator_block(256, 128)
        self.deconv_block_4 = self._generator_block(128, 128)
        self.deconv_block_5 = self._generator_block(128, 64)
        # Exit out of the network
        self.out_block = self._out_block(64, self.image_channels)

    #############################################################################################################
    # METHODS:

    def _in_block(self, in_features, out_features): 
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
        ) 

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
            nn.Dropout2d(self.gen_dropout)
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
        # Input noise vector: [batch_size, self.latent_vector_size]
        assert (x.shape[1] == self.latent_vector_size)

        # Input block:
        x = self.in_block(x)
        # [batch_size, 8*8*512]
        assert (x.shape[1] == 8*8*512)       

        # Reshape 8*8*512 to [batch_size, 512, 8, 8]
        x = x.view(-1, 512, 8, 8)
        # [batch_size, 512, 8, 8]
        assert (x.shape[1] == 512 and
                x.shape[2] == 8 and
                x.shape[3] == 8)     

        # Convolutional upscaling:
        # 8x8 -> 16x16 
        x = self.deconv_block_1(x)
        assert (x.shape[1] == 256 and
                x.shape[2] == 16 and
                x.shape[3] == 16)    
        # 16x16 -> 32x32 
        x = self.deconv_block_2(x)
        assert (x.shape[1] == 256 and
                x.shape[2] == 32 and
                x.shape[3] == 32)    
        # 32x32 -> 64x64
        x = self.deconv_block_3(x)
        assert (x.shape[1] == 128 and
                x.shape[2] == 64 and
                x.shape[3] == 64)    
        # 64x64 -> 128x128
        x = self.deconv_block_4(x)
        assert (x.shape[1] == 128 and
                x.shape[2] == 128 and
                x.shape[3] == 128)    
        # 128x128 -> 256x256
        x = self.deconv_block_5(x)
        assert (x.shape[1] == 64 and
                x.shape[2] == 256 and
                x.shape[3] == 256)   
        
        # Output block:
        # 256x256 -> 512x512
        x = self.out_block(x)
        assert (x.shape[1] == 1 and
                x.shape[2] == 512 and
                x.shape[3] == 512)   

        return x
    
"""
class Generator(nn.Module):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):

        super(Generator, self).__init__()

        # Settings parameter
        # Size of z latent vector (i.e. size of generator input)
        self.latent_vector_size = setting["latent_vector_size"]
        # Size of feature maps in generator
        self.channel_multi = setting["gen_fm_size"]
        # Number of channels in the training images. For color images this is 3
        self.image_channels = setting["img_channels"]
        # Dropout
        self.gen_dropout = setting['gen_dropout']
        # Batch size
        self.batch_size = setting["batch_size"]

        # Entry into the network
        self.in_block = self._in_block(self.latent_vector_size, 8*8*self.channel_multi*64)
        # Define deconvolutional blocks
        self.deconv_block_1 = self._generator_block(self.channel_multi * 64, self.channel_multi * 32)
        self.deconv_block_2 = self._generator_block(self.channel_multi * 32, self.channel_multi * 16)
        self.deconv_block_3 = self._generator_block(self.channel_multi * 16, self.channel_multi * 8)
        self.deconv_block_4 = self._generator_block(self.channel_multi * 8, self.channel_multi * 4)
        self.deconv_block_5 = self._generator_block(self.channel_multi * 4, self.channel_multi * 2)
        # Exit out of the network
        self.out_block = self._out_block(self.channel_multi * 2, self.image_channels)

    #############################################################################################################
    # METHODS:

    def _in_block(self, in_features, out_features): 
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        ) 

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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.gen_dropout)
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
        # Input noise vector: [64, 512]
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 512)

        # Alternative to different noise vectors
        # Adds a dense input layer, which has as input the latent vector size
        # and as output the amount of nodes to reshape it into 8x8x512
        # x = x.view(-1, 512*1*1) -> torch.Size([64, 512])
        # print(x.shape)

        # Input block: 
        x = self.in_block(x)
        # [batch_size, 32768]
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 8*8*self.channel_multi*64)       

        # Reshape 32768 to 512x8x8
        x = x.view(-1, 512, 8, 8)
        # [batch_size, 512, 8, 8]
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 512 and
                x.shape[2] == 8 and
                x.shape[3] == 8)     

        # Convolutional upscaling:
        # 8x8 -> 16x16 
        x = self.deconv_block_1(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 256 and
                x.shape[2] == 16 and
                x.shape[3] == 16)    
        # 16x16 -> 32x32 
        x = self.deconv_block_2(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 128 and
                x.shape[2] == 32 and
                x.shape[3] == 32)    
        # 32x32 -> 64x64
        x = self.deconv_block_3(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 64 and
                x.shape[2] == 64 and
                x.shape[3] == 64)    
        # 64x64 -> 128x128
        x = self.deconv_block_4(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 32 and
                x.shape[2] == 128 and
                x.shape[3] == 128)    
        # 128x128 -> 256x256
        x = self.deconv_block_5(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 16 and
                x.shape[2] == 256 and
                x.shape[3] == 256)   
        
        # Output block:
        # 256x256 -> 512x512
        x = self.out_block(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 1 and
                x.shape[2] == 512 and
                x.shape[3] == 512)   

        return x
"""

