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
        self.img_channels = setting["img_channels"]
        self.img_size = setting["img_size"]
        self.lrelu_alpha = setting["lrelu_alpha"]
        self.batch_size = setting["batch_size"]
        # Conv patameters
        self.kernel_size = setting["conv_kernel_size"]
        self.stride = setting["conv_stride"]
        self.padding = setting["conv_padding"]
        self.out_padding = setting["conv_out_padding"]
        self.size_min_feature_maps = setting["size_min_feature_maps"]
        self.gen_chan_per_layer = setting["gen_chan_per_layer"]

        # Entry into the network
        self.input_block = self._input_block(self.latent_vector_size, self.size_min_feature_maps*self.size_min_feature_maps*self.gen_chan_per_layer[0])
        # Out: [batch_size, max_ch=512*size_min_feature_maps=4*size_min_feature_maps=4] -> Reshape
        # Deconvolutional layers: 
        # Input: [batch_size, max_ch=512, size_min_feature_maps=4, size_min_feature_maps=4]
        self.deconv_block_1 = self._deconv_block(self.gen_chan_per_layer[0], self.gen_chan_per_layer[1])  # Out: [batch_size, ch, 8, 8] 
        self.deconv_block_2 = self._deconv_block(self.gen_chan_per_layer[1], self.gen_chan_per_layer[2])  # Out: [batch_size, ch, 16, 16]
        self.deconv_block_3 = self._deconv_block(self.gen_chan_per_layer[2], self.gen_chan_per_layer[3])  # Out: [batch_size, ch, 32, 32]
        self.deconv_block_4 = self._deconv_block(self.gen_chan_per_layer[3], self.gen_chan_per_layer[4])  # Out: [batch_size, ch, 64, 64]
        self.deconv_block_5 = self._deconv_block(self.gen_chan_per_layer[4], self.gen_chan_per_layer[5])  # Out: [batch_size, ch, 128, 128]
        self.deconv_block_6 = self._deconv_block(self.gen_chan_per_layer[5], self.gen_chan_per_layer[6])  # Out: [batch_size, ch, 256, 256]
        # Exit out of the network
        self.output_block = self._output_block(self.gen_chan_per_layer[6], self.img_channels)
        # Out: [batch_size, img_channels, img_size, img_size]


    #############################################################################################################
    # METHODS:

    def _input_block(self, in_features, out_features): 
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            # According to DeepSeek the next two lines are not necessary:
            # nn.BatchNorm1d(out_features),
            # nn.LeakyReLU(self.lrelu_alpha, inplace=True),
        ) 
    
    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(

            # Strided convolutional layer
            nn.ConvTranspose2d(in_channels, 
                               out_channels, 
                               kernel_size=self.kernel_size, 
                               stride=self.stride, 
                               padding=self.padding, 
                               output_padding=self.out_padding,
                               # bias=False if conv/deconv layer is followed by a batch-, layer- group- or instance normalization layer
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),

            # Intermediate layer with no change of image size or channel number
            nn.Conv2d(out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    # bias=False if conv/deconv layer is followed by a batch-, layer- group- or instance normalization layer
                    bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
        )

    def _output_block(self, in_channels, out_channels):
        return nn.Sequential(

            # Strided convolutional layer
            nn.ConvTranspose2d(in_channels, 
                               out_channels, 
                               kernel_size=self.kernel_size, 
                               stride=self.stride, 
                               padding=self.padding, 
                               output_padding=self.out_padding,
                               bias=True),

            nn.Tanh(),
        )

    #############################################################################################################
    # FORWARD:

    def forward(self, x):
        # Input noise vector: [batch_size, self.latent_vector_size]
        # print(x.shape)
        assert (x.shape[1] == self.latent_vector_size)

        # Input block:
        x = self.input_block(x)
        # print(x.shape)
        assert (x.shape[1] == self.size_min_feature_maps*self.size_min_feature_maps*self.gen_chan_per_layer[0])    
        # Out: [batch_size, max_ch=512*size_min_feature_maps=4*size_min_feature_maps=4]

        # Reshape 4*4*512 to [batch_size, max_ch=512, size_min_feature_maps=4, size_min_feature_maps=4]
        x = x.view(-1, self.gen_chan_per_layer[0], self.size_min_feature_maps, self.size_min_feature_maps)
        assert (x.shape[1] == self.gen_chan_per_layer[0] and
                x.shape[2] == self.size_min_feature_maps and
                x.shape[3] == self.size_min_feature_maps)     

        # Convolutional upscaling:
        x = self.deconv_block_1(x)
        assert (x.shape[1] == self.gen_chan_per_layer[1] and
                x.shape[2] == 8 and
                x.shape[3] == 8) 
        # Out: [batch_size, ch, 8, 8] 
        
        x = self.deconv_block_2(x)
        assert (x.shape[1] == self.gen_chan_per_layer[2] and
                x.shape[2] == 16 and
                x.shape[3] == 16)   
        # Out: [batch_size, ch, 16, 16]

        x = self.deconv_block_3(x)
        assert (x.shape[1] == self.gen_chan_per_layer[3] and
                x.shape[2] == 32 and
                x.shape[3] == 32) 
        # Out: [batch_size, ch, 32, 32]
          
        x = self.deconv_block_4(x)
        assert (x.shape[1] == self.gen_chan_per_layer[4] and
                x.shape[2] == 64 and
                x.shape[3] == 64)    
        # Out: [batch_size, ch, 64, 64]

        x = self.deconv_block_5(x)
        assert (x.shape[1] == self.gen_chan_per_layer[5] and
                x.shape[2] == 128 and
                x.shape[3] == 128)    
        # Out: [batch_size, ch, 128, 128]

        x = self.deconv_block_6(x)
        assert (x.shape[1] == self.gen_chan_per_layer[6] and
                x.shape[2] == 256 and
                x.shape[3] == 256) 
        # Out: [batch_size, ch, 256, 256]
        
        # Output block:
        x = self.output_block(x)
        assert (x.shape[1] == self.img_channels and
                x.shape[2] == self.img_size and
                x.shape[3] == self.img_size) 
        # Out: [batch_size, img_channels, img_size, img_size]  

        return x
