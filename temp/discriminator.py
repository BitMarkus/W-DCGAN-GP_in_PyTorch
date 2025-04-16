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
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = setting["num_gpu"]
        # Number of channels in the training images. For color images this is 3
        self.input_channels = setting["img_channels"]
        self.disc_dropout = setting["disc_dropout"]
        self.lrelu_alpha = setting["lrelu_alpha"]
        self.batch_size = setting["batch_size"]
        self.img_width = setting["img_size"]
        self.img_height = setting["img_size"]

        # Network input:
        self.in_block = self._in_block(self.input_channels, 64)
        # Define convolutional blocks = encoder
        self.conv_block_1 = self._discriminator_block(64, 128)
        self.conv_block_2 = self._discriminator_block(128, 128)
        self.conv_block_3 = self._discriminator_block(128, 256)
        self.conv_block_4 = self._discriminator_block(256, 256)
        self.conv_block_5 = self._discriminator_block(256, 512)
        # Define decoder
        # in_features = number of feature maps from the layer before = 256
        # pool_size = 8 as the last feature maps have a 8x8 px dimension
        self.decoder = self._decoder(in_features=512, out_features=1, pool_size=8)

    #############################################################################################################
    # METHODS:

    def _in_block(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,            
            ):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                # No batch normalization in first layer
                nn.LeakyReLU(self.lrelu_alpha, inplace=True),
                nn.Dropout2d(self.disc_dropout)
                )             

    def _discriminator_block(        
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
            ):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                # NO batch normalization when using a Wasserstein GAN with gradient penalty!
                # nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(self.lrelu_alpha, inplace=True),
                nn.Dropout2d(self.disc_dropout)
                ) 

    def _decoder(self, in_features, out_features, pool_size):  
        return nn.Sequential(
            # 1x1 convolution to reduce feature maps to number of classes
            nn.Conv2d(in_features, out_features, 1, 1, 0, bias=False),
            # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/4
            # https://blog.paperspace.com/global-pooling-in-convolutional-neural-networks/
            # Global average pooling: 8 as the size of the last feature maps is 8x8 
            nn.AvgPool2d(pool_size),
            # No activation function here!
            # Vanilla GAN: nn.BCEWithLogitsLoss is a cross entropy loss that comes inside a sigmoid function
            # W-GAN: no activation function is needed at all here
        )  
    
    #############################################################################################################
    # FORWARD:

    def forward(self, x):
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.input_channels and 
                x.shape[2] == self.img_height and 
                x.shape[3] == self.img_width)
             
        
        # Network input
        # 512x512 -> 256x256
        x = self.in_block(x)
        # print(x.shape)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 64 and 
                x.shape[2] == 256 and 
                x.shape[3] == 256)
        
        # ENCODER
        # Conv block 1:
        # 256x256 -> 128x128
        x = self.conv_block_1(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 128 and 
                x.shape[2] == 128 and 
                x.shape[3] == 128)
        # Conv block 2:
        # 128x128 -> 64x64
        x = self.conv_block_2(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 128 and 
                x.shape[2] == 64 and 
                x.shape[3] == 64)
        # Conv block 3:
        # 64x64 -> 32x32
        x = self.conv_block_3(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 256 and 
                x.shape[2] == 32 and 
                x.shape[3] == 32)
        # Conv block 4:
        # 32x32 -> 16x16
        x = self.conv_block_4(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 256 and 
                x.shape[2] == 16 and 
                x.shape[3] == 16)
        # Conv block 5:
        # 16x16 -> 8x8
        x = self.conv_block_5(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 512 and 
                x.shape[2] == 8 and 
                x.shape[3] == 8)
        
        # DECODER
        x = self.decoder(x) 
        # print(x.shape)
        # Reshapes the tensor from the global average pooling layer [batch_size, 1, 1, 1]
        # to the desired output tensor [batch size, 1]
        x = x.view(-1, 1 * 1 * 1)
        # print(x.shape)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 1)

        return x

    
"""
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        # Settings parameter
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = setting["num_gpu"]
        # Size of feature maps in generator
        self.channel_multi = setting["disc_fm_size"]
        # Number of channels in the training images. For color images this is 3
        self.input_channel = setting["img_channels"]
        # Dropout
        self.disc_dropout = setting["disc_dropout"]

        self.disc = nn.Sequential(
            # 512x512 -> 256x256
            self._discriminator_block(self.input_channel, self.channel_multi, first_layer=True),
            # 256x256 -> 128x128
            self._discriminator_block(self.channel_multi, self.channel_multi * 2),
            # 128x128 -> 64x64
            self._discriminator_block(self.channel_multi * 2, self.channel_multi * 4),
            # 64x64 -> 32x32
            self._discriminator_block(self.channel_multi * 4, self.channel_multi * 8),
            # 32x32 -> 16x16
            self._discriminator_block(self.channel_multi * 8, self.channel_multi * 16),
            # 16x16 -> 8x8
            self._discriminator_block(self.channel_multi * 16, self.channel_multi * 32),
            # 8x8 -> 4x4
            self._discriminator_block(self.channel_multi * 32, self.channel_multi * 64),
            # 4x4 -> 1x1
            self._discriminator_block(self.channel_multi * 64, 1, final_layer=True)
        )

    def _discriminator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        first_layer=False,
        final_layer=False,
    ):
        if(first_layer):
            # No batch normalization in first layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, inplace=True),
                # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
                nn.Dropout2d(self.disc_dropout)
            )
        elif(final_layer):
            # No activation function in last layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, 0),
                nn.Sigmoid() # <- For testing purposes
                # No activation function here!
                # nn.BCEWithLogitsLoss is a cross entropy loss that comes inside a sigmoid function
            )        
        else:
            # All hidden layers in between
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(self.disc_dropout)
            )   

    def forward(self, img):
        out = self.disc(img)
        # print("disc shape out 1:", disc_pred.shape)
        # out = disc_pred.view(len(disc_pred), -1)
        # print("disc shape out 2:", out.shape)
        # print("out:", out)
        return out
"""
