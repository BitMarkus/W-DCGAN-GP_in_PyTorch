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
        self.input_channels = setting["img_channels"]
        self.disc_dropout = setting["disc_dropout"]
        self.lrelu_alpha = setting["lrelu_alpha"]
        self.batch_size = setting["batch_size"]
        self.img_width = setting["img_size"]
        self.img_height = setting["img_size"]
        self.num_classes = setting["num_classes"]

        # Network input:
        self.in_block = self._in_block(self.input_channels, 16)
        # Define convolutional blocks = encoder
        self.conv_block_1 = self._discriminator_block(16, 32)
        self.conv_block_2 = self._discriminator_block(32, 64)
        self.conv_block_3 = self._discriminator_block(64, 64)
        self.conv_block_4 = self._discriminator_block(64, 128)
        self.conv_block_5 = self._discriminator_block(128, 128)
        self.conv_block_6 = self._discriminator_block(128, 256)
        self.conv_block_7 = self._discriminator_block(256, 256)
        self.conv_block_8 = self._discriminator_block(256, 512)
        # Define decoder
        # in_features = number of feature maps from the layer before = 512
        self.out_block = self._out_block(in_features=512, out_features=self.num_classes)

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
                # nn.Dropout2d(self.disc_dropout)
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
                # nn.Dropout2d(self.disc_dropout)
                ) 

    def _out_block(self, in_features, out_features):  
        return nn.Sequential(
            # 1x1 convolution to reduce feature maps to number of classes
            nn.Conv2d(in_features, out_features, 1, 1, 0, bias=False),
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
                x.shape[1] == 16 and 
                x.shape[2] == 256 and 
                x.shape[3] == 256)
        
        # ENCODER
        # Conv block 1:
        # 256x256 -> 128x128
        x = self.conv_block_1(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 32 and 
                x.shape[2] == 128 and 
                x.shape[3] == 128)
        # Conv block 2:
        # 128x128 -> 64x64
        x = self.conv_block_2(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 64 and 
                x.shape[2] == 64 and 
                x.shape[3] == 64)
        # Conv block 3:
        # 64x64 -> 32x32
        x = self.conv_block_3(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 64 and 
                x.shape[2] == 32 and 
                x.shape[3] == 32)
        # Conv block 4:
        # 32x32 -> 16x16
        x = self.conv_block_4(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 128 and 
                x.shape[2] == 16 and 
                x.shape[3] == 16)
        # Conv block 5:
        # 16x16 -> 8x8
        x = self.conv_block_5(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 128 and 
                x.shape[2] == 8 and 
                x.shape[3] == 8)
        # Conv block 6:
        # 8x8 -> 4x4
        x = self.conv_block_6(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 256 and 
                x.shape[2] == 4 and 
                x.shape[3] == 4)
        # Conv block 7:
        # 4x4 -> 2x2
        x = self.conv_block_7(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 256 and 
                x.shape[2] == 2 and 
                x.shape[3] == 2)
        # Conv block 8:
        # 2x2 -> 1x1
        x = self.conv_block_8(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == 512 and 
                x.shape[2] == 1 and 
                x.shape[3] == 1)
        
        # DECODER
        x = self.out_block(x) 
        # print(x.shape)
        # Reshapes the tensor from the last cov layer [batch_size, num_classes, 1, 1]
        # to the desired output tensor [batch size, num_classes]
        x = x.view(-1, self.num_classes * 1 * 1)
        # print(x.shape)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.num_classes)

        return x

