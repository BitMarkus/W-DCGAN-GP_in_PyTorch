#############
# GENERATOR #
#############

from torch import nn
# Own modules
from attention import SelfAttention
from settings import setting

class Generator(nn.Module):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):

        super(Generator, self).__init__()

        # Settings parameter
        self.latent_vector_size = setting["latent_vector_size"]
        self.img_channels = setting["img_channels"]
        self.img_size = setting["img_size"]
        self.lrelu_alpha = setting["lrelu_alpha"]
        self.batch_size = setting["batch_size"]
        # Use self attention
        self.use_attention = setting["use_attention"]
        self.attention_min_channels = setting["attention_min_channels"]

        # Conv patameters
        self.kernel_size = setting["conv_kernel_size"]
        self.stride = setting["conv_stride"]
        self.padding = setting["conv_padding"]
        self.out_padding = setting["conv_out_padding"]
        self.size_min_feature_maps = setting["size_min_feature_maps"]
        self.gen_chan_per_layer = setting["gen_chan_per_layer"]

        # Entry into the network
        self.input_block = self._input_block(
            self.latent_vector_size, 
            self.size_min_feature_maps*self.size_min_feature_maps*self.gen_chan_per_layer[0]
        )

        # Out: [batch_size, max_ch=512*size_min_feature_maps=4*size_min_feature_maps=4] -> Reshape
        # Deconvolutional layers with attention: 
        # Input: [batch_size, max_ch=512, size_min_feature_maps=4, size_min_feature_maps=4]
        # Create deconvolutional blocks as a ModuleList
        self.deconv_blocks = nn.ModuleList()
        for i in range(len(self.gen_chan_per_layer) - 1):
            self.deconv_blocks.append(
                self._residual_deconv_block(self.gen_chan_per_layer[i], self.gen_chan_per_layer[i+1])
            )

        # Exit out of the network
        self.output_block = self._output_block(self.gen_chan_per_layer[6], self.img_channels)
        # Out: [batch_size, img_channels, img_size, img_size]


    #############################################################################################################
    # METHODS:

    def _input_block(self, in_features, out_features): 
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
        ) 

    def _residual_deconv_block(self, in_channels, out_channels):

        # Only add attention if enabled and channels are large enough
        use_attention = (self.use_attention and out_channels >= self.attention_min_channels)
        
        return nn.ModuleDict({

            # Main path: A sequence of deconvolution + convolution operations with nonlinearities
            "main": nn.Sequential(

                # Upsampling and channel decrease
                nn.ConvTranspose2d(
                    in_channels, 
                    out_channels,         
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.out_padding,
                    bias=False
                    ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(self.lrelu_alpha, inplace=True),
                
                # Convolution without change of spatial size or channel number
                nn.Conv2d(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(self.lrelu_alpha, inplace=True),

                # Add attention here if enabled
                SelfAttention(out_channels) if self.use_attention else nn.Identity()
            ),

            # Skip path: A simpler path that adapts the input to match the output dimensions
            "skip": nn.Sequential(

                # Upsample residual if spatial size changes
                nn.ConvTranspose2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.out_padding,
                    # The purpose of skip connections is to provide an unmodified path when possible
                    # When you have bias=True, you're adding learnable parameters that could modify the signal
                    # With bias=False, the operation is purely linear (weights only)
                    bias=False
                )
                if in_channels != out_channels or self.stride != 1  
                else nn.Identity(),

                # The 1x1 convolution branch would only activate if channels differed but stride was 1
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=1, 
                    bias=False
                )
                if in_channels != out_channels and self.stride == 1
                else nn.Identity()
            )
        })

    def _output_block(self, in_channels, out_channels):
        return nn.Sequential(

            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                output_padding=self.out_padding,
                bias=True
            ),

            nn.Tanh(),
        )

    #############################################################################################################
    # FORWARD:

    def forward(self, x):

        # Input noise vector: [batch_size, self.latent_vector_size]
        assert (x.shape[1] == self.latent_vector_size)

        # Input block
        x = self.input_block(x)
        assert (x.shape[1] == self.size_min_feature_maps*self.size_min_feature_maps*self.gen_chan_per_layer[0])    
        # Out: [batch_size, max_ch*size_min_feature_maps*size_min_feature_maps]

        # Reshape to [batch_size, max_ch, size_min_feature_maps, size_min_feature_maps]
        x = x.view(-1, self.gen_chan_per_layer[0], self.size_min_feature_maps, self.size_min_feature_maps)
        assert (x.shape[1] == self.gen_chan_per_layer[0] and
                x.shape[2] == self.size_min_feature_maps and
                x.shape[3] == self.size_min_feature_maps)     
        
        # Process through each residual block
        for block in self.deconv_blocks:
            residual = x
            x = block["main"](x)  # Main path
            residual = block["skip"](residual)  # Skip path (1x1 conv if needed)
            x = x + residual  # Residual connection

        assert (x.shape[1] == self.gen_chan_per_layer[6] and
                x.shape[2] == 256 and
                x.shape[3] == 256) 
        # Out: [batch_size, ch, 256, 256]

        # Output block
        x = self.output_block(x)
        assert (x.shape[1] == self.img_channels and
                x.shape[2] == self.img_size and
                x.shape[3] == self.img_size) 
        # Out: [batch_size, img_channels, img_size, img_size]  

        return x