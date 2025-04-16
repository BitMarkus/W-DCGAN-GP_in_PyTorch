##########
# CRITIC #
##########

from torch import nn
# Own modules
from settings import setting
from attention import SelfAttention

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
        # Use self attention
        self.use_attention = setting["use_attention"]
        self.attention_min_channels = setting["attention_min_channels"]
        self.attention_layers = setting["attention_layers"]

        # Conv patameters
        self.kernel_size = setting["conv_kernel_size"]
        self.stride = setting["conv_stride"]
        self.padding = setting["conv_padding"]
        self.size_min_feature_maps = setting["size_min_feature_maps"]
        self.crit_chan_per_layer = setting["crit_chan_per_layer"]

        # Network input:
        # Define convolutional blocks
        self.conv_block_1 = self._conv_block(self.img_channels, self.crit_chan_per_layer[0])            # Out: [batch_size, ch, 256, 256]
        self.conv_block_2 = self._conv_block(self.crit_chan_per_layer[0], self.crit_chan_per_layer[1])  # Out: [batch_size, ch, 128, 128]
        self.conv_block_3 = self._conv_block(self.crit_chan_per_layer[1], self.crit_chan_per_layer[2])  # Out: [batch_size, ch, 64, 64]
        self.conv_block_4 = self._conv_block(self.crit_chan_per_layer[2], self.crit_chan_per_layer[3])  # Out: [batch_size, ch, 32, 32]
        self.conv_block_5 = self._conv_block(self.crit_chan_per_layer[3], self.crit_chan_per_layer[4])  # Out: [batch_size, ch, 16, 16]
        self.conv_block_6 = self._conv_block(self.crit_chan_per_layer[4], self.crit_chan_per_layer[5])  # Out: [batch_size, ch, 8, 8]
        self.conv_block_7 = self._conv_block(self.crit_chan_per_layer[5], self.crit_chan_per_layer[-1]) # Out: [batch_size, max_ch=512, size_min_feature_maps=4, size_min_feature_maps=4
        # Define decoder
        self.decoder = self._decoder()  # Out: [batch_size, 1]

        # SelfAttention
        if self.use_attention:
            self.attention_layers = []
            for i, (in_ch, out_ch) in enumerate(zip(self.crit_chan_per_layer, self.crit_chan_per_layer[1:])):
                if out_ch >= self.attention_min_channels and f"layer_{i}" in self.attention_layers:
                    self.attention_layers.append(i)
            
            self.attention_modules = nn.ModuleList([
                SelfAttention(out_ch) 
                for i, out_ch in enumerate(self.crit_chan_per_layer[1:])
                if i in self.attention_layers
            ])

    #############################################################################################################
    # METHODS: 

    # Improved version according to DeepSeek:
    def _conv_block(self, in_channels, out_channels):
        # Only add attention if enabled and channels are large enough
        use_attention = (self.use_attention and out_channels >= self.attention_min_channels)
        
        layers = [
            # Strided convolutional layer with spectral normalization
            nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=self.kernel_size, 
                    stride=self.stride, 
                    padding=self.padding,
                    # bias=False if conv/deconv layer is followed by a batch-, layer- group- or instance normalization layer
                    bias=False),
            # NO batch normalization when using a Wasserstein GAN with gradient penalty!
            # Instead use InstanceNorm without learnable parameters (affine=False)
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
            nn.Dropout2d(self.dropout),

            # Extra convolutional layer with no change of image size or channel number
            nn.Conv2d(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    # bias=False if conv/deconv layer is followed by a batch-, layer- group- or instance normalization layer
                    bias=False),
            # NO batch normalization when using a Wasserstein GAN with gradient penalty!
            # Instead use InstanceNorm without learnable parameters (affine=False)
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(self.lrelu_alpha, inplace=True),
        ] 

        # Add attention at the end of the block if enabled
        if use_attention:
            layers.append(SelfAttention(out_channels))
            
        return nn.Sequential(*layers)

    # Improved version according to DeepSeek:
    def _decoder(self):  
        return nn.Sequential(
            # Adaptive Pooling: Unlike standard pooling (e.g., AvgPool2d), you specify the output size (e.g., (H, W), here: (1, 1)), 
            # # and PyTorch automatically computes the required kernel size and stride to achieve that size
            # In: [batch_size, max_feature_maps=512, size_min_feature_maps=4, size_min_feature_maps=4]
            nn.AdaptiveAvgPool2d(1),  # Out: [batch_size, max_feature_maps, 1, 1]
            nn.Flatten(),             # Out: [batch_size, max_feature_maps]
            nn.Linear(self.crit_chan_per_layer[-1], 1),  # Out: [batch_size, 1]
        )
    
    # Returns attention maps from all attention layers
    def get_attention_maps(self, x):
        if not self.use_attention:
            return []
            
        attention_maps = []
        blocks = [
            self.conv_block_1, self.conv_block_2, self.conv_block_3,
            self.conv_block_4, self.conv_block_5, self.conv_block_6,
            self.conv_block_7
        ]
        
        for block in blocks:
            for layer in block:
                if isinstance(layer, SelfAttention):
                    attention_maps.append(layer.get_attention_map(x))
                x = layer(x)
                
        return attention_maps

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
                x.shape[1] == self.crit_chan_per_layer[0] and 
                x.shape[2] == 256 and 
                x.shape[3] == 256)
        # Out: [batch_size, ch, 256, 256]

        x = self.conv_block_2(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.crit_chan_per_layer[1] and 
                x.shape[2] == 128 and 
                x.shape[3] == 128)
        # Out: [batch_size, ch, 128, 128]

        x = self.conv_block_3(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.crit_chan_per_layer[2] and 
                x.shape[2] == 64 and 
                x.shape[3] == 64)
        # Out: [batch_size, ch, 64, 64]

        x = self.conv_block_4(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.crit_chan_per_layer[3] and 
                x.shape[2] == 32 and 
                x.shape[3] == 32)
        # Out: [batch_size, ch, 32, 32]

        x = self.conv_block_5(x)
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.crit_chan_per_layer[4] and 
                x.shape[2] == 16 and 
                x.shape[3] == 16)
        # Out: [batch_size, ch, 16, 16]

        x = self.conv_block_6(x) 
        assert (x.shape[0] <= self.batch_size and 
                x.shape[1] == self.crit_chan_per_layer[5] and 
                x.shape[2] == 8 and 
                x.shape[3] == 8)
        # Out: [batch_size, ch, 8, 8]

        x = self.conv_block_7(x) 
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
