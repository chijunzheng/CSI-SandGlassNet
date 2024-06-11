import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineNetBlock(nn.Module):
    def __init__(self, in_channels, mid_channel1, mid_channel2, out_channels):
        super(RefineNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channel1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channel1)
        self.conv2 = nn.Conv2d(mid_channel1, mid_channel2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel2)
        self.conv3 = nn.Conv2d(mid_channel2, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        #Save the input for the skip connection
        shortcut = x

        #Apply the RefineNet block Operations
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.3)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.3)
        x = self.bn3(self.conv3(x))

        #Add the skip connection
        x += shortcut

        #Apply activation after addition
        x = F.leaky_relu(x, negative_slope=0.3)
        return x

class CsiNet(nn.Module):
    def __init__(self, M=512, Nc=32, Nt=32):
        super(CsiNet, self).__init__()

        self.Nc = Nc
        self.Nt = Nt
        # Encoder
        self.encoder_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.encoder_bn = nn.BatchNorm2d(2)
        self.encoder_fc = nn.Linear(2 * Nc * Nt, M) # Find out why parameter size is 2*Nt*Nc*M

        # Decoder
        self.decoder_fc = nn.Linear(M, 2 * Nc * Nt)
        self.refinenet_block1 = RefineNetBlock(2, 8, 16, 2)
        self.refinenet_block2 = RefineNetBlock(2, 8, 16, 2)
        self.final_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = F.leaky_relu(self.encoder_bn(self.encoder_conv(x)), negative_slope=0.3)
        x = x.view(x.size(0), -1)  # Flatten
        codeword = self.encoder_fc(x)

        # Decoder
        x = self.decoder_fc(codeword)
        x = x.view(-1, 2, self.Nc, self.Nt)  # Reshape to spatial dimensions

        # RefineNet blocks
        x = self.refinenet_block1(x)
        x = self.refinenet_block2(x)

        # Final conv layer
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return codeword, x