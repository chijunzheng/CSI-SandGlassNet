# Note: diff csinet_refinetx.py and csinet_encdec_refinetx.py to see changes to refinenet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock

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

# RefineNet is modified to use stack of transformer block
class RefineTX(nn.Module):
    def __init__(self, patch_size, ifm, d_model, nhead, ffn_ratio, n_block):
        super(RefineTX, self).__init__()
        
        # input to refinenet was a spatial feature map of #ifm channel
        # we applied non-overlapping convolution to patchify the feature map to tokens
        self.conv = nn.Conv2d(in_channels=ifm, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm2d(d_model)

        # 
        tx_layers=[]
        for _ in range(n_block):
            tx_layers.append(TransformerBlock(d_model=d_model, num_heads=nhead, expansion_ratio=ffn_ratio))
        self.tx_layers = nn.Sequential(*tx_layers)

    def forward(self, x):
        # tokenize by non-overlapping convolution, followed by batch normalization and activation
        x = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.3)
        
        # reorganize spatial tensor to series of tokens
        x = x.permute(0, 2, 3, 1) # move channel to the last dimension
        bs, h, w, oc = x.shape
        x = x.reshape(bs, -1, oc) # put tokens in linear fashion

        # feed tokens to transformer layers
        x = self.tx_layers(x)

        # putting back into spatial layout so it works with next block of the same type
        x = x.reshape(bs, h, w, oc).permute(0, 3, 1, 2) 
        return x
    

class CsiNet_EncDecTX(nn.Module):
    def __init__(self, M=512, Nc=32, Nt=32):
        super(CsiNet_EncDecTX, self).__init__()

        self.Nc = Nc
        self.Nt = Nt
        
        # Encoder
        # !!! We directly tokenize the input with RefineTX (4x4 patch as a token)
        # self.encoder_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        # self.encoder_bn = nn.BatchNorm2d(2)
        self.enc_refinetx = RefineTX(patch_size=4, ifm=2, d_model=64, nhead=4, ffn_ratio=4, n_block=1)
        self.enc_upconv = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=4, stride=4) 

        self.encoder_fc = nn.Linear(2 * Nc * Nt, M) # Find out why parameter size is 2*Nt*Nc*M

        # Decoder
        self.decoder_fc = nn.Linear(M, 2 * Nc * Nt)

        # !!! We do not use refinenet block
        # self.refinenet_block1 = RefineNetBlock(2, 8, 16, 2)
        # self.refinenet_block2 = RefineNetBlock(2, 8, 16, 2)

        # !!! we use our new Refine network that uses convolution and transformer layers
        self.dec_refinetx = RefineTX(patch_size=4, ifm=2, d_model=64, nhead=4, ffn_ratio=4, n_block=3)
        
        # !!! final_conv is changed to Convtransposed2d to upsample feature map to match required.
        # self.final_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.dec_upconv = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=4, stride=4) 
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        # x = F.leaky_relu(self.encoder_bn(self.encoder_conv(x)), negative_slope=0.3)
        x = self.enc_refinetx(x)
        x = self.enc_upconv(x)

        x = x.view(x.size(0), -1)  # Flatten
        codeword = self.encoder_fc(x)

        # Decoder
        x = self.decoder_fc(codeword)
        x = x.view(-1, 2, self.Nc, self.Nt)  # Reshape to spatial dimensions

        # New transformer-based RefineNet
        x = self.dec_refinetx(x)
        x = self.dec_upconv(x)

        # Final conv layer
        x = self.sigmoid(x)
        return codeword, x