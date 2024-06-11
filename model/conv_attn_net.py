import torch
import torch.nn as nn
import torch.nn.functional as F


class EncConvAttnMLP(nn.Module):
    def __init__(self, patch_size, ic, oc, nhead, mlp_ratio):
        super(EncConvAttnMLP, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=patch_size, stride=patch_size)
        self.conv_act = nn.ReLU()
        self.bn = nn.BatchNorm2d(oc)
        self.attn = nn.MultiheadAttention(embed_dim=oc, num_heads=nhead)
        self.ln_attn = nn.LayerNorm(oc)
        self.mlp_up = nn.Linear(oc, oc*mlp_ratio)
        self.mlp_act = nn.ReLU() 
        self.mlp_down = nn.Linear(oc*mlp_ratio, oc)
        self.ln_mlp = nn.LayerNorm(oc)

    def forward(self, x):
        x = self.conv_act(self.bn(self.conv(x)))
        x = x.permute(0, 2, 3, 1) # move channel to the last dimension
        bs, h, w, oc = x.shape
        x = x.reshape(bs, -1, oc) # put tokens in linear fashion

        residual = x
        x = self.ln_attn(x)
        x = self.attn(x, x, x, need_weights=False)
        x = residual + x[0]
        residual = x
        x = self.ln_mlp(x)
        x = self.mlp_down(self.mlp_act(self.mlp_up(x)))
        x = residual + x
        x = x.reshape(bs, h, w, oc).permute(0, 3, 1, 2) # putting back into spatial layout so it works with next block of the same type
        return x

class DecConvAttnMLP(nn.Module):
    def __init__(self, patch_size, ic, oc, nhead, mlp_ratio):
        super(DecConvAttnMLP, self).__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim=ic, num_heads=nhead)
        self.ln_attn = nn.LayerNorm(ic)
        self.mlp_up = nn.Linear(ic, ic*mlp_ratio)
        self.mlp_act = nn.ReLU() 
        self.mlp_down = nn.Linear(ic*mlp_ratio, ic)
        self.ln_mlp = nn.LayerNorm(ic)

        self.upconv = nn.ConvTranspose2d(in_channels=ic, out_channels=oc, kernel_size=patch_size, stride=patch_size) #assume always merging 2x2 neighbouring pixel in non-overlapping fashion
        self.upconv_act = nn.ReLU()
        self.bn = nn.BatchNorm2d(oc)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, c)
        residual = x
        x = self.ln_attn(x)
        x = self.attn(x, x, x, need_weights=False)
        x = residual + x[0]
        residual = x
        x = self.ln_mlp(x)
        x = self.mlp_down(self.mlp_act(self.mlp_up(x)))
        x = residual + x

        x = x.reshape(bs, h, w, c)
        x = x.permute(0, 3, 1, 2)          
        x = self.upconv_act(self.bn(self.upconv(x)))
        return x


class ConvAttnNet(nn.Module):
    def __init__(self, 
                 M=4, Nc=32, Nt=32,
                 data_channel=2,
                 patch_size=2,
                 oc1=64, nhead1=4, mlp_ratio1=4,
                 oc2=128, nhead2=4, mlp_ratio2=4):
        super(ConvAttnNet, self).__init__()

        compressed_dim = data_channel*Nc*Nt//M

        # Encoder
        self.encoder_blk1 = EncConvAttnMLP(patch_size=patch_size, ic=data_channel, oc=oc1, nhead=nhead1, mlp_ratio=mlp_ratio1)
        self.encoder_blk2 = EncConvAttnMLP(patch_size=patch_size, ic=oc1,          oc=oc2, nhead=nhead2, mlp_ratio=mlp_ratio2)

        spatial_dim = Nc//patch_size//patch_size
        self.encoder_conv = nn.Conv2d(in_channels=oc2, out_channels= compressed_dim//spatial_dim//spatial_dim,
                                      kernel_size=spatial_dim, stride=1)  

        # Decoder
        self.decoder_upconv = nn.ConvTranspose2d(in_channels=compressed_dim//spatial_dim//spatial_dim, out_channels=oc2,
                                             kernel_size=spatial_dim, stride=1)
        self.decoder_blk2 = DecConvAttnMLP(patch_size=patch_size, ic=oc2,          oc=oc1, nhead=nhead2, mlp_ratio=mlp_ratio2)
        self.decoder_blk1 = DecConvAttnMLP(patch_size=patch_size, ic=oc1, oc=data_channel, nhead=nhead1, mlp_ratio=mlp_ratio1)

    def forward(self, x):
        # compression
        encoded_x = self.encoder_blk2(self.encoder_blk1(x))

        codeword = self.encoder_conv(encoded_x)
        decoded_word = self.decoder_upconv(codeword)

        reconstructed_x = self.decoder_blk1(self.decoder_blk2(decoded_word))
        
        return codeword, reconstructed_x
    

            # bs, c, h, w = encoded_x.shape
        # codeword  = self.encoder_fc(encoded_x.reshape(bs, -1))
                # .reshape(bs, c, h, w)