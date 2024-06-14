import torch
import torch.nn as nn
import math
#from .transformer_block import TransformerBlock
from collections import OrderedDict
import json
# class PositionalEmbedding(nn.Module):
#     def __init__(self, num_positions, emb_size):
#         super(PositionalEmbedding, self).__init__()
#         self.register_buffer('pe', self.initialize_pe(num_positions, emb_size))

#     def initialize_pe(self, num_positions, emb_size):
#         pe = torch.zeros(num_positions, emb_size)
#         position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)

#     def forward(self, x):
#         bs, oc, h, w = x.shape
#         x = x.permute(0, 2, 3, 1)
#         x = x.reshape(bs, -1, oc) # put tokens in linear fashion
    
#         x = x + self.pe[:, :x.size(1), :]

#         x = x.reshape(bs, h, w, oc).permute(0, 3, 1, 2)
#         return x


class DownSampleConvTX(nn.Module):
    def __init__(self, emb_in, emb_out, patch_size, tx_nhead, tx_ffn_ratio, n_tx, dropout_rate):
        super(DownSampleConvTX, self).__init__()
        self.down_conv = nn.Conv2d(emb_in, emb_out, kernel_size=(patch_size, patch_size), stride=patch_size)

        # txblk=[]
        # for _ in range(n_tx):
        #     txblk.append(TransformerBlock(emb_dim=emb_out, num_heads=tx_nhead, expansion_ratio=tx_ffn_ratio, dropout_rate=dropout_rate))
        # self.txblk = nn.Sequential(*txblk)

    def forward(self, x):
        # input and output of this loop is standardized as spatial tensor
        x = self.down_conv(x)

        bs, oc, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, oc) # layout as a series of tokens
    
        #x = self.txblk(x)

        x = x.reshape(bs, h, w, oc).permute(0, 3, 1, 2) # layout as spatial tensor
        return x

class Encoder(nn.Module):
    def __init__(self, data_hw, data_channel, cr, stage1_emb, stage2_emb, patch_size, n_tx, tx_nhead, tx_ffn_ratio, dropout_rate):
        super(Encoder, self).__init__()
        self.downsample_s1 = DownSampleConvTX(
            emb_in=data_channel, 
            emb_out=stage1_emb, 
            patch_size=patch_size,
            tx_nhead=4,
            tx_ffn_ratio=tx_ffn_ratio,
            n_tx=n_tx,
            dropout_rate=dropout_rate)

        self.downsample_s2 = DownSampleConvTX(
            emb_in=stage1_emb, 
            emb_out=stage2_emb, 
            patch_size=patch_size,
            tx_nhead=4,
            tx_ffn_ratio=tx_ffn_ratio,
            n_tx=n_tx,
            dropout_rate=dropout_rate)

        s2_spatial_dim = data_hw // patch_size // patch_size
        self.fc_compress = nn.Linear(s2_spatial_dim**2 * stage2_emb, int(data_hw**2 * data_channel / cr))

    def forward(self, x):
        x = self.downsample_s2(
                self.downsample_s1(x)
                )

        codeword = self.fc_compress(x.reshape(x.shape[0], -1))
        return codeword


class UpSampleConvTX(nn.Module):
    def __init__(self, emb_in, emb_out, patch_size, tx_nhead, tx_ffn_ratio, n_tx, dropout_rate):
        super(UpSampleConvTX, self).__init__()

        #txblk=[]
        #for _ in range(n_tx):
            #txblk.append(TransformerBlock(emb_dim=emb_in, num_heads=tx_nhead, expansion_ratio=tx_ffn_ratio, dropout_rate=dropout_rate))
        #self.txblk = nn.Sequential(*txblk)

        self.up_conv = nn.ConvTranspose2d(emb_in, emb_out, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # input and output of this loop is standardized as spatial tensor
        bs, oc, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, oc) # layout as a series of tokens
    
        #x = self.txblk(x)

        x = x.reshape(bs, h, w, oc).permute(0, 3, 1, 2) # layout as spatial tensor

        x = self.up_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, data_hw, data_channel, cr, stage1_emb, stage2_emb, patch_size, n_tx, tx_nhead, tx_ffn_ratio, dropout_rate):
        super(Decoder, self).__init__()
        self.stage2_emb = stage2_emb
        self.s2_spatial_dim = data_hw // patch_size // patch_size 
        self.fc_decompress = nn.Linear(int(data_hw**2 * data_channel / cr), self.s2_spatial_dim**2 * stage2_emb)
        
        self.upsample_s2 = UpSampleConvTX(
            emb_in=stage2_emb, 
            emb_out=stage1_emb, 
            patch_size=patch_size,
            tx_nhead=4,
            tx_ffn_ratio=tx_ffn_ratio,
            n_tx=n_tx,
            dropout_rate=dropout_rate)
        
        self.upsample_s1 = UpSampleConvTX(
            emb_in=stage1_emb, 
            emb_out=data_channel, 
            patch_size=patch_size,
            tx_nhead=4,
            tx_ffn_ratio=tx_ffn_ratio,
            n_tx=n_tx,
            dropout_rate=dropout_rate)
        
    def forward(self, codeword):
        x = self.fc_decompress(codeword.reshape(codeword.shape[0], -1))
        x = x.reshape(codeword.shape[0], self.stage2_emb, self.s2_spatial_dim, self.s2_spatial_dim)

        reconstructed_output = self.upsample_s1(
                self.upsample_s2(x)
                )
        return reconstructed_output

class SandGlassNet(nn.Module):
    def __init__(self, data_hw, data_channel, cr, stage1_emb, stage2_emb, patch_size, n_tx, tx_nhead, tx_ffn_ratio, dropout_rate):
        super(SandGlassNet, self).__init__()
        self.cr = cr
        self.code_dim = data_hw**2 * data_channel // self.cr
        assert data_hw**2 * data_channel % self.cr == 0, f"data dim is not divisible by cr={self.cr}, pls revise compression ratio"
        
        model_arch = OrderedDict(
            data_hw = data_hw, 
            data_channel = data_channel, 
            cr = cr, 
            stage1_emb = stage1_emb, 
            stage2_emb = stage2_emb, 
            patch_size = patch_size, 
            n_tx = n_tx, 
            tx_nhead = tx_nhead, 
            tx_ffn_ratio = tx_ffn_ratio, 
            dropout_rate = dropout_rate
        )

        #self.pos_embed = PositionalEmbedding(data_hw**2, data_channel)
        # !!! Important encoder and decoder is mirrored
        # therefore, decoder start with s2
        self.encoder = Encoder(data_hw, data_channel, cr, stage1_emb, stage2_emb, patch_size, n_tx, tx_nhead, tx_ffn_ratio, dropout_rate)
        self.decoder = Decoder(data_hw, data_channel, cr, stage1_emb, stage2_emb, patch_size, n_tx, tx_nhead, tx_ffn_ratio, dropout_rate)

        print(f"SandGlassNet Params:\n{json.dumps(model_arch, indent=True)}")

    def forward(self, x):
        #x = self.pos_embed(x)

        codeword = self.encoder(x)
        reconstructed_output = self.decoder(codeword)
        return codeword, reconstructed_output