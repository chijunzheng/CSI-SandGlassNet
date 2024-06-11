import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.emb = nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size, patch_size), stride=patch_size)

    def forward(self, x):
        x = self.emb(x)  # [batch_size, emb_size, num_patches, num_patches]
        x = x.flatten(2)  # Flatten the patches
        return x.transpose(1, 2)  # [batch_size, num_patches, emb_size]

class PositionalEmbedding(nn.Module):
    def __init__(self, num_positions, emb_size):
        super(PositionalEmbedding, self).__init__()
        self.register_buffer('pe', self.initialize_pe(num_positions, emb_size))

    def initialize_pe(self, num_positions, emb_size):
        pe = torch.zeros(num_positions, emb_size)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm)
        x = attn_output + x
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = ff_output + x
        return x

class PatchMerging(nn.Module):
    def __init__(self, emb_size_1, emb_size_2):
        super(PatchMerging, self).__init__()
        self.emb_size_1 = emb_size_1
        self.emb_size_2 = emb_size_2
        self.conv = nn.Conv2d(in_channels=emb_size_1, out_channels=emb_size_2, kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, seq_len, emb_size = x.shape
        side_length = int(math.sqrt(seq_len))
        if side_length * side_length != seq_len:
            raise ValueError("Sequence length is not a perfect square")

        # Reshape and permute to prepare for convolution
        x = x.view(batch_size, side_length, side_length, emb_size).permute(0, 3, 1, 2)  # [batch_size, emb_size, side_length, side_length]
        x = self.conv(x)  # [batch_size, emb_size_2, new_side_length, new_side_length]

        # Get the new side length
        new_side_length = x.size(2)

        # Permute and reshape to match the expected output
        x = x.permute(0, 2, 3, 1).reshape(batch_size, new_side_length * new_side_length, self.emb_size_2)  # [batch_size, new_seq_len, emb_size_2]
        return x

class PatchDivision(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchDivision, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        if len(x.shape) == 4:
            batch_size, channels, height, width = x.shape
        elif len(x.shape) == 3:
            batch_size, new_seq_len, emb_size = x.shape
            side_length = int(math.sqrt(new_seq_len))
            x = x.view(batch_size, side_length, side_length, emb_size).permute(0, 3, 1, 2)
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions")
        
        # Perform the deconvolution
        x = self.deconv(x)  # [batch_size, out_channels, new_side_length, new_side_length]
        
        # Get the new side length
        new_side_length = x.size(2)
        
        # Permute and reshape to match the expected output
        x = x.permute(0, 2, 3, 1).reshape(batch_size, new_side_length * new_side_length, x.size(1))  # [batch_size, new_seq_len, out_channels]
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size_1, emb_size_2, img_size, num_heads, ff_dim, dropout_rate, compression_ratio):
        super(Encoder, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size_1)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(num_patches, emb_size_1)
        self.transformer_block1 = TransformerBlock(emb_size_1, num_heads, ff_dim, dropout_rate)
        self.patch_merge = PatchMerging(emb_size_1, emb_size_2)
        self.transformer_block2 = TransformerBlock(emb_size_2, num_heads, ff_dim, dropout_rate)
        self.fc_compress = nn.Linear(emb_size_2 * 8 * 8, int(in_channels * img_size**2 * compression_ratio))

    def forward(self, x):
        x = self.patch_embed(x)  # Shape: [batch_size, num_patches, emb_size_1]
        x = self.pos_embed(x)  # Add positional embeddings
        x = self.transformer_block1(x)
        x = self.patch_merge(x)  # Shape: [batch_size, new_num_patches, emb_size_2]
        x = self.transformer_block2(x)
        codeword = self.fc_compress(x.view(x.size(0), -1))
        return codeword

class Decoder(nn.Module):
    def __init__(self, emb_size_1, emb_size_2, img_size, in_channels, num_heads, ff_dim, dropout_rate, compression_ratio):
        super(Decoder, self).__init__()
        self.fc_expand = nn.Linear(int(in_channels * img_size**2 * compression_ratio), emb_size_2 * 8 * 8)
        self.transformer_block3 = TransformerBlock(emb_size_2, num_heads, ff_dim, dropout_rate)
        self.patch_divide1 = PatchDivision(emb_size_2, emb_size_1)
        self.transformer_block4 = TransformerBlock(emb_size_1, num_heads, ff_dim, dropout_rate)
        self.patch_divide2 = nn.ConvTranspose2d(emb_size_1, 2, kernel_size=2, stride=2)  # Corrected this line
        self.emb_size_1 = emb_size_1
        self.emb_size_2 = emb_size_2
        
    def forward(self, x):
        x = self.fc_expand(x).view(x.size(0), self.emb_size_2, 8, 8).permute(0, 2, 3, 1).reshape(x.size(0), 64, self.emb_size_2)
        x = self.transformer_block3(x)
        x = self.patch_divide1(x.view(x.size(0), 64, self.emb_size_2))
        x = self.transformer_block4(x.view(x.size(0), 256, self.emb_size_1))
        x = x.view(x.size(0), 16, 16, self.emb_size_1).permute(0, 3, 1, 2)  # [batch_size, emb_size_1, 16, 16]
        reconstructed_output = self.patch_divide2(x)  # [batch_size, 2, 32, 32]
        return reconstructed_output

class SandGlassNet(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size_1, emb_size_2, img_size, num_heads, ff_dim, dropout_rate, compression_ratio):
        super(SandGlassNet, self).__init__()
        self.encoder = Encoder(in_channels, patch_size, emb_size_1, emb_size_2, img_size, num_heads, ff_dim, dropout_rate, compression_ratio)
        self.decoder = Decoder(emb_size_1, emb_size_2, img_size, in_channels, num_heads, ff_dim, dropout_rate, compression_ratio)

    def forward(self, x):
        codeword = self.encoder(x)
        reconstructed_output = self.decoder(codeword)
        return codeword, reconstructed_output



# #Hyperparameters
# epochs = 1000
# learning_rate = 0.0001
# batch_size = 200
# img_size = 32
# patch_size = 2
# in_channels = 2
# emb_size_1 = 64
# emb_size_2 = 128
# num_heads = 4
# ff_dim = 256
# dropout_rate = 0.1
# compression_ratio = 0.25

# # Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define Model
# model = NewCsiNet(in_channels, patch_size, emb_size_1, emb_size_2, img_size, num_heads, ff_dim, dropout_rate, compression_ratio)
# model.to(device)