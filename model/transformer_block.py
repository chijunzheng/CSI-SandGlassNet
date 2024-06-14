import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim, expansion_ratio, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.expansion_ratio = expansion_ratio
        self.emb_dim = emb_dim
        self.dff = emb_dim*expansion_ratio
        self.up = nn.Linear(self.emb_dim, self.dff)
        self.do_up = nn.Dropout(dropout_rate)
        self.down = nn.Linear(self.dff, self.emb_dim)
        self.do_down = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.do_down(self.down(
                    self.do_up(F.relu(self.up(x)))
                ))

# class TransformerBlock(nn.Module):
#     def __init__(self, emb_dim, num_heads, expansion_ratio, dropout_rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout_rate)
#         self.ffn = FeedForwardNetwork(emb_dim, expansion_ratio, dropout_rate)

#         self.ln_mha = nn.LayerNorm(emb_dim)
#         self.ln_ffn = nn.LayerNorm(emb_dim)

#     def forward(self, x, mask=None):
#         # we use pre-LN transformer layer
#         # see paper: On Layer Normalization in the Transformer Architecture
#         residual = x
#         x_norm = self.ln_mha(x)
#         attn_output, _ = self.mha(x_norm, x_norm, x_norm)
#         out1 = residual + attn_output

#         residual = out1
#         ffn_output = self.ffn(self.ln_mha(out1))
#         out2 = residual + ffn_output

#         return out2

# # Example usage:
# emb_dim = 512
# num_heads = 8
# dff = 2048
# dropout_rate = 0.1

# transformer_block = TransformerBlock(emb_dim, num_heads, dff, dropout_rate)

# # Dummy input
# x = torch.rand(64, 10, emb_dim)  # (batch_size, sequence_length, emb_dim)
# mask = None

# # Forward pass
# output = transformer_block(x, mask)
# print(output.shape)  # Expected output: torch.Size([64, 10, 512])
