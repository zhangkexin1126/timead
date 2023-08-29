import torch
import torch.nn as nn
import torch.nn.functional as F

from src import net_tcn
from src import net_trans

class TCN_Encoder(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of save channels.
    @param depth Depth of the network.
    @param out_channels Number of save channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, tcnblockdim, depth, out_channels, kernel_size, inputmode):
        super(TCN_Encoder, self).__init__()

        self.inputmode = inputmode

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size, default = 1

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else tcnblockdim
            layers += [net_tcn.CausalConvolutionBlock(
                in_channels_block, tcnblockdim, kernel_size, dilation_size
            )]
            # dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [net_tcn.CausalConvolutionBlock(
            tcnblockdim, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)
        self.N_LAYER = len(layers)


    def forward(self, x):
        """Require input: BCL"""
        out = self.network(x)
        return out
    
    # def forward(self, x):
    #     """Input X: BCL"""
    #     out = []
    #     for k in range(self.N_LAYER):
    #         x = self.network[k](x)
    #         out.append(x)
    #     return out
    
    # def forward_last(self, x):
    #     if self.inputmode == "blc":
    #         x = x.permute(0, 2, 1)
    #         out = self.forward_layer(x)
    #         # out = out.permute(0, 2, 1)  # return to B, L, out_dim
    #     elif self.inputmode == "bcl":
    #         out = self.network(x)
    #     return out


class Trans_Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 in_length,
                 n_heads,
                 n_layers,
                 out_channels,
                 transblockdim=256,
                 dropout=0.1,
                 pos_encoding='fixed',
                 activation='gelu',
                 norm='BatchNorm',
                 ):
        super(Trans_Encoder, self).__init__()

        # d_model, dropout=0.1, max_len=1024, scale_factor=1.0
        # position encoding
        self.pos_enc = net_trans.get_pos_encoder(pos_encoding)(in_channels, max_len=in_length)
        if norm == 'LayerNorm':
            encoder_layer = net_trans.TransformerLayerNormEncoderLayer(d_model=in_channels,
                                                       nhead=n_heads,
                                                       dim_feedforward=transblockdim,
                                                       dropout=dropout,
                                                       activation=activation)
        elif norm == 'BatchNorm':
            encoder_layer = net_trans.TransformerBatchNormEncoderLayer(d_model=in_channels,
                                                       nhead=n_heads,
                                                       dim_feedforward=transblockdim,
                                                       dropout=dropout,
                                                       activation=activation)

        self.transformer_encoder = net_trans.TransformerBlock(encoder_layer=encoder_layer, num_layers=n_layers)
        self.projection = nn.Linear(in_channels, out_channels, bias=True)


    def forward(self, x, key_padding_masks=None, src_masks=None):
        '''
        :param ts: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
        :return:
        '''
        out = self.pos_enc(x)  # add positional encoding
        out = self.transformer_encoder(out, src_key_padding_mask=key_padding_masks, src_mask=src_masks)  # (seq_length, batch_size, d_model)
        out = self.projection(out)


        # out = self.act(out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        # out = out.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        # out = self.dropout_module(out)  # (batch_size, seq_length, d_model)
        # out = out.permute(0, 2, 1)
        # #out = out.reshape(out.shape[0], -1)
        # #print(out.shape)
        # out = self.pooling_layer(out).squeeze()
        # #print(out.shape)
        # pred = self.predict_layer(out)
        return out