import math
import copy
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList


#From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# class FixedPositionalEncoding(nn.Module):
#     r"""Inject some information about the relative or absolute position of the tokens
#         in the sequence. The positional encodings have the same dimension as
#         the embeddings, so that the two can be summed. Here, we use sine and cosine
#         functions of different frequencies.
#     .. math::
#         \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#         \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#         \text{where pos is the word position and i is the embed idx)
#     Args:
#         d_model: the embed dim (required).
#         dropout: the dropout value (default=0.1).
#         max_len: the max. length of the incoming sequence (default=1024).
#     """
#
#     def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
#         super(FixedPositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)  # positional encoding
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
#
#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         """
#         print(x.shape)
#         print(self.pe.shape)
#         print(self.pe[:,:x.size(1), :].shape)
#
#         hh
#
#         x = x + self.pe[:x.size(1), :]
#         return self.dropout(x)

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(x.shape, self.pe.shape)
        x = x + self.pe[:x.size(1), :]
        return x

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(1), :]
        x = x.permute(1, 0, 2)
        return x

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerBlock(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerBlock, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerLayerNormEncoderLayer(nn.modules.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerLayerNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerLayerNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class Transformer_Encoder(nn.Module):
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
        super(Transformer_Encoder, self).__init__()

        # d_model, dropout=0.1, max_len=1024, scale_factor=1.0
        # position encoding
        self.pos_enc = get_pos_encoder(pos_encoding)(in_channels, max_len=in_length)
        if norm == 'LayerNorm':
            encoder_layer = TransformerLayerNormEncoderLayer(d_model=in_channels,
                                                       nhead=n_heads,
                                                       dim_feedforward=transblockdim,
                                                       dropout=dropout,
                                                       activation=activation)
        elif norm == 'BatchNorm':
            encoder_layer = TransformerBatchNormEncoderLayer(d_model=in_channels,
                                                       nhead=n_heads,
                                                       dim_feedforward=transblockdim,
                                                       dropout=dropout,
                                                       activation=activation)

        self.transformer_encoder = TransformerBlock(encoder_layer=encoder_layer, num_layers=n_layers)
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