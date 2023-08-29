import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .model_EmbedLayer import DataEmbedding, DataEmbedding_Value
from tkinter import _flatten

from src.model_EmbedLayer import DataEmbedding_Value
from src.model_EncoderLayer import TCN_Encoder, Trans_Encoder
from src.model_DecoderLayer import Reconstruct_Decoder_MLP, Projector, Point_Predictor, Context_Predictor

from einops import rearrange
from tkinter import _flatten
import random


class TSAD(nn.Module):
    def __init__(self, args):
        super(TSAD, self).__init__()

        self.args = args
        self.tsnorm = RevIN(num_features=args.tsdim)
        self.ts_embed = DataEmbedding_Value(args.tsdim, args.tsembeddim)
        if args.encoder_select == "tcn":
            self.encoder_block = TCN_Encoder(in_channels=args.tsembeddim,
                                                 tcnblockdim=args.tcnblockdim,
                                                 out_channels=args.representation_dim,
                                                 depth=args.tcndepth,
                                                 kernel_size=args.tcnkernelsize,
                                                 inputmode=args.tcninputmode)
        elif args.encoder_select == "trans":
            self.encoder_block = Trans_Encoder(in_channels=args.tsembeddim,
                                                     in_length=args.trans_in_length,
                                                     n_heads=args.trans_n_heads,
                                                     n_layers=args.trans_n_layers,
                                                     out_channels=args.representation_dim,
                                                     transblockdim=args.trans_block_dim)


        self.decoder = Reconstruct_Decoder_MLP(args.representation_dim, args.tsdim)
        self.projector = Projector(args.representation_dim, args.representation_dim)
        self.context_predictor = Context_Predictor(embed_dim=args.representation_dim, num_heads=4)
        # self.context_predictor_2 = Context_Predictor(embed_dim=args.representation_dim, num_heads=4)
        # self.context_predictor_3 = Context_Predictor(embed_dim=args.representation_dim, num_heads=4)
        self.point_predictor = Point_Predictor(args.representation_dim, args.representation_dim)
        self.reduce_dim_layer = nn.AdaptiveAvgPool1d(output_size=1) # Global Average Pooling

        self.mask_r = 20
        

    def forward(self, x):
        """x: [B, L, C]"""
        
        x_raw = x
        # print("---->xraw", x_raw.shape)
        x_norm = self.tsnorm(x_raw, 'norm')

        z = self.ts_embed(x_norm) # embedding, z [B, C, L]

        if self.args.encoder_select == "trans":
            encoder_out = self.encoder_block(z.permute(0, 2, 1)) # tcn_out [out1, out2, out3, ...], by num_layers, B,L,C
        else:
            encoder_out = self.encoder_block(z).permute(0, 2, 1) # trans_out [out1, out2, out3, ...], by num_layers, B,L,C
        encoder_out_raw = encoder_out # B L D

        """GLobal Rescontruct, AE """
        """Maybe rec is not a good choince, using 使用一个点个周围点的相似度可能更合适？"""
        x_rec = self.decoder(encoder_out)

        """Local Prediction, """
        B = encoder_out.shape[0]
        L = encoder_out.shape[1]
        D = encoder_out.shape[2]
        encoder_out = [encoder_out]*L
        encoder_out = torch.stack(encoder_out, dim=1) # b l c -> b l l c
        point_masks = self.neighbor_mask(encoder_out_raw, r=0)
        neighbor_masks = self.neighbor_mask(encoder_out_raw, r=self.mask_r)
        target = encoder_out.masked_select(~point_masks).view(B, L, 1, D)
        inputs_long = encoder_out.masked_select(point_masks).view(B, L, L-1, D)
        inputs_short = encoder_out.masked_select(neighbor_masks).view(B, L, L-self.mask_r*2-1, D)

        target = rearrange(target, "b l h d -> (b l) h d").squeeze()
        inputs_long = rearrange(inputs_long, "b l h d -> (b l) h d")
        inputs_short = rearrange(inputs_short, "b l h d -> (b l) h d")

        context_short, _ = self.context_predictor(inputs_short, inputs_short, inputs_short)
        context_long, _ = self.context_predictor(inputs_long, inputs_long, inputs_long)
        # context out
        context_short = self.reduce_dim_layer(context_short.transpose(1,2)).squeeze()
        context_long = self.reduce_dim_layer(context_long.transpose(1,2)).squeeze()
        # cross prediction
        point_out_short = self.point_predictor(context_short)
        point_out_long = self.point_predictor(context_long)
        point_real = target

        """Mask Local Prediction"""
        context_rep = [context_short, context_long]
        predicted_rep = [point_out_short, point_out_long]
        true_rep = point_real

        return x_raw, x_rec, encoder_out_raw, true_rep, predicted_rep, context_rep

    def prediction_mask(self, x):
        B, L, D = x.shape
        predict_mask = torch.ones((L, L, D), dtype=bool, device=x.device)
        for i in range(L):
            predict_mask[i, i, :] = 0
        predict_mask = [predict_mask]*B
        predict_mask = torch.stack(predict_mask, dim=0)
        return predict_mask

    def neighbor_mask(self, x, r = 10):
        if r == 0:
            predict_mask = self.prediction_mask(x)
            return predict_mask
        else:
            B, L, D = x.shape
            predict_mask = torch.ones((L, L, D), dtype=bool, device=x.device)
            for i in range(L):
                if i < r:
                    p = 2*r-i
                    predict_mask[i, 0:i+p+1, :] = 0
                elif i > L-r-1:
                    p = r+1-(L-i)
                    predict_mask[i, i-r-p:, :] = 0
                else:
                    predict_mask[i, i-r:i+r+1, :] = 0
            predict_mask = [predict_mask]*B
            predict_mask = torch.stack(predict_mask, dim=0)
            return predict_mask

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight=self.affine_weight.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.affine_bias=self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

