import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
import numpy as np


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 return_attn=False):
        super(LoFTREncoderLayer, self).__init__()

        self.return_attn = return_attn

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def get_attn_dot_product(self, query, key):
        Q = query.transpose(1,2)
        K = key.transpose(1,2) # 1 x 8 x 4800 x 32
        #print(Q.shape)
        #print(K.transpose(-1, -2).shape) # 1 x 8 x 32 x 4800
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim)
        #print(scores.shape)
        #attn = nn.Softmax(dim=-1)(scores)
        attn = scores
        #print(attn.shape)
        return attn # B x nheads x nPixels_0 x nPixels_1  # 1 x 8 x 4800 x 4800


    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        if self.return_attn:
            attn = self.get_attn_dot_product(query, key)
            return x + message, attn
        else:
            return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config, onnx, return_attn):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.onnx = onnx
        self.return_attn = return_attn
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'], return_attn)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        if not self.onnx: # onnx complains that tensor should not be converted to boolean
            assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        if self.return_attn:
            self_attn0, cross_attn0 = [], []
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0, self_attn0_i = layer(feat0, feat0, mask0, mask0)
                    feat1, _ = layer(feat1, feat1, mask1, mask1)
                    self_attn0.append(self_attn0_i)
                elif name == 'cross':
                    feat0, cross_attn0_i = layer(feat0, feat1, mask0, mask1)
                    feat1, _ = layer(feat1, feat0, mask1, mask0)
                    cross_attn0.append(cross_attn0_i)
                else:
                    raise KeyError
                
            self_attn0 = torch.stack(self_attn0)
            cross_attn0 = torch.stack(cross_attn0)
            self_attn0 = self_attn0.permute([1, 0, 2, 3, 4])
            cross_attn0 = cross_attn0.permute([1, 0, 2, 3, 4])
            
            # Return all layers self and cross attentions
            #print(self_attn0.shape) # B x nLayers x nHeads x nPixels x nPixels  # 1 x 4 x 8 x 4800 x 4800
            #print(cross_attn0.shape)
            return feat0, feat1, self_attn0, cross_attn0

        else:

            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                elif name == 'cross':
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)
                else:
                    raise KeyError

            return feat0, feat1
