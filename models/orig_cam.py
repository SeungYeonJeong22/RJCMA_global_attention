from __future__ import absolute_import
from __future__ import division

from torch.nn import init
import torch
import math
from torch import nn
from torch.nn import functional as F
import sys
from .av_crossatten import DCNLayer
from .layer import LSTM
from copy import deepcopy

from .audguide_att import BottomUpExtract as AVGA
import torch
import torch.nn as nn
import math
import sys

sys.path.append('./models')
from mwt import MWTF

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cosine for odd indices
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# class MultiWindowTrasnformer(nn.Module):
#     def __init__(self, d_model=512, nhead=4, num_encoder_layers=4, num_decoder_layers=6, dropout=0.6):
#         super(MultiWindowTrasnformer, self).__init__()
#         self.avga = AVGA(512, 512)
        
#         self.positional_encoding = PositionalEncoding(d_model, dropout)
#         self.audio_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout),
#             num_layers=num_encoder_layers
#         )
#         self.video_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout),
#             num_layers=num_encoder_layers
#         )

#         self.vregressor = nn.Sequential(
#             nn.Linear(d_model, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.6),
#             nn.Linear(128, 1)
#         )



class LSTM_CAM(nn.Module):
    def __init__(self):
        super(LSTM_CAM, self).__init__()
        self.coattn = DCNLayer(512, 512, 1, 0.6)
        self.avga = AVGA(512, 512)

        self.MWT = MWTF(feature_dim=512, temporal_window_lengths=[4,8,16])

        self.audio_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True) # output: (batch, sequence, features)
        self.video_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True) # output: (batch, sequence, features)

        self.vregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        self.Joint = LSTM(1024, 512, 2, dropout=0, residual_embeddings=True)

        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        self.init_weights()

    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


    def forward(self, f1_norm, f2_norm):
        video = F.normalize(f2_norm, dim=-1)
        audio = F.normalize(f1_norm, dim=-1)

        audio = self.MWT(audio)
        video = self.avga(video, audio)
        video = self.MWT(video)
        
        # # Tried with LSTMs also
        # audio = self.audio_extract(audio)
        # video = self.video_attn(video, audio)
        # video = self.video_extract(video)

        video, audio = self.coattn(video, audio)

        audiovisualfeatures = torch.cat((video, audio), -1)
        
        audiovisualfeatures = self.Joint(audiovisualfeatures)
        vouts = self.vregressor(audiovisualfeatures) #.transpose(0,1))
        aouts = self.aregressor(audiovisualfeatures) #.transpose(0,1))

        return vouts.squeeze(2), aouts.squeeze(2)  #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)