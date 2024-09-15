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

from .audguide_att import BottomUpExtract

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.6, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Global_Attention_Transformer(nn.Module):
    def __init__(self, dim=256, dropout=0.6, seq_length=16, n_head=8, n_layer=1) -> None:
        super(Global_Attention_Transformer, self).__init__()
        self.first_forward = True

        self.layer = nn.Linear(dim*2, dim)
        
        self.pos_enc = PositionalEncoding(dim, dropout, seq_length)

        enc_layer1 = nn.TransformerEncoderLayer(d_model=dim, nhead=n_head)
        self.encoder1 = nn.TransformerEncoder(enc_layer1, num_layers=n_layer)

        enc_layer2 = nn.TransformerEncoderLayer(d_model=dim, nhead=n_head)
        self.encoder2 = nn.TransformerEncoder(enc_layer2, num_layers=n_layer)        

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        if self.layer.bias is not None:
            nn.init.zeros_(self.layer.bias)        

    def forward(self, data, memory=None):
        if not memory==None:
            data4 = torch.cat((memory, data), dim=-1)
            data3 = self.layer(data4)
            data2 = self.pos_enc(data3)
            data1 = self.encoder2(data2)

            return data1
        else:
            data2 = self.pos_enc(data)
            data1 = self.encoder1(data2)

            return data1


class GAT_LSTM_CAM(nn.Module):
    def __init__(self):
        super(GAT_LSTM_CAM, self).__init__()
        # self.coattn = DCNLayer(512, 512, 2, 0.6)
        self.coattn = DCNLayer(512, 512, 2, 0.6) # 각 모달리티 당 기존 512 + GAT 512 (vid, aud)

        self.video_attn = BottomUpExtract(256, 256)
        self.audio_extract = LSTM(512, 256, 2, 0.1, residual_embeddings=True)
        self.video_extract = LSTM(256, 256, 2, 0.1, residual_embeddings=True)

        self.video_GAT = Global_Attention_Transformer()
        self.audio_GAT = Global_Attention_Transformer()

        self.vregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))


        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))
        
        # self.Joint = LSTM(1024, 512, 2, dropout=0, residual_embeddings=True)
        self.Joint = LSTM(1024, 512, 2, dropout=0, residual_embeddings=True)        
        
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

        net.apply(init_func)  # apply the initialization function <init_func>


    def forward(self, f1_norm, f2_norm, global_vid=None, global_aud=None):
        video = F.normalize(f2_norm, dim=-1, eps=1e-6)
        audio = F.normalize(f1_norm, dim=-1, eps=1e-6)

        # Tried with LSTMs also
        audio = self.audio_extract(audio)
        video = self.video_attn(video, audio)
        video = self.video_extract(video)

        gat_vid = self.video_GAT(video, global_vid)
        gat_aud = self.audio_GAT(audio, global_aud)

        gloabl_video = torch.cat((video, gat_vid), dim=-1)
        gloabl_audio = torch.cat((audio, gat_aud), dim=-1)
        
        gloabl_video_attn, gloabl_audio_attn = self.coattn(gloabl_video, gloabl_audio)

        audiovisualfeatures = torch.cat((gloabl_video_attn, gloabl_audio_attn), -1)
        
        audiovisualfeatures = self.Joint(audiovisualfeatures)
        vouts = self.vregressor(audiovisualfeatures) #.transpose(0,1))
        aouts = self.aregressor(audiovisualfeatures) #.transpose(0,1))

        return vouts.squeeze(2), aouts.squeeze(2), global_vid, global_aud  