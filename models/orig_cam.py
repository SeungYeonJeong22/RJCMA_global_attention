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
import torch
import torch.nn as nn
import math
import sys


sys.path.append('./models')
from TCN import TemporalConvNet

# class TLAB(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(TLAB, self).__init__()
#         self.lstm = LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.3, residual_embeddings=True)
#         self.tcn = TemporalConvNet(
#             num_inputs=input_dim, num_channels=[hidden_dim, hidden_dim], kernel_size=3, dropout=0.3
#         )

#         # Attention
#         self.query_fc = nn.Linear(hidden_dim, hidden_dim)  # Q: LSTM output (global information)
#         self.key_fc = nn.Linear(hidden_dim, hidden_dim)    # K: TCN output (local information)
#         self.value_fc = nn.Linear(hidden_dim, hidden_dim)  # V: TCN output (local information)
#         self.attention_softmax = nn.Softmax(dim=-1)        # Softmax for Attention weights

#     def forward(self, x):
#         lstm_feat = self.lstm(x)  # Output: (batch, seq_len, hidden_dim)
#         tcn_feat = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # Output: (batch, seq_len, hidden_dim)

#         # (batch, seq_len, hidden_dim)
#         Q = self.query_fc(lstm_feat) 
#         K = self.key_fc(tcn_feat)    
#         V = self.value_fc(tcn_feat)  

#         # Compute Attention weights
#         attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (batch, seq_len, seq_len)
#         attention_weights = self.attention_softmax(attention_scores)  # Normalize scores

#         # Weighted sum of V
#         attended_feat = torch.matmul(attention_weights, V)  # (batch, seq_len, hidden_dim)

#         # Combine attended features with global (LSTM) features
#         combined_feat = lstm_feat + attended_feat  # (batch, seq_len, hidden_dim)

#         return combined_feat


class TLAB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(TLAB, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.6, residual_embeddings=True)
        self.tcn = TemporalConvNet(
            num_inputs=input_dim, num_channels=[hidden_dim, hidden_dim], kernel_size=3, dropout=0.2
        )
        # Attention
        self.query_fc = nn.Linear(hidden_dim, hidden_dim)  # Q: LSTM output (global information)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)    # K: TCN output (local information)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)  # V: TCN output (local information)
        self.attention_softmax = nn.Softmax(dim=-1)        # Softmax for Attention weights


    def forward(self, x):
        lstm_feat = self.lstm(x)  # Output: (batch, seq_len, hidden_dim)
        tcn_feat = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # Output: (batch, seq_len, hidden_dim)

        # (batch, seq_len, hidden_dim)
        Q = self.query_fc(lstm_feat) 
        K = self.key_fc(tcn_feat)    
        V = self.value_fc(tcn_feat)  

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (batch, num_heads, seq_len, seq_len)
        attention_weights = self.attention_softmax(attention_scores)  # Softmax 적용
        attention_weights = self.dropout(attention_weights)  # Dropout 적용

        # Attention 가중합
        attended_feat = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # Combine attended features with global (LSTM) features
        combined_feat = lstm_feat + attended_feat  # (batch, seq_len, hidden_dim)

        return combined_feat


class TLAB_CAM(nn.Module):
    def __init__(self):
        super(TLAB_CAM, self).__init__()
        self.coattn = DCNLayer(512, 512, 2, 0.6)
        # self.avga = AVGA(512, 512)

        # # Audio and Video TLABs
        # self.audio_tlab = TLAB(512, 512)
        # self.video_tlab = TLAB(512, 512)


        self.audio_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True) # output: (batch, sequence, features)
        self.video_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True) # output: (batch, sequence, features)
        self.video_attn = BottomUpExtract(512, 512)

        self.face_transform = nn.Linear(1408, 25088)
        
        self.vregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        # self.Joint = TLAB(1024, 512)
        self.Joint = LSTM(1024, 512, 2, dropout=0.2, residual_embeddings=True)

        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))
        
        self.weight_video = nn.Parameter(torch.tensor(0.6))
        self.weight_face = nn.Parameter(torch.tensor(0.4))        

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


    def forward(self, f1_norm, f2_norm, face_feate):
        video = F.normalize(f2_norm, dim=-1)
        audio = F.normalize(f1_norm, dim=-1)
        face = F.normalize(face_feate, dim=-1)

        face = self.face_transform(face)

        video = self.weight_video * video + self.weight_face * face  

        # # Tried with LSTMs also
        audio = self.audio_extract(audio)
        video = self.video_attn(video, audio)
        video = self.video_extract(video)

        video, audio = self.coattn(video, audio)

        audiovisualfeatures = torch.cat((video, audio), -1)
        
        audiovisualfeatures = self.Joint(audiovisualfeatures)
        vouts = self.vregressor(audiovisualfeatures) #.transpose(0,1))
        aouts = self.aregressor(audiovisualfeatures) #.transpose(0,1))

        return vouts.squeeze(2), aouts.squeeze(2)  #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)