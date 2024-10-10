from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.utils import clip_gradient

import utils.utils as utils
from utils.exp_utils import pearson
from EvaluationMetrics.ICC import compute_icc
from EvaluationMetrics.cccmetric import ccc

from utils.utils import Normalize
from utils.utils import calc_scores
import logging
# import models.resnet as ResNet
#import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
import math
from losses.CCC import CCC
import subprocess

torch.autograd.set_detect_anomaly(True)

learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 30
lr = 0.0001
scaler = torch.cuda.amp.GradScaler()

def train(train_loader, model, criterion, optimizer, scheduler, epoch, lr, cam, time_chk_path):
	print('\nEpoch: %d' % epoch)
	global Train_acc
	#wandb.watch(audiovisual_model, log_freq=100)
	#wandb.watch(cam, log_freq=100)

	# switch to train mode
	#audiovisual_model.train()
	model.eval()
	cam.train()

	epoch_loss = 0
	vout = list()
	vtar = list()

	aout = list()
	atar = list()

	if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
		frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
		decay_factor = learning_rate_decay_rate ** frac
		current_lr = lr * decay_factor
		utils.set_lr(optimizer, current_lr)  # set the decayed rate
	else:
		current_lr = lr
	######## chckpoint 없을 때 이거부터 ##########
	utils.set_lr(optimizer, current_lr)
	############################################
	print('learning_rate: %s' % str(current_lr))
	logging.info("Learning rate")
	logging.info(current_lr)
	#torch.cuda.synchronize()
	#t1 = time.time()
	n = 0
	global_vid_fts, global_aud_fts= None, None
  

	for batch_idx, (visualdata, audiodata, labels_V, labels_A) in tqdm(enumerate(train_loader),
				 										 total=len(train_loader), position=0, leave=True):
     
     
		optimizer.zero_grad(set_to_none=True)
		audiodata = audiodata.cuda()#.unsqueeze(2)

		visualdata = visualdata.cuda()#permute(0,4,1,2,3).cuda()
  
		st2 = time.time()
		# if batch_idx==3:break

		with torch.cuda.amp.autocast():
			with torch.no_grad():
				b, seq_t, c, subseq_t, h, w = visualdata.size()
				visual_feats = torch.empty((b, seq_t, 25088), dtype=visualdata.dtype, device = visualdata.device)
				aud_feats = torch.empty((b, seq_t, 512), dtype=visualdata.dtype, device = visualdata.device)

				for i in range(visualdata.shape[0]):
					aud_feat, visualfeat, _ = model(audiodata[i,:,:,:], visualdata[i, :, :, :,:,:])
					# visual_feats[i,:,:] = visualfeat
					visual_feats[i,:,:] = visualfeat.view(seq_t, -1)
					aud_feats[i,:,:] = aud_feat


			audiovisual_vouts,audiovisual_aouts, global_vid_fts, global_aud_fts = cam(aud_feats, visual_feats, global_vid_fts, global_aud_fts)
   
			voutputs = audiovisual_vouts.view(-1, audiovisual_vouts.shape[0]*audiovisual_vouts.shape[1])
			aoutputs = audiovisual_aouts.view(-1, audiovisual_aouts.shape[0]*audiovisual_aouts.shape[1])
			vtargets = labels_V.view(-1, labels_V.shape[0]*labels_V.shape[1]).cuda()
			atargets = labels_A.view(-1, labels_A.shape[0]*labels_A.shape[1]).cuda()
   
			v_loss = criterion(voutputs, vtargets)
			a_loss = criterion(aoutputs, atargets)
   
			final_loss = v_loss + a_loss
   
			epoch_loss += final_loss.cpu().data.numpy()

		# scaler.scale(final_loss).backward(retain_graph=True)
		# scaler.step(optimizer)
		# scaler.update()

		with torch.autograd.set_detect_anomaly(True):
			final_loss.backward(retain_graph=True)
			optimizer.step()
		n = n + 1

		vout = vout + voutputs.squeeze(0).detach().cpu().tolist()
		vtar = vtar + vtargets.squeeze(0).detach().cpu().tolist()

		aout = aout + aoutputs.squeeze(0).detach().cpu().tolist()
		atar = atar + atargets.squeeze(0).detach().cpu().tolist()
  
	scheduler.step(epoch_loss / n)

	if (len(vtar) > 1):
		train_vccc = ccc(vout, vtar)
		train_accc = ccc(aout, atar)
	else:
		train_acc = 0
	print("Train Accuracy")
	print("Valence CCC: ", train_vccc)
	print("Arousal CCC: ",train_accc)
 
	return train_vccc, train_accc, final_loss
