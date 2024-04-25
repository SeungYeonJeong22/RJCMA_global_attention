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
#import wandb
learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 30
lr = 0.0001
scaler = torch.cuda.amp.GradScaler()

def train(train_loader, model, criterion, optimizer, scheduler, epoch, cam, seed):
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
	print('learning_rate: %s' % str(current_lr))
	logging.info("Learning rate")
	logging.info(current_lr)
	#torch.cuda.synchronize()
	#t1 = time.time()
	n = 0
	time_chk_file = f"./time_chk_seed_{seed}.txt"

	for batch_idx, (visualdata, audiodata, labels_V, labels_A) in tqdm(enumerate(train_loader),
				 										 total=len(train_loader), position=0, leave=True):

		optimizer.zero_grad(set_to_none=True)
		audiodata = audiodata.cuda()#.unsqueeze(2)

		visualdata = visualdata.cuda()#permute(0,4,1,2,3).cuda()
  
		st2 = time.time()


		with torch.cuda.amp.autocast():
			with torch.no_grad():
				b, seq_t, c, subseq_t, h, w = visualdata.size()
				visual_feats = torch.empty((b, seq_t, 25088), dtype=visualdata.dtype, device = visualdata.device)
				aud_feats = torch.empty((b, seq_t, 512), dtype=visualdata.dtype, device = visualdata.device)

				for i in range(visualdata.shape[0]):
					st1 = time.time()
					aud_feat, visualfeat, _ = model(audiodata[i,:,:,:], visualdata[i, :, :, :,:,:])
					ed1 = time.time()

					pre_trained_model_time = ed1 - st1
					with open(time_chk_file, 'a') as f:
						f.write(f"Time pre_trained_model: {pre_trained_model_time}\n")
					visual_feats[i,:,:] = visualfeat
					aud_feats[i,:,:] = aud_feat

			st2 = time.time()
			audiovisual_vouts,audiovisual_aouts = cam(aud_feats, visual_feats)
			ed2 = time.time()
   
			time_cam_model= ed2 - st2
			with open(time_chk_file, 'a') as f:
				f.write(f"Time cam model: {time_cam_model}\n")
				f.write(f"Epoch: {epoch}\n")
				f.write(f"batch_idx: {batch_idx}\n")
				f.write("----"*20)
				f.write("\n")
			f.close()


			voutputs = audiovisual_vouts.view(-1, audiovisual_vouts.shape[0]*audiovisual_vouts.shape[1])
			aoutputs = audiovisual_aouts.view(-1, audiovisual_aouts.shape[0]*audiovisual_aouts.shape[1])
			vtargets = labels_V.view(-1, labels_V.shape[0]*labels_V.shape[1]).cuda()
			atargets = labels_A.view(-1, labels_A.shape[0]*labels_A.shape[1]).cuda()

			v_loss = criterion(voutputs, vtargets)
			a_loss = criterion(aoutputs, atargets)
			final_loss = v_loss + a_loss
			epoch_loss += final_loss.cpu().data.numpy()
		scaler.scale(final_loss).backward()
		scaler.step(optimizer)
		scaler.update()
		n = n + 1

		vout = vout + voutputs.squeeze(0).detach().cpu().tolist()
		vtar = vtar + vtargets.squeeze(0).detach().cpu().tolist()

		aout = aout + aoutputs.squeeze(0).detach().cpu().tolist()
		atar = atar + atargets.squeeze(0).detach().cpu().tolist()

	scheduler.step(epoch_loss / n)

	if (len(vtar) > 1):
		train_vacc = ccc(vout, vtar)
		train_aacc = ccc(aout, atar)
	else:
		train_acc = 0
	print("Train Accuracy")
	print(train_vacc)
	print(train_aacc)
 
	return train_vacc, train_aacc, final_loss
