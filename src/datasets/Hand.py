from __future__ import print_function, absolute_import

import os
import numpy as np
import pickle
import random
import math

import torch
import torch.utils.data as data

from src.utils.osutils import *
from src.utils.imutils import *
from src.utils.transforms import *

class Hand(data.Dataset):
	def __init__(self, split, inp_res=800, out_res=100, isTrain=True, sigma=1,
				 scale_factor=0.25, rot_factor=30, label_type='Gaussian'):
	
		self.is_train = isTrain         # training set or test set
		self.inp_res = inp_res
		self.out_res = out_res
		self.sigma = sigma
		self.scale_factor = scale_factor
		self.rot_factor = rot_factor
		self.label_type = label_type

		self.split = split
		# create train/val split
		# self.mean, self.std = self._compute_mean()

	def _compute_mean(self):
		meanstd_file = './data/RenderHand/mean.pth.tar'
		if isfile(meanstd_file):
			meanstd = torch.load(meanstd_file)
		else:
			mean = torch.zeros(3)
			std = torch.zeros(3)
			for index in self.split.train:
				img = self.split.get_sample(index)[0].transpose(2,0,1)
				img = torch.Tensor(img)
				mean += img.view(img.size(0), -1).mean(1)
				std += img.view(img.size(0), -1).std(1)
			mean /= len(self.train)
			std /= len(self.train)
			meanstd = {
				'mean': mean,
				'std': std,
				}
			torch.save(meanstd, meanstd_file)
		if self.is_train:
			print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
			print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
			
		return meanstd['mean'], meanstd['std']

	def __getitem__(self, index):
		if self.is_train:
			img, label = self.split.get_sample(self.split.train[index])
		else:
			img, label = self.split.get_sample(self.split.valid[index])
		r = 0
		img = img.transpose(2,0,1)
		img = img / 255.0 - 0.5;
		img = torch.Tensor(img)
		pts = torch.Tensor(label)
		nparts = pts.size(0)
		if self.is_train:
			# s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
			# r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

			# Flip
			# if random.random() <= 0.5:
				# img = torch.from_numpy(fliplr(img.numpy())).float()
				# pts = shufflelr(pts, width=img.size(2), dataset='RHD')
				# c[0] = img.size(2) - c[0]

			# Color
			img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
			img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
			img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

		# Prepare image and groundtruth map
		# inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
		# inp = color_normalize(inp, self.mean, self.std)
		inp = color_normalize(img, [0,0,0], [1,1,1])
		# inp = color_normalize(img, self.mean, self.std)

		# Generate ground truth
		tpts = pts.clone()
		tpts[:,:2] = tpts[:,:2] / (img.shape[1] / self.out_res)
		
		target = torch.zeros(nparts, self.out_res, self.out_res)

		for i in range(nparts):
			target[i] = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)

		# Meta info
		meta = {'index' : index, 'pts' : pts, 'tpts' : tpts}
		return inp, target

	def __len__(self):
		if self.is_train:
			return len(self.split.train)
		else:
			return len(self.split.valid)
