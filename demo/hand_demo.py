import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib.pyplot as plt
import numpy as np
import util
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
sys.path.append("..")
from src.models.OpenPose import openpose_hand
#parser = argparse.ArgumentParser()
#parser.add_argument('--t7_file', required=True)
#parser.add_argument('--pth_file', required=True)
#args = parser.parse_args()

torch.set_num_threads(torch.get_num_threads())
weight_name = '../checkpoint/RenderHand/checkpoint.pth.tar'
test_image = './sample_image/1.png'

# visualize
colors = [
		[100.,  100.,  100.], 
		[100.,    0.,    0.],
		[150.,    0.,    0.],
		[200.,    0.,    0.],
		[255.,    0.,    0.],
		[100.,  100.,    0.],
		[150.,  150.,    0.],
		[200.,  200.,    0.],
		[255.,  255.,    0.],
		[  0.,  100.,   50.],
		[  0.,  150.,   75.],
		[  0.,  200.,  100.],
		[  0.,  255.,  125.],
		[  0.,   50.,  100.],
		[  0.,   75.,  150.],
		[  0.,  100.,  200.],
		[  0.,  125.,  255.],
		[100.,    0.,  100.],
		[150.,    0.,  150.],
		[200.,    0.,  200.],
		[255.,    0.,  255.]]
def Hand_Inference(oriImg, Model = None, Name = ""):
	num_classes = 21
	if Model == None:
		model = openpose_hand(num_classes = num_classes)
		model = torch.nn.DataParallel(model).cuda().float()
		model.load_state_dict(torch.load(weight_name)['state_dict'])
		model.eval()
	else:
		model = Model
	param_, model_ = config_reader('config_hand')

	#torch.nn.functional.pad(img pad, mode='constant', value=model_['padValue'])
	# tic = time.time()

	#test_image = 'a.jpg'
	with torch.no_grad():
		imageToTest = Variable(T.unsqueeze(torch.from_numpy(oriImg.transpose(2,0,1)).float(),0)).cuda()
		multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]
		heatmap_avg = torch.zeros((len(multiplier),num_classes,oriImg.shape[0], oriImg.shape[1])).cuda()
	# paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()
	# print heatmap_avg.size()

	# toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()
	for m in range(len(multiplier)):
		scale = multiplier[m]
		# h = int(oriImg.shape[0]*scale)
		# w = int(oriImg.shape[1]*scale)
		# pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
		# pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
		# new_h = h+pad_h
		# new_w = w+pad_w

		# imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		# imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
		# imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1)) / 256.0 - 0.5
		imageToTest_padded = oriImg[:,:,:,np.newaxis].transpose(3,2,0,1).astype(np.float32) / 256.0 - 0.5


		# plt.pause(10)
		# plt.imshow(imageToTest_padded.transpose(0,2,3,1)[0,:,:,::-1] + 0.5)
		with torch.no_grad():		
			feed = Variable(T.from_numpy(imageToTest_padded)).cuda()
			output2 = model(feed)[-1]

		heatmap = nn.Upsample((oriImg.shape[0], oriImg.shape[1]), mode = 'bilinear', align_corners=True).cuda()(output2)     
		heatmap_avg[m] = heatmap[0].data
		for output in output2[0]:
			xxx = imageToTest_padded[0,:,:,:].transpose(1,2,0)[:,:,::-1] + 0.5
			xxx = cv2.resize(xxx, (100,100))
			print(output.shape)
			plt.imshow(xxx)
			plt.imshow(output, alpha = 0.3)
			plt.plot()
			plt.pause(1)
		
	# toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()

	heatmap_avg = T.mean(heatmap_avg, 0)
	heatmap_avg = T.transpose(T.transpose(T.squeeze(heatmap_avg),0,1),1,2).cuda() 
	heatmap_avg = heatmap_avg.cpu().numpy()

	# toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()

	all_peaks = []
	peak_counter = 0

	#maps = 
	for part in range(21):
		map_ori = heatmap_avg[:,:,part]
		# plt.imshow(oriImg[:,:,[2,1,0]])
		# plt.imshow(heatmap_avg[:,:,part], alpha=.3)
		# plt.savefig("demo_heat/test_part_"+ str(part) + ".png")
		# plt.savefig("demo_heat/test_part_"+ str(part) +"for" + '_'.join(Name.split('/')[-1].split(".")) + ".png")
		map = gaussian_filter(map_ori, sigma=3)

		map_left = np.zeros(map.shape)
		map_left[1:,:] = map[:-1,:]
		map_right = np.zeros(map.shape)
		map_right[:-1,:] = map[1:,:]
		map_up = np.zeros(map.shape)
		map_up[:,1:] = map[:,:-1]
		map_down = np.zeros(map.shape)
		map_down[:,:-1] = map[:,1:]
		
		peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
		peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
		
		peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
		id = range(peak_counter, peak_counter + len(peaks))
		peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

		all_peaks.append(peaks_with_score_and_id)
		peak_counter += len(peaks)

	# print all_peaks

	
	# for i in range(22):
	# 	for j in range(len(all_peaks[i])):
	# 		cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
	# 		cv2.putText(canvas,str(i),all_peaks[i][j][0:2],font,1,colors[i],1)
	ans = []
	for i in range(21):
		if len(all_peaks[i]) == 0:
			ans.append([0,0,0])
			continue
		Max = 0
		for j in range(len(all_peaks[i])):
			if all_peaks[i][j][3] > all_peaks[i][Max][3]:
				Max = j
		ans.append(list(all_peaks[i][Max][:2]) + [float(all_peaks[i][Max][2])])
		
	# toc =time.time()
	# print 'time is %.5f'%(toc-tic)     
	# print ans
	return ans

def tmp_Hand_Inference(oriImg):
	model = openpose_hand()     
	model.load_state_dict(torch.load(weight_name))
	model.cuda()
	model.float()
	model.eval()

	param_, model_ = config_reader('config_hand')

	#torch.nn.functional.pad(img pad, mode='constant', value=model_['padValue'])
	tic = time.time()

	#test_image = 'a.jpg'
	
	imageToTest = Variable(T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(),0),2,3),1,2),volatile=True).cuda()

	multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]

	heatmap_avg = torch.zeros((len(multiplier),22,oriImg.shape[0], oriImg.shape[1])).cuda()
	paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()
	#print heatmap_avg.size()

	# toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()
	for m in range(len(multiplier)):
		scale = multiplier[m]
		h = int(oriImg.shape[0]*scale)
		w = int(oriImg.shape[1]*scale)
		pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
		pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
		new_h = h+pad_h
		new_w = w+pad_w

		imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
		imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1)) / 256.0 - 0.5
		
		feed = Variable(T.from_numpy(imageToTest_padded)).cuda()      
		output2 = model(feed)[-1]
		heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)     
		heatmap_avg[m] = heatmap[0].data
		
		
	# toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()
		
	heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda() 
	heatmap_avg=heatmap_avg.cpu().numpy()
	toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()

	all_peaks = []
	peak_counter = 0

	# for i in range(5, 21):
	# 	heatmap_avg[i] /= heatmap_avg[i].max()

	heatmap_avg[:,:,5] = np.maximum(heatmap_avg[:,:,5], np.maximum(heatmap_avg[:,:, 9], np.maximum(heatmap_avg[:,:,13],heatmap_avg[:,:,17])))
	heatmap_avg[:,:,6] = np.maximum(heatmap_avg[:,:,6], np.maximum(heatmap_avg[:,:,10], np.maximum(heatmap_avg[:,:,14],heatmap_avg[:,:,18])))
	heatmap_avg[:,:,7] = np.maximum(heatmap_avg[:,:,7], np.maximum(heatmap_avg[:,:,11], np.maximum(heatmap_avg[:,:,15],heatmap_avg[:,:,19])))			
	heatmap_avg[:,:,8] = np.maximum(heatmap_avg[:,:,8], np.maximum(heatmap_avg[:,:,12], np.maximum(heatmap_avg[:,:,16],heatmap_avg[:,:,20])))
	#maps = 
	for part in range(9):
		map_ori = heatmap_avg[:,:,part]
		plt.imshow(oriImg[:,:,[2,1,0]])
		plt.imshow(heatmap_avg[:,:,part], alpha=.5)
		plt.savefig("demo_heat/test_part_"+ str(part) +" of " + '_'.join(test_image.split('/')[-1].split(".")) + ".png")
		map = gaussian_filter(map_ori, sigma=3)

		map_left = np.zeros(map.shape)
		map_left[1:,:] = map[:-1,:]
		map_right = np.zeros(map.shape)
		map_right[:-1,:] = map[1:,:]
		map_up = np.zeros(map.shape)
		map_up[:,1:] = map[:,:-1]
		map_down = np.zeros(map.shape)
		map_down[:,:-1] = map[:,1:]
		
		peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
		peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
		
		peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
		id = range(peak_counter, peak_counter + len(peaks))
		peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

		all_peaks.append(peaks_with_score_and_id)
		peak_counter += len(peaks)

	# print all_peaks
	ans = []
	for i in range(9):
		for j in range(len(all_peaks[i])):
			ans.append(all_peaks[i][j][:3])
	print ans
	# for i in range(9):
	# 	if len(all_peaks[i]) == 0:
	# 		ans.append(-1)
	# 		continue
	# 	Max = 0
	# 	for j in range(len(all_peaks[i])):
	# 		if all_peaks[i][j][3] > all_peaks[i][Max][3]:
	# 			Max = j
	# 	ans.append(list(all_peaks[i][Max][:2]))		
	return ans

def draw_hand(canvas, joint, numclass =22, with_number = False, Edge = True):
	hand_map = [[0, 1],[1 , 2],[2 , 3],[3 , 4],
				[0, 5],[5 , 6],[6 , 7],[7 , 8],
				[0, 9],[9 ,10],[10,11],[11,12],
				[0,13],[13,14],[14,15],[15,16],
				[0,17],[17,18],[18,19],[19,20]]
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in range(len(joint)):
		if type(joint[i]) == int:
			continue
		cv2.circle(canvas, tuple(joint[i][:2]), 4, colors[i], thickness=-1)
		if with_number:
			cv2.putText(canvas,str(i),tuple(joint[i][:2]),font,1,colors[i],1)
	if Edge:	
		for edge in hand_map:
			u,v = edge
			if type(joint[u]) == int or type(joint[v]) == int:
				continue
			cv2.line(canvas,tuple(joint[u][:2]),tuple(joint[v][:2]),colors[v],3)
	return canvas


if __name__ == '__main__':
	import pickle
	split = pickle.load(open('../data/RenderHand/split.pickle'))
	split.Data_Path = '/home/liangjic/data/RenderHand/'
	plt.figure()
	# oriImg = cv2.imread(test_image) # B,G,R order
	# joint = Hand_Inference(oriImg)
	# canvas = cv2.imread(test_image) # B,G,R order
	model = openpose_hand(num_classes = 21)
	model = torch.nn.DataParallel(model).cuda().float()
	model.load_state_dict(torch.load(weight_name)['state_dict'])
	oriImg = cv2.imread('./hands_green.jpeg')
	oriImg = cv2.resize(oriImg,(800,800))
	joint = Hand_Inference(oriImg, Model = model)
	canvas = oriImg.copy()
	canvas = draw_hand(canvas, joint, Edge = True)
	plt.imshow(canvas[:,:,::-1])
	plt.pause(1)
	# for i in split.valid:
	# 	print i
	# 	oriImg, label = split.get_sample(i)
	# 	canvas = oriImg.copy()
	# 	canvas = draw_hand(canvas, label)
	# 	plt.subplot(1,2,1)
	# 	plt.imshow(canvas[:,:,::-1])

	# 	joint = Hand_Inference(oriImg, Model = model, Name = str(i))
	# 	# print joint
	# 	joint = np.array(joint).astype(np.int)
	# 	canvas = oriImg.copy()
	# 	canvas = draw_hand(canvas, joint, Edge = True)
	# 	plt.subplot(1,2,2)
	# 	plt.imshow(canvas[:,:,::-1])
	# 	plt.pause(1)
	cv2.imwrite('result.png',canvas)