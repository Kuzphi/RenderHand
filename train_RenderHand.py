from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from src import Bar
from src.utils.logger import Logger, savefig
from src.utils.evaluation import accuracy, AverageMeter, final_preds
from src.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from src.utils.osutils import mkdir_p, isfile, isdir, join
from src.utils.imutils import batch_with_heatmap
from src.utils.transforms import fliplr, flip_back
import src.models as models
import src.datasets as datasets
from src.utils.split import Split
import cv2
import numpy as np

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

idx = range(1,21)

best_acc = 0

# plt.switch_backend('agg')

def main(args):
	global best_acc

	# create checkpoint dir
	if not isdir(args.checkpoint):
		mkdir_p(args.checkpoint)

	# create model
	print("==> creating model '{}', num_classes={}".format(args.arch, args.num_classes))
	model = models.__dict__[args.arch](length = args.length, num_classes=args.num_classes)

	# define loss function (criterion) and optimizer
	criterion = torch.nn.MSELoss().cuda()

	optimizer = torch.optim.RMSprop(model.parameters(), 
								lr=args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	title = 'OpenPose-' + args.arch
	if args.resume:
		if isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_acc = checkpoint['best_acc']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	else:        
		if isfile(args.model_weight):
			weight_dict = torch.load(args.model_weight)
			# print (weight_dict.keys())
			# print (model.state_dict().keys())
			model.load_state_dict(weight_dict)
		logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
		logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

	model = torch.nn.DataParallel(model).cuda()

	cudnn.benchmark = True
	print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

	# Data loading code
	Select_Models  = ['Model0', 'Model1']
	Select_Actinos = ['ArmRotate','Fist','WristRotate', 'Tap']
	if os.path.exists('./data/RenderHand/split.pickle'):
		import pickle
		split = pickle.load(open('./data/RenderHand/split.pickle'))
	else:
		split = Split('./data/RenderHand/', Hand_Model = Select_Models, Hand_Action = Select_Actinos)
	
	train_loader = torch.utils.data.DataLoader(
		datasets.Hand(split),
		batch_size=args.train_batch, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	print ("train loader", len(train_loader))

	val_loader = torch.utils.data.DataLoader(
		datasets.Hand(split, isTrain = 0),
		batch_size=args.test_batch, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	print ("valid loader", len(val_loader))

	if args.evaluate:
		print('\nEvaluation only') 
		loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.debug, args.flip)
		print(loss, acc)
		save_pred(predictions, checkpoint=args.checkpoint)
		return

	lr = args.lr
	for epoch in range(args.start_epoch, args.epochs):
		lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
		print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

		# train for one epoch
		train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.debug, args.flip)

		# evaluate on validation set
		valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.num_classes,
													  args.debug, args.flip)

		# append logger file
		logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

		# remember best acc and save checkpoint
		is_best = valid_acc > best_acc
		best_acc = max(valid_acc, best_acc)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_acc': best_acc,
			'optimizer' : optimizer.state_dict(),
		}, predictions, is_best, checkpoint=args.checkpoint)

	logger.close()
	# logger.plot(['Train Loss', 'Val Loss'])
	# savefig(os.path.join(args.checkpoint, 'Loss.eps'))
	logger.plot(['Train Acc', 'Val Acc'])
	savefig(os.path.join(args.checkpoint, 'Acc.eps'))

def train(train_loader, model, criterion, optimizer, debug=False, flip=True):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acces = AverageMeter()
	distes = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	gt_win, pred_win = None, None
	bar = Bar('Processing', max=len(train_loader))
	for i, (inputs, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		input_var = torch.autograd.Variable(inputs.cuda())
		target_var = torch.autograd.Variable(target.cuda(async=True))

		# compute output
		output = model(input_var)
		score_map = output[-1].data.cpu()
		loss = criterion(output[0], target_var)
		for j in range(1, len(output)):
			loss += criterion(output[j], target_var)
		acc, dist = accuracy(score_map, target, idx)
		if debug: # visualize groundtruth and predictions
			gt_batch_img = batch_with_heatmap(inputs, target)
			pred_batch_img = batch_with_heatmap(inputs, score_map)
			if not gt_win or not pred_win:
				ax1 = plt.subplot(121)
				ax1.title.set_text('Groundtruth')
				gt_win = plt.imshow(gt_batch_img[:,:,::-1])
				ax2 = plt.subplot(122)
				ax2.title.set_text('Prediction')
				pred_win = plt.imshow(pred_batch_img[:,:,::-1])
			else:
				gt_win.set_data(gt_batch_img)
				pred_win.set_data(pred_batch_img)
			plt.plot()
			plt.pause(.5)
			

		# measure accuracy and record loss
		losses.update(loss.item(), inputs.size(0))
		acces.update(acc[0], inputs.size(0))
		distes.update(dist[0], inputs.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# plot progress
		bar.suffix  = '({batch}/{size}) Data: {data:.1f}s | Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f} | Dist {dist:.3f}'.format(
					batch=i + 1,
					size=len(train_loader),
					data=data_time.val,
					bt=batch_time.avg,
					total=bar.elapsed_td,
					eta=bar.eta_td,
					loss=losses.avg,
					acc=acces.avg,
					dist=distes.avg
					)
		bar.next()

	bar.finish()
	return losses.avg, acces.avg

def validate(val_loader, model, criterion, num_classes, debug=False, flip=True):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acces = AverageMeter()
	distes = AverageMeter()

	# predictions
	predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

	# switch to evaluate mode
	model.eval()

	gt_win, pred_win = None, None
	end = time.time()
	bar = Bar('Processing', max=len(val_loader))
	for i, (inputs, target) in enumerate(val_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		with torch.no_grad():
			input_var = torch.autograd.Variable(inputs.cuda())
			target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input_var)
		score_map = output[-1].data.cpu()
		if flip:
			flip_input_var = torch.autograd.Variable(
					torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(), 
					volatile=True
				)
			flip_output_var = model(flip_input_var)
			flip_output = flip_back(flip_output_var[-1].data.cpu())
			score_map += flip_output


		loss = 0
		for o in output:
			loss += criterion(o[:,:21,:,:], target_var)
		acc, dist = accuracy(score_map[:,:21,:,:].contiguous(), target.cpu(), idx)

		# generate predictions		
		# preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
		preds = score_map
		# for n in range(score_map.size(0)):
		# 	predictions[meta['index'][n], :, :] = preds[n, :, :]
		# print(debug)
		if debug:
			gt_batch_img = batch_with_heatmap(inputs, target)
			pred_batch_img = batch_with_heatmap(inputs, score_map)

			sz = tuple([x * 4 for x in gt_batch_img[:,:,0].shape])
			gt_batch_img = cv2.resize(gt_batch_img,(sz[1],sz[0]),)
			pred_batch_img = cv2.resize(pred_batch_img, (sz[1],sz[0]))

			if not gt_win or not pred_win:
				# plt.imshow(gt_batch_img)
				plt.subplot(121)
				gt_win = plt.imshow(gt_batch_img[:,:,::-1])
				plt.subplot(122)
				pred_win = plt.imshow(pred_batch_img[:,:,::-1])
			else:
				gt_win.set_data(gt_batch_img)
				pred_win.set_data(pred_batch_img)

			plt.savefig("./tmp/" + str(i) + ".png", dpi = 1000, bbox_inches='tight')

		# measure accuracy and record loss
		losses.update(loss.item(), inputs.size(0))
		acces.update(acc[0], inputs.size(0))
		distes.update(dist[0], inputs.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# plot progress
		bar.suffix  = '({batch}/{size}) Data: {data:.1f}s | Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f} | Dist {dist:.3f}'.format(
					batch=i + 1,
					size=len(val_loader),
					data=data_time.val,
					bt=batch_time.avg,
					total=bar.elapsed_td,
					eta=bar.eta_td,
					loss=losses.avg,
					acc=acces.avg,
					dist=distes.avg
					)
		bar.next()

	bar.finish()
	return losses.avg, acces.avg, predictions


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
	# Model structure
	parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
						choices=model_names,
						help='model architecture: ' +
							' | '.join(model_names) +
							' (default: resnet18)')

	parser.add_argument('-l', '--length', default=2000, type=int, metavar='N',
						help='Length of data sequence')

	parser.add_argument('--features', default=256, type=int, metavar='N',
						help='Number of features in the hourglass')
	parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
						help='Number of residual modules at each location in the hourglass')
	parser.add_argument('--num-classes', default=21, type=int, metavar='N',
						help='Number of keypoints')

	parser.add_argument('--out_res', default = 80, type = int, metavar = 'N',
						help='resolution of output')
	# Training strategy
	parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', default=90, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
						help='manual epoch number (useful on restarts)')
	parser.add_argument('--train-batch', default=6, type=int, metavar='N',
						help='train batchsize')
	parser.add_argument('--test-batch', default=6, type=int, metavar='N',
						help='test batchsize')
	parser.add_argument('--lr', '--learning-rate', default=2.5e-5, type=float,
						metavar='LR', help='initial learning rate')
	parser.add_argument('--momentum', default=0, type=float, metavar='M',
						help='momentum')
	parser.add_argument('--weight-decay', '--wd', default=0, type=float,
						metavar='W', help='weight decay (default: 0)')
	parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
						help='Decrease learning rate at these epochs.')
	parser.add_argument('--gamma', type=float, default=0.1,
						help='LR is multiplied by gamma on schedule.')
	# Data processing
	parser.add_argument('-f', '--flip', dest='flip', action='store_true',
						help='flip the input during validation')
	parser.add_argument('--sigma', type=float, default=1,
						help='Groundtruth Gaussian sigma.')
	parser.add_argument('--sigma-decay', type=float, default=0,
						help='Sigma decay rate for each epoch.')
	parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
						choices=['Gaussian', 'Cauchy'],
						help='Labelmap dist type: (default=Gaussian)')
	# Miscs
	parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
						help='path to save checkpoint (default: checkpoint)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('--model_weight', default='', type=str, metavar='PATH',
						help='path to model weight file(default: none)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
						help='evaluate model on validation set')
	parser.add_argument('-d', '--debug', dest='debug', action='store_true',
						help='show intermediate results')


	main(parser.parse_args())