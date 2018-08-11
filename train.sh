CUDA_VISIBLE_DEVICES=0,5 python train_RenderHand.py -a renderhand\
	--checkpoint checkpoint/RenderHand/\
	--num-classes 21\
	--train-batch 8\
	--test-batch 8\
	-j 12\
