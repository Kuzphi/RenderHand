CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -a openpose_hand\
	--checkpoint checkpoint/RenderHand-Model1TapValid/\
	--num-classes 21\
	--train-batch 8\
	--test-batch 8\
	-j 12\
