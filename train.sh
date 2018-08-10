CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -a openpose_hand \
	--checkpoint checkpoint/RenderHand/\
	--num-classes 21\
	--train-batch 16\
	--test-batch 16\
	-j 12\
