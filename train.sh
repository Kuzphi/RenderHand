CUDA_VISIBLE_DEVICES=0 python train.py -a openpose_hand \
	--resume checkpoint/RenderHand/checkpoint.pth.tar\
	--checkpoint checkpoint/RenderHand/\
	--num-classes 21\
	--train-batch 16\
	--test-batch 4\
	-j 12\
	-e -d
