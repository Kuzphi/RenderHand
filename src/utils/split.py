import os
import cv2
import json
import numpy as np
import pickle
class Split():
	def __init__(self,Data_Path, Hand_Model, Hand_Action):
		self.Data_Path = Data_Path
		self.models = Hand_Model
		self.actions = Hand_Action
		self.all = 0
		self.order = []
		self.labels = {}
		for model in self.models:
			for action in self.actions:
				folderpath = os.path.join(Data_Path, model, model + action)
				labelpath = os.path.join(folderpath, 'label.json')
				if not os.path.exists(labelpath):
					print("Warning: model %s action %s does not exists"%(model, action))
					continue

				label = json.load(open(labelpath))
				self.labels[model + action] = label
				self.order.append([len(label), model, action])
				self.all += len(label)

		self.Split()
		pickle.dump(self,open('/home/liangjic/data/RenderHand/split.pickle','w'))
	def Verify(self):
		# for model in self.models:
		# 	for action in self.actions:
				
		# 		folderpath = os.path.join(self.Data_Path, model, model + action)
		# 		imgpath    = os.path.join(folderpath, 'image')
		# 		labelpath  = os.path.join(folderpath, 'label.json')
		# 		if not os.path.exists(os.path.join(folderpath, 'label.json')):
		# 			print("Warning: %s %s does not exists"%(model, action))
		# 			continue

		# 		print("Verifing %s %s"%(model, action))
		# 		label = json.load(open(labelpath))
		# 		if model == 'Model1':
		# 			assert(len(label) == 15660)
		# 			for xid in range(15660):
		# 				file = os.path.join(imgpath, str(xid).zfill(7) + '.png')
		# 				assert(os.path.exists(file))
		# 		if model == 'Model0':
		# 			assert(len(label) == 9720)
		# 			for xid in range(9720):
		# 				file = os.path.join(imgpath, str(xid).zfill(7) + '.png')
		# 				assert(os.path.exists(file))
		# 		print("%s Verify OK !!"%(action))
		# for i in self.train:
		# 	index, model, action =  self.get(i);
		# 	if model == 'Model1':
		# 		assert(0 <= int(index))
		# 		assert(int(index) < 15660)
		# 	if model == 'Model0':
		# 		assert(0 <= int(index))
		# 		assert(int(index) < 9720)
		for i in self.train:
			self.get_sample(i)

	def Split(self, split_type = 'random'):
		gap = int(self.all * 0.2)
		permu = np.arange(self.all)
		np.random.shuffle(permu)
		self.valid = permu[:gap]
		self.train = permu[gap:]

	def get(self,index):
		for item in self.order:
			if index < item[0]:
				return (str(index).zfill(7), item[1], item[2])
			index -= item[0]

	def get_sample(self, index):
		w = self.get(index)
		imgpath = os.path.join(self.Data_Path, w[1], w[1] + w[2], 'image', w[0] + '.png')
		# labelpath = os.path.join(self.Data_Path, w[1], w[1] + w[2], 'label.json')
		# labeljson = json.load(open(labelpath, 'r'))
		img = cv2.imread(imgpath)
		assert(self.labels.has_key(w[1] + w[2]))
		label = self.labels[w[1]+w[2]][w[0]]
		coor = np.array(label['perspective'])
		coor[:,0] = (coor[:,0]) * img.shape[0]
		coor[:,1] = (1 - coor[:,1]) * img.shape[1]
		coor = coor.astype(np.int)
		return (img, coor)

	def show_sample(self):
		import random
		from matplotlib import pyplot as plt 
		index = random.randint(0, self.all - 1)
		img, label = self.get_sample(index)
		img = draw_hand(img, label)
		# ax = plt.figure()
		plt.imshow(img[:,:,::-1])
		plt.show()
if __name__ == '__main__':
	data_set = Split('./',['Model0', 'Model1'], ['ArmRotate','Fist','WristRotate', 'Tap'])
	# data_set = Split('./Render',['Model1'], ['Fist'])
	data_set.Verify()
	# while True:
	# 	data_set.show_sample()
