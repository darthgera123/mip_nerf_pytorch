import numpy as np
import os
import cv2
import json
import torch
from torch.utils.data import Dataset
import argparse
from skimage.transform import resize


class ImageData(Dataset):
	def __init__(self, path, width,height,train=True):
		self.width = width
		self.height = height
		img = cv2.imread(path)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		self.image = resize(img,(self.height,self.width),anti_aliasing=True)
		# self.image_shape = self.image.reshape(-1,3)
		coords = np.linspace(0,1,self.width,endpoint=False)
		all_coords = np.stack(np.meshgrid(coords,coords),-1).reshape(-1,2)
		
		if train:
			self.inp_coords = all_coords[::2]
		else:
			self.inp_coords = all_coords

	def __len__(self):
		return self.inp_coords.shape[0]

	def __getitem__(self, index):
		x,y = self.inp_coords[index]
		gt_image = self.image[(x*self.width).astype('int16'),(y*self.height).astype('int16')]
		pred = torch.from_numpy(self.inp_coords[index]).float()
		image = torch.from_numpy(gt_image).float()

		return pred,image

if __name__ == '__main__':
	train_data = ImageData("image.jpg",512,512,True)
	pred,image = train_data.__getitem__(20)
	print(pred,image)