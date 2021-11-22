import numpy as np
import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms as T
import argparse
from rays import get_rays, get_ray_directions


class NerfData(Dataset):
	def __init__(self, data_dir, split='train', scale=1):
		self.images_dir = data_dir
		self.scale = scale
		self.width = 800//scale
		self.height = 800//scale
		self.split = split
		self._read_meta(data_dir, split)

		# Needed for alpha compositing the points on ray
		self.white_back = True
		
	def __len__(self):
		if self.split == 'train':
			return len(os.listdir(os.path.join(self.images_dir,self.split)))
		else:
			return 8

	def _read_meta(self, path, split):
		self.data = json.load(
		    open(os.path.join(path, f'transforms_{split}.json')))['frames']
		self.camera_angle_x = json.load(open(os.path.join(path, f'transforms_{split}.json')))[
		                                'camera_angle_x']
		self.focal_length = 0.5*self.width/np.tan(0.5*self.camera_angle_x)

		# Intrinsics camera matrix
		self.K = np.eye(3)
		self.K[0, 0] = self.focal_length
		self.K[1, 1] = self.focal_length
		self.K[0, 2] = self.width/2
		self.K[1, 2] = self.height/2

		self.near = 2.0
		self.far = 6.0
		self.bounds = np.array([self.near, self.far])

		self.pinhole_extrinsics = np.array(
		    [self.height, self.width, self.focal_length])
		"""
		Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
		ray-tracing-generating-camera-rays/standard-coordinate-systems
		https://ksimek.github.io/2013/08/13/intrinsic/
		"""
		# ray direction for all pixels will be same
		self.directions = get_ray_directions(self.height, self.width, self.K)
		self.transform = T.ToTensor()

		# cache all train data together. val and test data can be generate per image
		if self.split == 'train':
			self.all_rays = []
			self.all_images = []
			for t, frame in enumerate(self.data):
				pose = np.array(frame['transform_matrix'])[:3, :4]
				c2w = torch.FloatTensor(pose)

				image_path = os.path.join(self.images_dir, f"{frame['file_path']}.png")
				img = Image.open(image_path)
				"""
				https://www.linkedin.com/pulse/afternoon-debugging-e-commerce-image-processing-nikhil-rasiwasia/
				Basically all images are 4 channel with alpha being the final one
				to read it correctly we need to blend A to RGB
				Note: We dont need alpha at all
				"""
				img = img.resize((self.width, self.height), Image.LANCZOS)
				# reading with PIL and then doing transform is giving alpha channel as well
				img = self.transform(img)
				valid_mask = (img[-1] > 0).flatten()  # valid color area. HxW
				img = img.view(4, -1).permute(1, 0)  # (HxW,4) RGBA
				img = img[:, :3]*img[:, -1:]+(1-img[:, -1:])  # Blend A to RGB

				rays_o, rays_d = get_rays(self.directions, c2w)
				rays_t = t*torch.ones(len(rays_o), 1)

				rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]),\
									self.far*torch.ones_like(rays_o[:, :1]), rays_t],axis=1)  # (h*w,9)
				
				self.all_rays.append(rays)

				self.all_images.append(img)

			self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.data)*h*w,3)
			self.all_images = torch.cat(self.all_images, 0)  # (len(self.data)*h*w,3)
			

	def __getitem__(self, index):
		if self.split == 'train':
			sample = {'rays': self.all_rays[index, :8],
						'ts': self.all_rays[index, 8].long(),
						'images': self.all_images[index]}
		else:
			frame = self.data[index]
			image_path = os.path.join(self.images_dir, f"{frame['file_path']}.png")
			img = Image.open(image_path)
			img = img.resize((self.width, self.height), Image.LANCZOS)
			img = self.transform(img)
			valid_mask = (img[-1] > 0).flatten()  # valid color area. HxW
			img = img.view(4, -1).permute(1, 0)  # (HxW,4) RGBA
			img = img[:, :3]*img[:, -1:]+(1-img[:, -1:])

			pose = np.array(frame['transform_matrix'])[:3, :4]
			c2w = torch.FloatTensor(pose)
			rays_o, rays_d = get_rays(self.directions, c2w)

			rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]),\
									self.far*torch.ones_like(rays_o[:, :1])],axis=1)   # (H,W,8)
			t = 0
			sample = {'rays': rays, 'ts': t*torch.ones(len(rays), dtype=torch.long),
						'images': img, 'c2w': c2w, 'valid_mask': valid_mask}

		return sample


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir', type=str, default='pinecone_dr.xml')
	parser.add_argument('--split', type=str, default='pinecone_dr.xml')
	args = parser.parse_args()
	train_data = NerfData(args.image_dir,args.split)
	print(train_data.__len__())
	print(train_data.__getitem__(42))


