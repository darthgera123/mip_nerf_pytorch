import os
import sys
import torch
from collections import defaultdict
from opt import get_opts

from torch.utils.data import DataLoader
from torch.optim import AdamW,Adam, lr_scheduler

from nerf_mlp import PosEmbedding,NeRF
from rendering import rendering
from dataset import NerfData
from metrics import *
from loss import MSELoss
from tqdm import tqdm 
import wandb

if __name__ == '__main__':
	hparams = get_opts()
	device = 'cuda'

	# Define dataloaders
	train_data = NerfData(hparams.image_dir,hparams.img_width,hparams.img_height,'train')
	train_dataloader = DataLoader(train_data,shuffle=True,num_workers=4,batch_size=hparams.batch_size,pin_memory=True)
	
	val_data = NerfData(hparams.image_dir,hparams.img_width,hparams.img_height,'val')
	val_dataloader = DataLoader(val_data,shuffle=False,num_workers=4,batch_size=1,pin_memory=True)


	# Define Embeddings
	xyz_L = hparams.embed_xyz
	embedding_xyz = PosEmbedding(xyz_L-1,xyz_L)

	dir_L = hparams.embed_dir
	embedding_dir = PosEmbedding(dir_L-1,dir_L)

	embdedding = {}
	embdedding['dir'] = embedding_dir
	embdedding['xyz'] = embedding_xyz

	# Define models and optimizer
	nerf_coarse = NeRF(types="coarse",density=8,width=256,skips=[4],in_channels_xyz=3+6*xyz_L,in_channels_dir=3+6*dir_L)
	models = {}
	models['coarse'] = nerf_coarse.to(device)

	if hparams.N_importance > 0:
		nerf_fine = NeRF(types="fine",density=8,width=256,skips=[4],in_channels_xyz=3+6*xyz_L,in_channels_dir=3+6*dir_L)
		models['fine'] = nerf_fine.to(device)

	optimizer = AdamW(list(models['coarse'].parameters()) + list(models['fine'].parameters()),lr = hparams.lr)
	lr_sched = lr_scheduler.ExponentialLR(optimizer,gamma = 1e-5)

	# define checkpoint and log folders here
	# named_tuple = time.localtime()
	# time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
	# log_dir = os.path.join(args.logs, time_string)
	# if not os.path.exists(log_dir):
	# 	os.makedirs(log_dir)

	# checkpoint_dir = args.ckpt + time_string
	# if not os.path.exists(checkpoint_dir):
	# 	os.makedirs(checkpoint_dir)

	# wandb.init(project='nerf',entity='erenyeager',dir=log_dir)
	# wandb.config = {
	# 	"lr" : hparams.lr,
	# 	"epochs" : hparams.num_epochs,
	# 	"batch_size" : args.batch_size
	# }

	# loss function
	criterion = MSELoss()

	#train loop
	for epoch in tqdm(range(hparams.num_epochs)):
		train_loss = 0
		# with torch.no_grad():
		# 	for samples in tqdm(train_dataloader,desc=f'Train: Epoch {epoch}'):
		# 		rays,image_pixel = samples['rays'],samples['images'] #(B,8) (B,3)
		# 		rays,image_pixel = rays.to(device),image_pixel.to(device)
		# 		B = rays.shape[0]
		# 		# print(B,rays.shape)
		# 		results = defaultdict(list)
		# 		for i in range(0,B,hparams.chunk):
		# 			print(rays[i:i+hparams.chunk].shape)
		# 			rendered_ray_chunks = rendering(models,embdedding,rays[i:i+hparams.chunk],
		# 											hparams.N_samples,hparams.use_disp,
		# 											hparams.perturb,hparams.noise_std,hparams.N_importance,
		# 											hparams.chunk,train_data.white_back,False)
		# 			for k,v in rendered_ray_chunks.items():
		# 				results[k] += [v]


		# 		for k,v in results.items():
		# 			results[k] = torch.cat(v,0)

		# 		print(results['rgb_coarse'])

		# 		loss = criterion(results,image_pixel)
		# 		# loss = sum(l for l in loss_d.values())

		# 		psnr_ = psnr(results['rgb_fine'],image_pixel)
		# 		train_loss += loss.item()
		# 		break
		# 	print("Train Loss",train_loss)

		models['coarse'].eval()
		models['fine'].eval()
		val_loss = 0
		for samples in tqdm(val_dataloader,desc=f'Val: Epoch {epoch}'):
			rays,image_pixel,valid_mask = samples['rays'],samples['images'],samples['valid_mask']
			rays, image_pixel,valid_mask = rays.to(device),image_pixel.to(device),valid_mask.to(device)
			rays = rays.squeeze()
			image_pixel = image_pixel.squeeze()

			B = rays.shape[0]
			# print(B,rays.shape)
			results = defaultdict(list)
			full_results = defaultdict(list)
			hparams.chunk = 1024*8
			for b in range(0,B,B//(1024*16)):
				for i in range(0,b,hparams.chunk):
					print(" Rays ",rays[i+b:i+b+hparams.chunk].shape)
					rendered_ray_chunks = rendering(models,embdedding,rays[i+b:i+b+hparams.chunk],
													hparams.N_samples,hparams.use_disp,hparams.perturb,
													hparams.noise_std,hparams.N_importance,hparams.chunk,
													val_data.white_back,False)
					for k,v in rendered_ray_chunks.items():
						results[k] += [v]

			for k,v in results.items():
				results[k] = torch.cat(v,0)
			print(results)
			exit()

			loss = criterion(results,image_pixel)
			# loss = sum(l for l in loss_d.values())
			val_loss += loss.item()
		print("Val Loss",val_loss)


				










