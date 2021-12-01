import os
import sys
import torch
from collections import defaultdict
from opt import get_opts

from torch.utils.data import Dataloader
from torch.optim import AdamW,Adam, lr_scheduler

from nerf_mlp import PosEmbedding,NeRF
from rendering import rendering
from dataset import NerfData
from metrics import *
from loss import MSELoss
from tqdm import tqdm 


if __name__ == '__main__':
	hparams = get_opts()
	device = 'cuda'

	# Define dataloaders
	train_data = NerfData(hparams.image_dir,hparams.image_width,hparams.image_height,'train')
	train_dataloader = Dataloader(train_data,shuffle=True,num_workers=4,batch_size=hparams.batch_size,pin_memory=True)
	
	val_data = NerfData(hparams.image_dir,hparams.image_width,hparams.image_height,'val')
	val_dataloader = Dataloader(val_data,shuffle=False,num_workers=4,batch_size=1,pin_memory=True)


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


	# loss function
	criterion = MSELoss()

	#train loop
	for epoch in tqdm(range(hparams.num_epochs)):

		with torch.no_grad():
			for samples in tqdm(train_dataloader):
				rays,image_pixel = samples['rays'],samples['images'] #(B,8) (B,3)
				rays,image_pixel = rays.to(device),images.to(device)
				B = rays.shape[0]
				results = defaultdict(list)
				for i in range(0,B,hparams.chunk):
					rendered_ray_chunks = rendering(models,embdedding,rays[i:i+hparams.chunk],
													hparams.N_samples,hparams.use_disp,
													hparams.pertub,hparams.noise_std,hparams.N_importance,
													hparams.chunk,train_dataset.white_back)
					for k,v in rendered_ray_chunks.items():
						results[k] += [v]


				for k,v in results.items():
					results[k] = torch.cat(v,0)

				loss = criterion(results,image_pixel)
				










