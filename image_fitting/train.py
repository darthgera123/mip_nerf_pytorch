import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW,Adam, lr_scheduler
from dataset import ImageData
from nerf_mlp import PosEmbedding,NeRF,GaussPosEmbedding,Siren
from tqdm import tqdm
import numpy as np
import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, default='dog.jpg')
	parser.add_argument('--logs', type=str, default='/scratch/aakash.kt/')
	parser.add_argument('--width', type=int, default=512)
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--load_ckpt', type=str, default='')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--no_pos',action='store_true')
	parser.add_argument('--basic',action='store_true')
	parser.add_argument('--fourier', action='store_true')
	parser.add_argument('--b_scale',type=float, default=1.0)
	parser.add_argument('--b_height',type=int, default=256)
	parser.add_argument('--siren',action='store_true')
	args = parser.parse_args()

	train_data = ImageData(args.image,args.width,args.height,True)
	train_dataloader = DataLoader(train_data,shuffle=True,num_workers=4,batch_size=args.batch_size,pin_memory=True)
	print("Len Train Data",train_data.__len__())
	val_data = ImageData(args.image,args.width,args.height,False)
	val_dataloader = DataLoader(val_data,shuffle=False,num_workers=4,batch_size=args.batch_size,pin_memory=True)
	print("Len Val Data",val_data.__len__())

	import wandb
	wandb.init(project='image_fit',entity='erenyeager',dir=args.logs)
	wandb.config = {
		"lr" : args.lr,
		"epochs" : args.epochs,
		"batch_size" : args.batch_size
	}
	in_channels = 2
	L = 10
	if args.no_pos:
		in_channels = 2
	if args.basic:
		embedding = PosEmbedding(2,L)
		in_channels = 2*L*2+2
	if args.fourier:
		embedding = GaussPosEmbedding(2,10,b_height=args.b_scale,b_scale=args.b_scale)
		in_channels = 2*L*args.b_scale+args.b_scale
	if args.siren:
		model = Siren(in_channels=in_channels,width=256,outermost_linear=True)
	else:
		model = NeRF(in_channels=in_channels,width=256)
	model = model.cuda()

	criterion = torch.nn.MSELoss()
	optimizer = AdamW(model.parameters(),lr=1e-4)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, \
		threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)



	for epoch in tqdm(range(args.epochs)):
		train_loss = 0
		model.train()
		for samples in tqdm(train_dataloader):
			torch.autograd.set_detect_anomaly(True)
			coord,gt = samples 
			coord,gt = coord.cuda(),gt.cuda()
			if not args.no_pos:
				coord_pos = embedding(coord)
				pred = model(coord_pos)
			else:
				pred = model(coord)
			
			loss = criterion(pred,gt)
			train_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
		
		
		wandb.log({"train_loss": train_loss/len(train_dataloader)})
		
		model.eval()
		image = []
		image_gt = []
		val_loss = 0
		for samples in tqdm(val_dataloader):
			coord,gt = samples
			coord,gt = coord.cuda(),gt.cuda()
			if not args.no_pos:
				coord_pos = embedding(coord)
				pred = model(coord_pos)
			else:
				pred = model(coord)
			loss = criterion(pred,gt)
			val_loss += loss.item()

			image.append(pred.detach().cpu().numpy())
			image_gt.append(gt.detach().cpu().numpy())
		
		wandb.log({"val_loss": val_loss/len(val_dataloader)})
		scheduler.step(val_loss)
		images = np.concatenate(image,axis=0).reshape(args.width,args.height,3)
		images_gt = np.concatenate(image_gt,axis=0).reshape(args.width,args.height,3)
		images_gt = np.rot90(images_gt,3)
		images_gt = np.fliplr(images_gt)
		images = np.rot90(images,3)
		images = np.fliplr(images)

		pred_img = (images*255).astype('uint8')
		gt_img = (images_gt*255).astype('uint8')
		psnr_met = psnr(pred_img,gt_img)
		wandb.log({"psnr":psnr_met})
		pred_img_log = wandb.Image(cv2.cvtColor(pred_img,cv2.COLOR_BGR2RGB), caption="Pred")
		gt_img_log = wandb.Image(cv2.cvtColor(gt_img,cv2.COLOR_BGR2RGB), caption="GT")
		wandb.log({"pred":pred_img_log,"gt":gt_img_log})
		cv2.imwrite("gt.png",gt_img)
		cv2.imwrite("pred.png",pred_img)




