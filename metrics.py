import torch
from kornia.losses import ssim as dssim

def mse(image_pred,image_gt,valid_mask=None,reduction='mean'):
	value = (image_pred-image_gt)**2
	if valid_mask is not None:
		value = value[valid_mask]
	if reduction == 'mean':
		return torch.mean(value)
	return value

def psnr(image_pred,image_gt,valid_mask,reduction='mean'):
	return -10*torch.log10(mse(image_pred,image_gt,valid_mask,reduction))

def ssim(image_pred,image_gt,reduction='mean'):
	"""
	image_pred is [1,3,H,W]
	"""
	dssim_out = dssim(image_pred,image_gt,3,reduction)
	return 1-2*dssim_out # in [-1,1]