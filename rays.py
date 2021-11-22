import torch
from kornia import create_meshgrid

def get_ray_directions(H,W,K):
	"""
	Ray directions are basically point connecting the pixel corner and camera center*fov
	This is independent of image and hence needs to be calculated only once
	Inputs:
	H,W,K [3x3 matrix consisting of intrinsics]
	Outputs:
	Directions (Ray is defined as o+td. Now o is constant and d is normalized direction for every pixel)
	dimensions [HxWx3]
	"""
	grid = create_meshgrid(H,W,normalized_coordinates=False)[0] # [H,W,2]
	i,j = grid.unbind(-1) # untangles grid
	focal_x,focal_y,center_x,center_y = K[0,0],K[1,1],K[0,2],K[1,2]
	# create directions and normalize them
	directions = torch.stack([(i-center_x)/focal_x,-(j-center_y)/focal_y, -torch.ones_like(i)],-1) #(H,W,3)

	return directions


def get_rays(directions,c2w):
	"""
	Returns rays for a camera in world space for all pixels in the image
	Input:
	directions : (HxWx3) precomputed ray directions in camera coordinate
	c2w : [3x4] matrix defining how to go from camera coordinate to world coordinate
	R|t is w2c and Rc|C is c2w. R = Rc.T and C is camera coords
	So here we rotate rays_d w.r.t world thus multiply with Rc.T
	and rays_o is simply the last column.
	https://ksimek.github.io/2012/08/22/extrinsic/

	Outputs:
	rays_d : normalized direction rays for all pixels
	rays_o : origin of rays
	"""
	# rotate ray directions from camera_coordinate to world coordinate
	rays_d = directions @ c2w[:,:3].T #(H,W,3)
	rays_d = rays_d/torch.norm(rays_d,dim=1,keepdim=True) #normalize the direction vector

	rays_o = c2w[:,3].expand(rays_d.shape) #(H,W,3) 3rd column is treated as origin
	# replicate the array from 1x3 to HxWx3

	rays_d = rays_d.view(-1,3)
	rays_o = rays_o.view(-1,3) #concatenate dimensions from H,W,3 to HxW,3

	return rays_o,rays_d
 
