import torch
from torch import nn


class PosEmbedding(nn.Module):
	def __init__(self,max_logscale,N_freqs,logscale=True):
		super(PosEmbedding,self).__init__()
		self.functions = [torch.sin,torch.cos]

		if logscale:
			self.freqs = 2**torch.linspace(0,max_logscale,N_freqs)
		else:
			self.freqs = torch.linspace(1,2**max_logscale,N_freqs)
		print(self.freqs)

	def forward(self,x):
		"""
		Input is a 3x1 vector and output is 3x21 (if N_freq is 10)
		Idea is that we project it to a higer dimension to be able to
		encode higher frequencies.
		[x] -> [x,sin(x),cos(x),sin(2x),cos(2x),......]
		Inputs:
		x : (Batch,3)
		Outputs:
		out : (Batch, 6*N_freqs+3)
		"""
		out = x
		for freq in self.freqs:
			for func in self.functions:
				out = torch.cat((out,func(freq*x)),axis=1)

		return out


class NeRF(nn.Module):

	def __init__(self,types,density,width,skips,in_channels_xyz,in_channels_dir):
		"""
		types = "coarse", "fine"
		density = no of layers for density (sigma) encoder
		width = no of hidden units in each layer
		skips = add skip connection in Dth layer
		in_channels_xyz = no of input channels for xyz 3 + 6*10 = 63
		in_channels_dir = no of input_channels for dir 3 + 6*4 = 27
		"""
		super(NeRF,self).__init__()
		self.types = types
		self.density = density
		self.width = width
		self.skips = skips
		self.in_channels_xyz = in_channels_xyz
		self.in_channels_dir = in_channels_dir

		"""
		The network architecture is 4-5 layers with input being Pos(x)
		and at the 4th layer adding Pos(x) again. 2 layers later we add
		encoded Pos(dir) and get the final color 
		"""

		# encode xyz
		for i in range(self.density):
			if i == 0:
				layer = nn.Linear(in_channels_xyz,width)
			elif i in skips:
				layer = nn.Linear(width+in_channels_xyz,width)
			else :
				layer = nn.Linear(width,width)
			layer = nn.Sequential(layer,nn.ReLU(True))
			setattr(self,f'xyz_encoding_{i+1}',layer)
		
		self.xyz_encoding_final = nn.Linear(width,width)

		#encode dir
		self.dir_encoding = nn.Sequential(
								nn.Linear(width+in_channels_dir,width//2),nn.ReLU(True))
		self.sigma = nn.Sequential(nn.Linear(width,1),nn.Softplus())
		self.rgb = nn.Sequential(nn.Linear(width//2,3),nn.Sigmoid())

	def forward(self,x,sigma_only = False):
		"""
		Encodes x and dir
		Input:
			x : Pos(x,y,z) and Direction (The ray directions are the dir)
			sigma_only : return only sigma
		Outputs (concatenated):
			rgb and sigma (Batch,4)
		Render this ray in rendering.py
		Sigma is the output of the second last layer while rgb is the final output
		"""
		if sigma_only:
			input_xyz = x
		else:
			input_xyz, input_dir = torch.split(x,[self.in_channels_xyz,self.in_channels_dir],-1)

		xyz_ = input_xyz

		for i in range(self.density):
			if i in self.skips:
				xyz_ = torch.cat([input_xyz,xyz_],dim=-1)
			xyz_ = getattr(self,f'xyz_encoding_{i+1}')(xyz_)

		sigma = self.sigma(xyz_) # (B,1)
		# print(sigma.shape)
		if sigma_only:
			return sigma
		xyz_encoding_final = self.xyz_encoding_final(xyz_)

		dir_encoding_input = torch.cat([xyz_encoding_final,input_dir],-1)
		dir_encoding = self.dir_encoding(dir_encoding_input)

		rgb = self.rgb(dir_encoding) #(B,3)
		# print(rgb.shape)
		output = torch.cat([rgb,sigma],-1) #(B,4)

		return output

if __name__ == '__main__':
	batch = 1024
	xyz_L = 10
	xyz_embedding = PosEmbedding(xyz_L-1,xyz_L)
	input_x  = torch.rand(batch,3)
	pos_x = xyz_embedding(input_x)
	print(pos_x.shape)

	dir_L = 4
	dir_embedding = PosEmbedding(dir_L-1,dir_L)
	input_dir  = torch.rand(batch,3)
	pos_dir = dir_embedding(input_x)
	print(pos_dir.shape)

	nerf_input = torch.cat([pos_x,pos_dir],dim=1)
	print(nerf_input.shape)
	model = NeRF(types="test",density=8,width=256,skips=[4],in_channels_xyz=63,in_channels_dir=27)
	nerf_output = model(nerf_input)
	print(nerf_output.shape)

