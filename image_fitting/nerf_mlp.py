import torch
from torch import nn


class PosEmbedding(nn.Module):
	def __init__(self,in_channels,N_freqs,logscale=True):
		super(PosEmbedding,self).__init__()
		self.functions = [torch.sin,torch.cos]
		self.in_channels = in_channels
		self.out_channels = in_channels*(len(self.functions)*N_freqs+1)
		if logscale:
			self.freqs = 2**torch.linspace(0,N_freqs-1,N_freqs)
		else:
			self.freqs = torch.linspace(1,2**(N_freqs-1),N_freqs)

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
    def __init__(self,width=256):
        """
        W: number of hidden units in each layer
        """
        super(NeRF, self).__init__()
        self.W = width
        
        self.layer1 = nn.Sequential(nn.Linear(3,self.W),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(self.W,self.W),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(self.W,self.W),
                                    nn.ReLU(True))
        self.output = nn.Sequential(nn.Linear(self.W,3),
                                    nn.ReLU(True))

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        out = self.output(l3)
        return out

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
	model = NeRF(width=256)
	nerf_output = model(nerf_input)
	print(nerf_output.shape)

