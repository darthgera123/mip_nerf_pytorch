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
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
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
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

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
	model = NeRF(types="test",density=8,width=256,skips=[4],in_channels_xyz=63,in_channels_dir=27)
	nerf_output = model(nerf_input)
	print(nerf_output.shape)

