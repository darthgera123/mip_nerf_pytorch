import torch
from torch import nn
import numpy as np

class PosEmbedding(nn.Module):
	def __init__(self, in_channels, N_freqs, logscale=True):
		super(PosEmbedding, self).__init__()
		self.functions = [torch.sin, torch.cos]
		self.in_channels = in_channels
		self.out_channels = in_channels*(len(self.functions)*N_freqs+1)
		if logscale:
			self.freqs = 2**torch.linspace(0, N_freqs-1, N_freqs)
			# [1,2,4,8,16,..,2**10]
		else:
			self.freqs = torch.linspace(1, 2**(N_freqs-1), N_freqs)
			# [1,2,3,4,5,6,7,...,2**10]

	def forward(self, x):
		"""
		Input is a 2x1 vector and output is 2x21 (if N_freq is 10)
		Idea is that we project it to a higer dimension to be able to
		encode higher frequencies.
		[x] -> [x,sin(x),cos(x),sin(2x),cos(2x),......]
		Inputs:
		x : (Batch,3)
		Outputs:
		out : (Batch, 6*N_freqs+3)
		"""
		out = x
		pi = torch.acos(torch.zeros(1)).item() * 2
		for freq in self.freqs:
			for func in self.functions:
				out = torch.cat((out, func(freq*x)), axis=1)

		return out


class GaussPosEmbedding(nn.Module):
	def __init__(self, in_channels, N_freqs, b_height=256, b_scale=1, logscale=True):
		super(GaussPosEmbedding, self).__init__()
		self.functions = [torch.sin, torch.cos]
		self.in_channels = in_channels
		self.out_channels = in_channels*(len(self.functions)*N_freqs+1)
		self.b_height = b_height
		self.B = torch.normal(0, b_scale, size=(
		    self.b_height, self.in_channels)).cuda()
		if logscale:
			self.freqs = 2**torch.linspace(0, N_freqs-1, N_freqs)
		else:
			self.freqs = torch.linspace(1, 2**(N_freqs-1), N_freqs)

	def forward(self, x):
		"""
		Input is a 2x1 vector and output is 3x21 (if N_freq is 10)
		Idea is that we project it to a higer dimension to be able to
		encode higher frequencies.
		[x] -> [x,sin(x),cos(x),sin(2x),cos(2x),......]
		Inputs:
		x : (Batch,3)
		Outputs:
		out : (Batch, 6*N_freqs+3)
		"""
		out = x@self.B.T
		pi = torch.acos(torch.zeros(1)).item() * 2
		inp = x@self.B.T
		for freq in self.freqs:
			for func in self.functions:
				out = torch.cat((out, func(freq*inp*pi)), axis=1)

		return out


class NeRF(nn.Module):
    def __init__(self, in_channels, width=256):
        """
        W: number of hidden units in each layer
        """
        super(NeRF, self).__init__()
        self.W = width
        self.in_channels = in_channels
        self.layer1 = nn.Sequential(nn.Linear(self.in_channels, self.W),
                                    nn.ReLU(False))
        self.layer2 = nn.Sequential(nn.Linear(self.W, self.W),
                                    nn.ReLU(False))
        self.layer3 = nn.Sequential(nn.Linear(self.W, self.W),
                                    nn.ReLU(False))
        self.output = nn.Sequential(nn.Linear(self.W, 3),
                                    nn.Sigmoid())

    def forward(self, x):

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        out = self.output(l3)
        return out


class SineLayer(nn.Module):
	"""
	if is_first = True (first layer),omerga_0 is a frequency factor which multiplies
	the activations before non-linearity. This is a hyperparamter.
	For all other layers, weights will be divided by omega_0 to keep the magnitude of
	activations constant but boost gradients
	"""

	def __init__(self, in_channels, out_channels, bias=True, is_first=False, omega_0=30):
		"""
		in_channels:
		out_channels:
		bias:
		is_first:
		omega_0:
		"""
		super(SineLayer, self).__init__()
		self.in_channels = in_channels
		self.omega_0 = omega_0
		self.is_first = is_first
		self.linear = nn.Linear(in_channels, out_channels, bias=bias)
		self.init_weights()

	def init_weights(self):
		with torch.no_grad():
			if self.is_first:
				self.linear.weight.uniform_(-1 / self.in_channels,
                                             1 / self.in_channels)
			else:
				self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)

	def forward(self, x):
		return torch.sin(self.omega_0*self.linear(x))


class Siren(nn.Module):

	def __init__(self, in_channels, width=256, out_channels=3, outermost_linear=False, first_omega=30, hidden_omega=30):
		"""
		in_channels:
		width:
		out_channels:
		outermost_linear=False:
		first_omega:
		hidden_omega:
		"""
		super(Siren, self).__init__()
		self.in_channels = in_channels
		self.width = width
		self.out_channels = out_channels
		self.outermost_linear = outermost_linear
		self.first_omega = first_omega
		self.hidden_omega = hidden_omega

		self.layer1 = SineLayer(self.in_channels, self.width,
		                        is_first=True, omega_0=self.first_omega)
		self.layer2 = SineLayer(self.width, self.width,
		                        is_first=False, omega_0=self.hidden_omega)
		self.layer3 = SineLayer(self.width, self.width,
		                        is_first=False, omega_0=self.hidden_omega)

		if self.outermost_linear:
			self.final_linear = nn.Linear(self.width, self.out_channels)
			with torch.no_grad():
				self.final_linear.weight.uniform_(-np.sqrt(6 / self.width) / self.hidden_omega,
                                              np.sqrt(6 / self.width) / self.hidden_omega)
		else:
			self.final_linear = SineLayer(
                self.width, out_channels, is_first=False, omega_0=hidden_omega)

	def forward(self,x):
		x = x.clone().detach().requires_grad_(True)
		l1 = self.layer1(x)
		l2 = self.layer2(l1)
		l3 = self.layer3(l2)
		out = self.final_linear(l3)
		return out

if __name__ == '__main__':
	batch = 256
	xyz_L = 10
	# xyz_embedding = GaussPosEmbedding(2,xyz_L)
	input_x  = torch.rand(batch,2)
	# pos_x = xyz_embedding(input_x)
	# print(pos_x.shape)
	model = Siren(in_channels=2,width=256,outermost_linear=True)
	out = model(input_x)
	print(out.shape)
	# dir_L = 4
	# dir_embedding = PosEmbedding(dir_L-1,dir_L)
	# input_dir  = torch.rand(batch,3)
	# pos_dir = dir_embedding(input_x)
	# print(pos_dir.shape)

	# nerf_input = torch.cat([pos_x,pos_dir],dim=1)
	# print(nerf_input.shape)
	# nerf_input = torch.rand(256,90)
	# nerf_gt = torch.rand(256,3)
	# model = NeRF(90,width=256)
	# nerf_output = model(nerf_input)
	# criterion = nn.MSELoss()
	# loss = criterion(nerf_output,nerf_gt)
	# loss.backward()
	# print(nerf_output.shape)

