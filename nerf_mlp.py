import torch
from torch import nn


class PosEmbedding(nn.Module):
	def __init__(self,max_logscale,N_freqs,logscale=True):
		super(PosEmbedding).__init__()
		self.functions = [torch.sin,torch.cos]

		if logscale:
			self.freqs = 2**torch.linspace(0,max_logscale,N_freqs)
		else:
			self.freqs = torch.linspace(1,2**max_logscale,N_freqs)

	def forward(self,x):
		out = [x]
		for freq in self.freqs:
			for func in self.functions:
				out += [func(freq*x)]

		return torch.cat(out,-1)


class NeRF(nn.Module):

	def __init__(self,t)

if __name__ == '__main__':
	