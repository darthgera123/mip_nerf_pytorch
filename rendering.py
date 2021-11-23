import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
	"""
	Importance sample the points on the ray
	Distribution defined by weights
	Inputs:
		bins : (N_rays,N_samples_+1) N_samples_ = No of coarse samples per ray-2
		weights : (N_rays, N_samples_ )
		N_importance : No of samples to draw from the distribution
		det : deterministic or not
		eps: a small no
	Outputs:
		samples: the sampled samples
	"""
	NN_rays, N_samples_ = weights.shape
	weights = weights + eps # prevent division by zero (don't do inplace op!)
	pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
	cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
	cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

	if det:
		u = torch.linspace(0, 1, N_importance, device=bins.device)
		u = u.expand(N_rays, N_importance)
	else:
		u = torch.rand(N_rays, N_importance, device=bins.device)
	u = u.contiguous()

	inds = torch.searchsorted(cdf, u, right=True)
	below = torch.clamp_min(inds-1, 0)
	above = torch.clamp_max(inds, N_samples_)

	inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
	cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
	bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

	denom = cdf_g[...,1]-cdf_g[...,0]
	denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
	                     # anyway, therefore any value for it is fine (set to 1 here)

	samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
	return samples

def rendering():
	pass

if __name__ == '__main__':
	N_rays = 3*3
	N_samples = 3

	bins =  torch.rand((N_rays,N_samples+1))
	weights = torch.rand((N_rays,N_samples))
	N_importance = 10
	det = True 
	eps = 1e-8

	samples = sample_pdf(bins,weights,N_importance,det,eps)
	print(samples)
