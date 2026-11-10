"""Background-only control and fixed spherical local optimal interpolation."""
import math
import torch
from offgrid_da.models.full_state_t2m import FullStateT2MAnalysis


class FullStateCorrection(FullStateT2MAnalysis):
    """Same template/initial convolution weights; attention parameters are inactive."""
    def forward(self, background, observations):
        x = self.input(observations['grid_state'])
        for block in self.blocks:
            x = x + block(x)
        return {'analysis': background + self.output(x)}


def shuffled_observations(obs, generator):
    result = {k: v.clone() for k, v in obs.items()}
    for i, mask in enumerate(obs['observation_mask']):
        indices = mask.nonzero().flatten()
        order = torch.randperm(len(indices), generator=generator).to(indices.device)
        result['obs_values_normalized'][i, indices] = obs['obs_values_normalized'][i, indices[order]]
    return result


def local_oi(background, obs, std, sigma_b_K=1., sigma_o_K=2.5,
             length_km=500., radius_km=1000., neighbors=16, chunk=256):
    """Local OI, exponential great-circle B covariance, diagonal R.

    Fixed a priori parameters, no validation tuning. Observations are temperature
    normalized with the same scale as background. Periodic bilinear H.
    """
    from offgrid_da.models.analysis import CrossAttentionAnalysisModel as Base
    b, _, h, w = background.shape
    lat, lon = torch.meshgrid(torch.linspace(90, -90, h, device=background.device),
                             torch.arange(w, device=background.device)*360/w, indexing='ij')
    grid = Base._xyz(lat.flatten(), lon.flatten()).double()
    out = background.clone()
    for batch in range(b):
        mask = obs['observation_mask'][batch]
        loc = obs['obs_locations'][batch, :, :2]
        mask = mask & torch.isfinite(loc).all(-1) & (loc[:, 0].abs() <= 90)
        loc = loc[mask].double()
        if not len(loc):
            continue
        xyz = Base._xyz(loc[:, 0], loc[:, 1])
        # Bilinear interpolation with a repeated longitude seam.
        col = loc[:, 1].remainder(360)*w/360
        row = (90-loc[:, 0])*(h-1)/180
        r0, c0 = row.floor().long(), col.floor().long()
        r1, c1 = (r0+1).clamp_max(h-1), (c0+1)%w
        fr, fc = row-r0, col-c0
        bg = background[batch, 0].double()
        hb = (1-fr)*((1-fc)*bg[r0,c0]+fc*bg[r0,c1])+fr*((1-fc)*bg[r1,c0]+fc*bg[r1,c1])
        innovation = (obs['obs_values_normalized'][batch, mask].double().reshape(-1)-hb)*std
        increments = []
        for start in range(0, h*w, chunk):
            sim, ix = (grid[start:start+chunk] @ xyz.T).topk(min(neighbors,len(loc)), dim=1)
            distance = sim.clamp(-1,1).acos()*6371.
            valid = distance <= radius_km
            chosen = xyz[ix]
            pair = (chosen @ chosen.transpose(-1,-2)).clamp(-1,1).acos()*6371.
            cov = sigma_b_K**2 * torch.exp(-pair/length_km)
            cov = cov * (valid[:,:,None] & valid[:,None,:])
            cov = cov + torch.eye(cov.shape[-1], device=bg.device)[None]*(sigma_o_K**2+1e-6)
            cross = sigma_b_K**2 * torch.exp(-distance/length_km) * valid
            coefficient = torch.linalg.solve(cov, (innovation[ix]*valid).unsqueeze(-1)).squeeze(-1)
            increments.append((cross*coefficient).sum(-1)/std)
        out[batch,0] += torch.cat(increments).reshape(h,w).to(out.dtype)
    return out
