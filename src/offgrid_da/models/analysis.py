"""Grid analysis with spherical, radius-limited observation attention."""
from __future__ import annotations
import math
import torch
from torch import nn
from .observation_encoder import ObservationEncoder

class CrossAttentionAnalysisModel(nn.Module):
    def __init__(self, channels=1, dim=64, heads=4, depth=2, injection='layered', radius_km=1000., max_neighbors=16, query_chunk=256):
        super().__init__()
        if injection not in {'entry','layered'} or radius_km <= 0 or max_neighbors < 1 or query_chunk < 1: raise ValueError('invalid attention configuration')
        self.injection=injection; self.radius_km=float(radius_km); self.max_neighbors=int(max_neighbors); self.query_chunk=int(query_chunk)
        self.observation_encoder=ObservationEncoder(dim=dim,use_innovation=False)
        self.input=nn.Conv2d(channels,dim,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,dim,3,padding=1)) for _ in range(depth)])
        self.attention=nn.ModuleList([nn.MultiheadAttention(dim,heads,batch_first=True) for _ in range(depth)])
        self.norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.output=nn.Conv2d(dim,channels,1); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    @staticmethod
    def _xyz(lat,lon):
        lat=torch.deg2rad(lat); lon=torch.deg2rad(lon)
        return torch.stack((lat.cos()*lon.cos(),lat.cos()*lon.sin(),lat.sin()),dim=-1)
    def _local(self,q,tokens,mask,locations,h,w,layer):
        b,n,d=tokens.shape
        if n==0: return torch.zeros_like(q)
        lat=torch.linspace(90,-90,h,device=q.device,dtype=torch.float32); lon=torch.arange(w,device=q.device,dtype=torch.float32)*(360./w)
        glat,glon=torch.meshgrid(lat,lon,indexing='ij'); grid=self._xyz(glat.reshape(-1),glon.reshape(-1))
        loc=locations.float(); valid=mask & torch.isfinite(loc).all(-1) & (loc[:,:,0].abs()<=90)
        safe=torch.where(valid.unsqueeze(-1),loc,torch.zeros_like(loc)); xyz=self._xyz(safe[:,:,0],safe[:,:,1])
        cosine=math.cos(min(self.radius_km/6371.0,math.pi)); k=min(n,self.max_neighbors); chunks=[]
        for start in range(0,h*w,self.query_chunk):
            end=min(start+self.query_chunk,h*w); sim=torch.einsum('qd,bnd->bqn',grid[start:end],xyz).masked_fill(~valid[:,None,:],-2.)
            dist,idx=sim.topk(k,dim=-1); good=dist>=cosine; count=end-start
            chosen=tokens[torch.arange(b,device=tokens.device)[:,None,None],idx]
            anygood=good.any(-1); safe_mask=good.clone(); safe_mask[...,0] |= ~anygood
            chosen=chosen.reshape(b*count,k,d); safe_mask=safe_mask.reshape(b*count,k)
            query=self.norms[layer](q[:,start:end]).reshape(b*count,1,d)
            update,_=self.attention[layer](query,chosen,chosen,key_padding_mask=~safe_mask,need_weights=False)
            chunks.append(update.reshape(b,count,d).masked_fill(~anygood.unsqueeze(-1),0))
        return torch.cat(chunks,dim=1)
    def forward(self,background,observations):
        encoded=self.observation_encoder(observations); tokens=encoded['tokens']; mask=encoded['observation_mask']
        locations=observations['obs_locations']; valid=mask & (locations[:,:,0].abs()<=90)
        x=self.input(background)
        for i,block in enumerate(self.blocks):
            x=x+block(x)
            if self.injection=='layered' or i==0:
                b,d,h,w=x.shape; q=x.flatten(2).transpose(1,2)
                update=self._local(q,tokens,valid,locations,h,w,i)
                x=x+update.transpose(1,2).reshape(b,d,h,w)
        increment=self.output(x).masked_fill(~valid.any(dim=1)[:,None,None,None],0)
        return {'analysis':background+increment,**encoded}
