"""Training-error-balanced full-state loss and transparent sufficient statistics."""
import json
from pathlib import Path
import numpy as np
import torch

GROUPS=[(str(i),[i]) for i in range(4)]+[(str(i),list(range(i,i+13))) for i in range(4,82,13)]

def fit_weights(mse,relative_floor=.1,absolute_floor=1e-6):
    mse=np.asarray(mse,dtype=np.float64)
    if mse.shape!=(82,) or not np.isfinite(mse).all() or not (mse>0).all():
        raise ValueError('Positive finite training-only error moments required')
    scale=np.empty(82);weights=np.empty(82)
    for _,ix in GROUPS:
        floor=max(absolute_floor,relative_floor*np.sqrt(mse[ix].mean()))
        scale[ix]=np.maximum(np.sqrt(mse[ix]),floor)
        group_mean=np.mean(mse[ix]/scale[ix]**2)
        weights[ix]=.1/len(ix)/scale[ix]**2/group_mean
    return weights,scale

def latitude_weights(x):
    return torch.cos(torch.deg2rad(torch.linspace(90,-90,x.shape[-2],device=x.device))).clamp_min(0)[None,None,:,None]

def balanced_loss(prediction,target,weights):
    lat=latitude_weights(prediction)
    w=torch.as_tensor(weights,dtype=prediction.dtype,device=prediction.device)[None,:,None,None]
    # Sum variable contributions, average only across samples and area.
    return ((prediction-target).square()*lat*w).sum()/lat.expand(prediction.shape[0],1,*prediction.shape[-2:]).sum()

def metrics(sse,denominator,weights,physical_std,names):
    mse=np.asarray(sse,dtype=np.float64)/denominator
    weights=np.asarray(weights)
    contributions={names[ix[0]].split('@')[0]:float((mse[ix]*weights[ix]).sum()) for _,ix in GROUPS}
    total=float((mse*weights).sum())
    return {'score':float(np.sqrt(total)),'group_weighted_mse':contributions,
            'group_fraction':{k:v/max(total,1e-30) for k,v in contributions.items()},
            'physical_rmse_by_channel':dict(zip(names,(np.sqrt(mse)*physical_std).tolist()))}
