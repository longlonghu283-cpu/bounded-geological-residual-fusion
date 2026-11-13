"""Sliding-window score averaging on a caller-supplied raster cache."""
import numpy as np
import torch

@torch.no_grad()
def predict_scene(model, data, device='cpu', stride=32, geology=True):
    if not 1<=stride<=64:raise ValueError('stride must be in [1,64]')
    height,width=data.valid.shape
    if min(height,width)<64:raise ValueError('Scene dimensions must be at least 64 pixels')
    model=model.to(device).eval();total=np.zeros((height,width),np.float64);count=np.zeros((height,width),np.uint32)
    ys=sorted(set(range(0,height-63,stride))|{height-64});xs=sorted(set(range(0,width-63,stride))|{width-64})
    for y in ys:
        for x in xs:
            a=data.patch(dict(y=y,x=x));image=torch.from_numpy(a[0][None]).to(device)
            geo=tuple(torch.from_numpy(z[None]).to(device) for z in a[3:]) if geology else None
            logits=model(image,geo) if geology else model(image)
            scores=torch.sigmoid(logits.float())[0,0].cpu().numpy();v=a[2][0]>0
            total[y:y+64,x:x+64]+=scores*v;count[y:y+64,x:x+64]+=v
    return np.divide(total,count,out=np.full_like(total,np.nan),where=count>0).astype('float32')
