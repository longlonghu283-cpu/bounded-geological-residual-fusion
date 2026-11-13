"""Reconstructed C2: fixed relative RMS cap, no auxiliary objectives."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import UNet

KAPPA=0.01
EPS=1e-8

def channel_rms(x):
    # Algebraically identical RMS with FP32 rescaling to avoid squared-norm
    # underflow for near-zero features. Detached scaling introduces no norm path.
    a=x.float();scale=a.detach().abs().amax(dim=1,keepdim=True)
    safe=torch.where(scale>0,scale,torch.ones_like(scale))
    return torch.linalg.vector_norm(a/safe,dim=1,keepdim=True)*safe/math.sqrt(x.shape[1])

def cap_residual(h,r):
    with torch.autocast(device_type=h.device.type,enabled=False):
        reference=channel_rms(h.detach())
        raw=channel_rms(r)
        alpha=torch.clamp(KAPPA*reference/(raw+EPS),max=1.)
        bounded=r.float()*alpha
    return bounded,alpha,reference,raw

class C2Fusion(nn.Module):
    def __init__(self,n_embeddings):
        super().__init__()
        self.embedding=nn.Embedding(n_embeddings,8,padding_idx=0)
        self.encoder=nn.Sequential(nn.Conv2d(10,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU())
        self.projection=nn.Conv2d(16,64,1)
        self.gate=nn.Conv2d(80,64,1)
        self.default_projection=(self.projection.weight.detach().clone(),self.projection.bias.detach().clone())
        self.variant='joint'
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.last={}
    def forward(self,h,geo,record=False):
        strata,distance,valid=geo
        z=self.embedding(strata.long()).permute(0,3,1,2)
        z=self.encoder(torch.cat([z,distance.to(z.dtype),valid.to(z.dtype)],dim=1))
        z=F.interpolate(z,size=h.shape[-2:],mode='nearest')
        mask=F.interpolate(valid.float(),size=h.shape[-2:],mode='nearest')
        gate=torch.sigmoid(self.gate(torch.cat([h,z],dim=1)))
        if self.variant=='no_gate':gate=torch.ones_like(gate)
        raw=mask*gate.float()*self.projection(z).float()
        bounded,alpha,reference,raw_rms=cap_residual(h,raw)
        if self.variant=='no_cap':bounded=raw.float();alpha=torch.ones_like(alpha)
        output=(h.float()+bounded).to(h.dtype)
        if record:
            with torch.no_grad():
                self.last=dict(valid=mask.detach(),gate=gate.detach().float(),raw_rms=raw_rms.detach(),bounded_rms=channel_rms(bounded).detach(),HSI_rms=reference.detach(),alpha=alpha.detach(),quantized_delta_rms=channel_rms(output.float()-h.float()).detach(),invalid_residual_max=float(bounded.masked_select((mask==0).expand_as(bounded)).abs().max()) if (mask==0).any() else 0.)
        return output

class C2GANUNet(nn.Module):
    def __init__(self,backbone,n_embeddings):
        super().__init__();self.backbone=backbone;self.geo=C2Fusion(n_embeddings);self.geo_enabled=True;self.record=False
    def forward(self,x,geology=None):
        b=self.backbone;x=b.spectral_gate(x)
        e1=b.enc1(x);e2=b.enc2(b.pool(e1));e3=b.enc3(b.pool(e2));v=b.bottleneck(b.pool(e3))
        v=b.dec3(v,e3);v=b.dec2(v,e2)
        if self.geo_enabled:
            if geology is None:raise ValueError('GEOLOGY_REQUIRED_WHEN_ENABLED')
            v=self.geo(v,geology,self.record)
        v=b.dec1(v,e1)
        return b.out(v)
