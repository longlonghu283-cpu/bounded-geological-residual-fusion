"""CPU-only validation on generated tensors. No training or research data access."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from georesidual.models import create_model
from georesidual.backbone import UNet
from georesidual.fusion import C2GANUNet, cap_residual, channel_rms
from georesidual.e3_engine import metric as metric3
from georesidual.e4_engine import metric as metric4, supervised
from sklearn.metrics import average_precision_score

torch.set_num_threads(2);torch.manual_seed(123)
x=torch.randn(1,158,64,64);checks={}
with torch.no_grad():
    for name in ['unet','gan_unet','gcn_unet','convlstm_unet','segformer_b0']:
        model=create_model(name).eval();result=model(x)
        assert result.shape==(1,1,64,64) and torch.isfinite(result).all()
        checks[name]='finite 64x64 output';del model
    base=UNet(158).eval();model=C2GANUNet(base,6).eval()
    codes=torch.randint(2,6,(1,64,64));distance=torch.randn(1,1,64,64);valid=torch.ones_like(distance);valid[:,:,:8,:8]=0
    geo=(codes,distance,valid)
    assert torch.equal(base(x),model(x,geo));checks['zero_initialisation']='exact backbone output'
    torch.nn.init.normal_(model.geo.projection.weight,std=.2)
    model.record=True;model(x,geo);stats=model.geo.last
    assert stats['invalid_residual_max']==0
    ratio=stats['bounded_rms']/stats['HSI_rms'].clamp_min(1e-8)
    assert float(ratio.max())<=.0100002
    checks['maximum_relative_RMS']=float(ratio.max())
    model.geo_enabled=False;assert torch.equal(base(x),model(x));checks['HSI_bypass']='exact backbone output'
    model.geo_enabled=True;model.geo.variant='no_gate';model(x,geo)
    assert torch.all(model.geo.last['gate']==1);checks['no_gate']='unit gate'
    model.geo.variant='no_cap';model(x,geo)
    assert torch.equal(model.geo.last['raw_rms'],model.geo.last['bounded_rms']);checks['no_cap']='raw residual retained'
for magnitude in [0.,1e-20,1.,1e10]:
    h=torch.randn(1,64,8,8)*magnitude;r=torch.randn_like(h,requires_grad=True)
    bounded,_,ref,_=cap_residual(h,r)
    assert torch.isfinite(bounded).all() and torch.all(channel_rms(bounded)<=.0100002*ref+1e-30)
    bounded.sum().backward();assert torch.isfinite(r.grad).all()
checks['RMS_extremes']='zero, tiny, normal, large reference and finite gradients'
labels=np.array([0,1,0,1,0,1]);prob=np.array([.1,.8,.8,.7,.1,.9])
m=metric3(labels,prob);assert m==metric4(labels,prob)
assert m['AP']==average_precision_score(labels,prob)
checks['metrics']='identical E3/E4 AP and threshold; tied scores checked'
logits=torch.randn(1,1,16,16,requires_grad=True);y=(torch.rand_like(logits)>.8).float()
loss,_,_=supervised(logits,y,torch.ones_like(y),4.)
loss.backward();assert torch.isfinite(logits.grad).all();checks['supervised_loss']='finite loss and gradients; no optimiser update'
print(json.dumps({'status':'PASS','scope':'synthetic CPU verification only','checks':checks},indent=2))
