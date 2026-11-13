"""Input and component controls; real-attribute and noise controls share coverage."""
from pathlib import Path
import numpy as np
import torch
from . import e4_engine as rt
from .e4_engine import Data, stream
ROOT=Path(".")
def apply_variant(p,variant):
    rt.ACTIVE_VARIANT=variant
    for model in [p['s'],p['t']]:
        model.geo.variant=variant
        model.geo_enabled=variant!='hsi'
        if variant=='nonzero_init':
            with torch.no_grad():
                model.geo.projection.weight.copy_(model.geo.default_projection[0]);model.geo.projection.bias.copy_(model.geo.default_projection[1])
    return p

class VariantData(Data):
    def __init__(self,fold,variant):
        super().__init__(fold);self.variant=variant
        if variant=='noise':
            values=np.load(ROOT/'controls'/f'FOLD{fold}.npz',allow_pickle=False)
            self.fake_strata=values['strata'];self.fake_distance=values['distance']
    def patch(self,w):
        a=super().patch(w)
        if self.variant=='strata':a[4]=np.zeros_like(a[4])
        elif self.variant=='fault':a[3]=np.where(a[5][0]>0,1,0).astype('int64')
        elif self.variant=='noise':
            y,x=w['y'],w['x'];ss=np.s_[y:y+64,x:x+64]
            a[3]=np.where(a[5][0]>0,self.fake_strata[ss],0).astype('int64')
            a[4]=np.asarray(self.fake_distance[ss],dtype='float32')[None]*a[5]
        return a
