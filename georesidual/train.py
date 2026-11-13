"""Opt-in 300-update runner for prepared, authorised inputs. Never runs on import."""
import argparse, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from . import e3_engine as e3, e4_engine as e4, controls

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['e3','e4'],required=True)
    p.add_argument('--model',choices=['unet','gan_unet','gcn_unet','convlstm_unet','segformer_b0'],default='gan_unet')
    p.add_argument('--variant',choices=['joint','hsi','strata','fault','noise','no_gate','nonzero_init','no_cap'],default='joint')
    p.add_argument('--seed',type=int,choices=[42,7,2026],required=True)
    p.add_argument('--fold',type=int,choices=range(4),required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('Frozen training requires CUDA with bfloat16 support; use smoke tests on CPU.')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=a.stage=='e4'
    torch.use_deterministic_algorithms(a.stage=='e4')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    e3.DATA=e4.DATA=a.data.resolve()
    if a.stage=='e3':
        pack=e3.make_pack(a.model,a.seed,a.fold);data=e3.RealData(a.fold)
    else:
        controls.ROOT=a.out.resolve()
        if a.variant=='noise':
            dest=a.out/'controls';dest.mkdir()
            lock=e4.read(e4.DATA/f'FOLD{a.fold}/GEOLOGY_LOCK.json')
            shape=np.load(e4.DATA/'COMMON_VALID.npy',mmap_mode='r').shape
            rng=np.random.default_rng(e4.stream(20261025,a.fold,'capacity_inputs_independent'))
            np.savez_compressed(dest/f'FOLD{a.fold}.npz',strata=rng.integers(2,lock['n_embeddings'],size=shape,dtype=np.int16),distance=rng.standard_normal(shape,dtype=np.float32))
        pack=controls.apply_variant(e4.make_pack(a.seed,a.fold),a.variant)
        data=controls.VariantData(a.fold,a.variant)
    best=None;selected=None
    for update in range(1,301):
        batch,trace=data.batch(pack)
        row=(e3.train_step if a.stage=='e3' else e4.step)(pack,batch,update,data.manifest['positive_class_weight'])
        with (a.out/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if update%30:continue
        if a.stage=='e3':
            m,keys,labels,prob=e3.evaluate(pack,data);v=dict(keys=keys,labels=labels,probabilities=prob);teacher=pack['teacher']
        else:
            m,v=e4.evaluate(pack,data);e4.independent_metrics(v,m);teacher=pack['t']
        rank=(m['AP'],-m['validation_loss'],-update)
        if best is None or rank>best:
            best=rank;selected=dict(update=update,**m)
            torch.save(teacher.state_dict(),a.out/'selected_teacher.pt')
            np.savez_compressed(a.out/'selected_DEV.npz',**v)
        (a.out/'result.json').write_text(json.dumps(dict(stage=a.stage,model=a.model,variant=a.variant,seed=a.seed,fold=a.fold,completed_updates=update,selected=selected),indent=2),encoding='utf-8')
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
