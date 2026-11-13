"""Research functions retained from the frozen implementation; paths are configurable."""
import os, copy, json, hashlib, random, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, f1_score, jaccard_score
DATA=Path("data")
from .backbone import UNet, MaskDiscriminator
from .fusion import C2GANUNet
ACTIVE_VARIANT="joint"
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def stream(seed,fold,name):return int.from_bytes(hashlib.sha256(f'{seed}|{fold}|{name}'.encode()).digest()[:8],'little')%(2**63-1)

def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def state_sha(a):
    if hasattr(a,'state_dict'):a=a.state_dict()
    h=hashlib.sha256()
    for k,v in sorted(a.items()):h.update(k.encode());h.update(tsha(v).encode())
    return h.hexdigest()

def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def make_pack(seed,fold):
    lock=read(DATA/f'FOLD{fold}/GEOLOGY_LOCK.json');seed_all(seed)
    backbone=UNet(158).cuda();disc=MaskDiscriminator().cuda()
    before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state_all()
    # Geology construction cannot perturb backbone, discriminator or data RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(stream(seed,fold,'geology_initialization'));model=C2GANUNet(backbone,lock['n_embeddings']).cuda()
    assert torch.equal(before_cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(before_cuda,torch.cuda.get_rng_state_all()))
    teacher=copy.deepcopy(model).requires_grad_(False).eval()
    args=dict(lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'group_name':'HSI_backbone'},{'params':model.geo.parameters(),'group_name':'geology_fusion'}],**args)
    od=torch.optim.AdamW(disc.parameters(),**args);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300,eta_min=0.)
    streams={n:np.random.default_rng(stream(seed,fold,n)) for n in ['P','U','geometry_P','geometry_U','synthetic']}
    noise=torch.Generator(device='cuda').manual_seed(stream(seed,fold,'spectral_noise'))
    return dict(s=model,t=teacher,d=disc,o=opt,od=od,sched=sched,streams=streams,noise=noise,seed=seed,fold=fold)

def capture(p):return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),streams={n:copy.deepcopy(g.bit_generator.state) for n,g in p['streams'].items()},noise=p['noise'].get_state())

def restore_rng(p,r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
    for n,s in r['streams'].items():p['streams'][n].bit_generator.state=s
    p['noise'].set_state(r['noise'].cpu())

def geometry(arrays,g):
    k,hf,vf=g;out=[]
    for a in arrays:
        a=np.rot90(a,k,axes=(-2,-1))
        if hf:a=np.flip(a,-1)
        if vf:a=np.flip(a,-2)
        out.append(np.ascontiguousarray(a))
    return out

def geotensor(arrays):return tuple(torch.from_numpy(np.stack(a)).cuda() for a in zip(*arrays))

class Data:
    def __init__(self,fold):
        self.fold=fold;self.manifest=read(DATA/f'FOLD{fold}/MANIFEST.json');self.lock=read(DATA/f'FOLD{fold}/GEOLOGY_LOCK.json')
        self.hsi=np.load(DATA/'HSI_158_FLOAT32.npy',mmap_mode='r');self.valid=np.load(DATA/'COMMON_VALID.npy',mmap_mode='r');self.label=np.load(DATA/'R225_LABEL.npy',mmap_mode='r');self.support=np.load(DATA/'EVALUATION_SUPPORT.npy',mmap_mode='r')
        self.strata=np.load(DATA/'MAP_UNIT_CODES.npy',mmap_mode='r');self.distance=np.load(DATA/'FAULT_DISTANCE_M.npy',mmap_mode='r');self.geo_valid=np.load(DATA/'GEO_VALID.npy',mmap_mode='r')
        self.lut=np.ones(int(self.strata.max())+1,dtype='int64');self.lut[0]=0
        for k,v in self.lock['raw_code_to_index'].items():self.lut[int(k)]=v
        norm=np.load(DATA/f'FOLD{fold}/NORMALIZATION.npz');self.mean=norm['mean'][:,None,None];self.std=norm['std'][:,None,None];self.expected=np.load(DATA/f'FOLD{fold}/DEV_SUPPORT.npz')
    def patch(self,w):
        y,x=w['y'],w['x'];ss=np.s_[y:y+64,x:x+64];v=np.asarray(self.valid[ss],dtype='float32')[None]
        h=np.clip((self.hsi[:,ss[0],ss[1]]-self.mean)/self.std,-5,5).astype('float32')*v
        mask=np.asarray(self.geo_valid[ss],dtype='float32')[None];indices=self.lut[self.strata[ss]].copy();indices[mask[0]==0]=0
        distance=np.nan_to_num((np.asarray(self.distance[ss],dtype='float32')-self.lock['fault_mean_m'])/self.lock['fault_std_m'],nan=0.)[None]*mask
        return [h,np.asarray(self.label[ss],dtype='float32')[None],v,indices,distance,mask]
    def batch(self,p):
        ps=p['streams']['P'].integers(len(self.manifest['train_P']),size=4).tolist();us=p['streams']['U'].integers(len(self.manifest['train_U']),size=8).tolist();groups=[];trace=dict(P_indices=ps,U_indices=us)
        for role,indices,key in [('P',ps,'train_P'),('U',us,'train_U')]:
            arrays=[];gs=[];rng=p['streams']['geometry_'+role]
            for i in indices:
                g=[int(rng.integers(4)),bool(rng.random()<.5),bool(rng.random()<.5)];gs.append(g);arrays.append(geometry(self.patch(self.manifest[key][i]),g))
            groups.append([torch.from_numpy(np.stack(a)).cuda() for a in zip(*arrays)]);trace[role+'_geometry']=gs
        trace['noise_state_sha256']=tsha(p['noise'].get_state());P,U=groups
        return (P[0],P[1],P[2],U[0],U[2],tuple(P[3:]),tuple(U[3:])),trace

def synthetic(p):
    rng=p['streams']['synthetic'];h=torch.from_numpy(rng.standard_normal((4,158,64,64),dtype=np.float32)).cuda();u=torch.from_numpy(rng.standard_normal((8,158,64,64),dtype=np.float32)).cuda();y=torch.zeros((4,1,64,64),device='cuda');y[:,:,22:40,24:42]=1
    def geo(n):
        codes=torch.from_numpy(rng.integers(0,p['s'].geo.embedding.num_embeddings,size=(n,64,64),dtype='int64')).cuda();mask=(codes>0).float()[:,None];d=torch.from_numpy(rng.standard_normal((n,1,64,64),dtype=np.float32)).cuda()*mask;return codes,d,mask
    return h,y,torch.ones_like(y),u,torch.ones((8,1,64,64),device='cuda'),geo(4),geo(8)

def supervised(logits,y,v,pw):
    logits=logits.float();y=y.float();v=v.float();bce=(F.binary_cross_entropy_with_logits(logits,y,pos_weight=logits.new_tensor(pw),reduction='none')*v).sum()/v.sum().clamp_min(1)
    prob=torch.sigmoid(logits)*v;t=y*v;dice=1-((2*(prob*t).sum((1,2,3))+1)/(prob.sum((1,2,3))+t.sum((1,2,3))+1)).mean();return bce+dice,bce,dice

def gradnorm(m):
    norm=torch.zeros((),device='cuda')
    for a in m.parameters():
        if a.grad is not None:
            if not torch.isfinite(a.grad).all():raise RuntimeError('NONFINITE_GRADIENT')
            norm+=a.grad.detach().float().square().sum()
    return float(norm.sqrt())

def stat_summary(stats):
    if not stats:return dict(valid_geo_fraction=0.,cap_trigger_fraction=0.,gate_mean=0.,residual_HSI_ratio_mean=0.,residual_HSI_ratio_max=0.)
    mask=stats['valid']>0
    if not mask.any():return dict(valid_geo_fraction=0.,cap_trigger_fraction=0.,gate_mean=0.,residual_HSI_ratio_mean=0.,residual_HSI_ratio_max=0.)
    ratio=stats['bounded_rms']/stats['HSI_rms'].clamp_min(1e-8)
    if ACTIVE_VARIANT!='no_cap':assert float(ratio[mask].max())<=.0100002,'C2_CAP_VIOLATION'
    assert stats['invalid_residual_max']==0.,'INVALID_GEO_INJECTION'
    return dict(valid_geo_fraction=float(mask.float().mean()),cap_trigger_fraction=float((stats['alpha'][mask]<1).float().mean()),gate_mean=float(stats['gate'].masked_select(mask.expand_as(stats['gate'])).mean()),residual_HSI_ratio_mean=float(ratio[mask].mean()),residual_HSI_ratio_max=float(ratio[mask].max()))

def step(p,b,update,pw,commit=True):
    x,y,pv,u,uv,gp,gu=b;s,t,d=p['s'],p['t'],p['d'];s.train();s.record=True;t.eval();d.train();p['o'].zero_grad(set_to_none=True);p['od'].zero_grad(set_to_none=True);gen=p['noise']
    weak=(u+.01*torch.randn(u.shape,device='cuda',generator=gen))*uv;scale=torch.empty((8,1,1,1),device='cuda').uniform_(.9,1.1,generator=gen)
    strong=(u*scale+.05*torch.randn(u.shape,device='cuda',generator=gen))*uv;strong=strong.masked_fill(torch.rand((8,158,1,1),device='cuda',generator=gen)<.08,0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        lp=s(x,gp)
        with torch.no_grad():lt=t(weak,gu)
        lu=s(strong,gu)
    if not all(torch.isfinite(a).all() for a in [lp,lt,lu]):raise RuntimeError('NONFINITE_FORWARD')
    sup,bce,dice=supervised(lp,y,pv,pw);tp=torch.sigmoid(lt.float());up=torch.sigmoid(lu.float());mask=((tp>=.7)|(tp<=.3))&(uv>0)
    cons=(tp[mask]-up[mask]).square().mean() if mask.any() else lu.float().sum()*0
    with torch.autocast('cuda',dtype=torch.bfloat16):score=d(torch.sigmoid(lp))
    adv=F.binary_cross_entropy_with_logits(score.float(),torch.ones_like(score,dtype=torch.float32));lam=.2*math.exp(-5*(1-min((update-1)/180,1))**2);loss=sup+lam*cons+.01*adv
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
    loss.backward();gn=gradnorm(s);projection_grad=gradnorm(s.geo.projection);gate_grad=gradnorm(s.geo.gate);embedding_grad=gradnorm(s.geo.embedding)
    p['od'].zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):real=d(y);fake=d(torch.sigmoid(lp.detach()))
    dl=.5*(F.binary_cross_entropy_with_logits(real.float(),torch.ones_like(real,dtype=torch.float32))+F.binary_cross_entropy_with_logits(fake.float(),torch.zeros_like(fake,dtype=torch.float32)))
    if not torch.isfinite(dl):raise RuntimeError('NONFINITE_D_LOSS')
    dl.backward();dn=gradnorm(d)
    if commit:
        p['o'].step();p['od'].step();p['sched'].step()
        with torch.no_grad():
            for a,c in zip(t.parameters(),s.parameters()):a.mul_(.99).add_(c,alpha=.01)
            for a,c in zip(t.buffers(),s.buffers()):a.copy_(c)
    row=dict(update=update,loss=float(loss.detach()),supervised=float(sup.detach()),BCE=float(bce.detach()),Dice=float(dice.detach()),consistency=float(cons.detach()),consistency_weight=lam,GAN_generator=float(adv.detach()),GAN_discriminator=float(dl.detach()),generator_gradient_l2=gn,discriminator_gradient_l2=dn,projection_gradient_l2=projection_grad,gate_gradient_l2=gate_grad,embedding_gradient_l2=embedding_grad,confidence_fraction=float(mask.float().mean()),learning_rate=p['o'].param_groups[0]['lr'],Student_committed=int(commit),GAN_committed=int(commit),EMA_committed=int(commit))
    row.update(stat_summary(s.geo.last));s.geo.last={};return row

def metric(y,prob):
    assert np.isfinite(prob).all() and set(np.unique(y))=={0,1}
    order=np.argsort(-prob,kind='stable');p=prob[order];labels=y[order];ends=np.r_[np.where(p[:-1]!=p[1:])[0],len(y)-1];tp=np.cumsum(labels,dtype='int64')[ends];fp=ends+1-tp;fn=int(y.sum())-tp;fs=2*tp/(2*tp+fp+fn);j=int(np.argmax(fs));threshold=float(p[ends[j]]);pred=prob>=threshold
    TP=int((pred&(y==1)).sum());FP=int((pred&(y==0)).sum());FN=int((~pred&(y==1)).sum())
    return dict(AP=float(average_precision_score(y,prob)),F1=2*TP/(2*TP+FP+FN),IoU=TP/(TP+FP+FN),Precision=TP/(TP+FP),Recall=TP/(TP+FN),threshold=threshold,TP=TP,FP=FP,FN=FN,n=len(y),positives=int(y.sum()),background=int((y==0).sum()))

@torch.no_grad()
def evaluate(p,data):
    t=p['t'];t.eval();kk=[];pp=[];losses=[];width=data.valid.shape[1];rows=data.manifest['val_P']
    for start in range(0,len(rows),4):
        ws=rows[start:start+4];a=[data.patch(w) for w in ws];x,y,v,codes,distance,valid=[torch.from_numpy(np.stack(z)).cuda() for z in zip(*a)]
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=t(x,(codes,distance,valid))
        loss,*_=supervised(logits,y,v,data.manifest['positive_class_weight']);losses.append(float(loss));probs=torch.sigmoid(logits.float()).cpu().numpy()[:,0]
        for w,pr in zip(ws,probs):
            r,c=w['y'],w['x'];sup=data.support[r:r+64,c:c+64];mask=sup>0;yy,xx=np.where(mask);kk.append((yy+r)*width+xx+c);pp.append(pr[mask])
    keys,inv=np.unique(np.concatenate(kk),return_inverse=True);prob=np.bincount(inv,weights=np.concatenate(pp))/np.bincount(inv);y=data.expected['labels'];assert np.array_equal(keys,data.expected['keys'])
    m=metric(y,prob);m['validation_loss']=float(np.mean(losses));return m,dict(keys=keys,labels=y,probabilities=prob)

def independent_metrics(vectors,record):
    y,p=vectors['labels'],vectors['probabilities'];pred=p>=record['threshold'];tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    vals=dict(AP=average_precision_score(y,p),F1=f1_score(y,pred),IoU=jaccard_score(y,pred),TP=int(tp),FP=int(fp),FN=int(fn),Precision=tp/(tp+fp),Recall=tp/(tp+fn))
    assert all(abs(record[k]-v)<1e-10 for k,v in vals.items())
    prec,rec,thr=precision_recall_curve(y,p);f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-30);chosen=thr[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]];assert abs(record['threshold']-chosen)<1e-12
