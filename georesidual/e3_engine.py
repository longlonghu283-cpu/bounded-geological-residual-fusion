"""Research functions retained from the frozen implementation; paths are configurable."""
import os, copy, json, hashlib, random, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, f1_score, jaccard_score
DATA=Path("data")
from . import models
DEVICE=torch.device("cuda")
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def seed_stream(seed,fold,name):return int.from_bytes(hashlib.sha256(f'{seed}|{fold}|{name}'.encode()).digest()[:8],'little')%(2**63-1)

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def state_sha(model):
    h=hashlib.sha256()
    for k,t in model.state_dict().items():h.update(k.encode());h.update(str(tuple(t.shape)).encode());h.update(tensor_sha(t).encode())
    return h.hexdigest()

def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def capture(pack):
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),streams={k:copy.deepcopy(v.bit_generator.state) for k,v in pack['streams'].items()},noise=pack['noise'].get_state())

def restore(pack,rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'])
    torch.cuda.set_rng_state_all(rng['cuda'])
    for k,s in rng['streams'].items():pack['streams'][k].bit_generator.state=s
    pack['noise'].set_state(rng['noise'])

def make_pack(model,seed,fold):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    student=models.create_model(model,158).to(DEVICE);teacher=copy.deepcopy(student).requires_grad_(False).eval()
    opt=torch.optim.AdamW(student.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300,eta_min=0.)
    disc=models.MaskDiscriminator().to(DEVICE) if model=='gan_unet' else None
    dopt=torch.optim.AdamW(disc.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False) if disc is not None else None
    streams={k:np.random.default_rng(seed_stream(seed,fold,k)) for k in ['P','U','geometry_P','geometry_U','synthetic']}
    noise=torch.Generator(device=DEVICE).manual_seed(seed_stream(seed,fold,'spectral_noise'))
    return dict(student=student,teacher=teacher,optimizer=opt,scheduler=scheduler,discriminator=disc,discriminator_optimizer=dopt,streams=streams,noise=noise,model=model,seed=seed,fold=fold)

def supervised(logits,target,valid,weight):
    logits=logits.float();target=target.float();valid=valid.float()
    bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=logits.new_tensor(weight),reduction='none')
    bce=(bce*valid).sum()/valid.sum().clamp_min(1)
    p=torch.sigmoid(logits)*valid;t=target*valid
    dims=(1,2,3);dice=1-((2*(p*t).sum(dims)+1)/(p.sum(dims)+t.sum(dims)+1)).mean()
    return bce+dice,bce,dice

def geometry(a,rng):
    k=int(rng.integers(4));horizontal=bool(rng.random()<.5);vertical=bool(rng.random()<.5)
    out=[]
    for x in a:
        x=np.rot90(x,k,axes=(-2,-1))
        if horizontal:x=np.flip(x,axis=-1)
        if vertical:x=np.flip(x,axis=-2)
        out.append(np.ascontiguousarray(x))
    return out,[k,horizontal,vertical]

class RealData:
    def __init__(self,fold):
        self.cube=np.load(DATA/'HSI_158_FLOAT32.npy',mmap_mode='r');self.valid=np.load(DATA/'COMMON_VALID.npy',mmap_mode='r')
        self.label=np.load(DATA/'R225_LABEL.npy',mmap_mode='r');self.support=np.load(DATA/'EVALUATION_SUPPORT.npy',mmap_mode='r')
        root=DATA/f'FOLD{fold}';self.manifest=read(root/'MANIFEST.json');self.expected=np.load(root/'DEV_SUPPORT.npz')
        norm=np.load(root/'NORMALIZATION.npz');self.mean=norm['mean'][:,None,None];self.std=norm['std'][:,None,None]
    def patch(self,row):
        y,x=row['y'],row['x'];win=np.s_[y:y+64,x:x+64]
        a=np.clip((np.asarray(self.cube[:,win[0],win[1]])-self.mean)/self.std,-5,5).astype(np.float32)
        v=np.array(self.valid[win],dtype=np.float32)[None];a*=v
        return a,np.array(self.label[win],dtype=np.float32)[None],v
    def batch(self,pack):
        ps=pack['streams']['P'].integers(len(self.manifest['train_P']),size=4).tolist()
        us=pack['streams']['U'].integers(len(self.manifest['train_U']),size=8).tolist()
        p=[];y=[];pv=[];u=[];uv=[];pg=[];ug=[]
        for i in ps:
            (a,b,c),g=geometry(self.patch(self.manifest['train_P'][i]),pack['streams']['geometry_P'])
            p.append(a);y.append(b);pv.append(c);pg.append(g)
        for i in us:
            a,_,c=self.patch(self.manifest['train_U'][i]);(a,c),g=geometry((a,c),pack['streams']['geometry_U'])
            u.append(a);uv.append(c);ug.append(g)
        values=[torch.from_numpy(np.stack(a)).to(DEVICE) for a in [p,y,pv,u,uv]]
        trace=dict(P_indices=ps,U_indices=us,P_geometry=pg,U_geometry=ug,noise_state_sha256=tensor_sha(pack['noise'].get_state()))
        return values,trace

def synthetic_batch(pack):
    rng=pack['streams']['synthetic']
    p=torch.from_numpy(rng.standard_normal((4,158,64,64),dtype=np.float32)).to(DEVICE)
    u=torch.from_numpy(rng.standard_normal((8,158,64,64),dtype=np.float32)).to(DEVICE)
    y=torch.zeros(4,1,64,64,device=DEVICE);y[:,:,22:40,24:42]=1
    pv=torch.ones_like(y);uv=torch.ones(8,1,64,64,device=DEVICE)
    trace=dict(P_sha256=tensor_sha(p),U_sha256=tensor_sha(u),noise_state_sha256=tensor_sha(pack['noise'].get_state()))
    return [p,y,pv,u,uv],trace

def grad_record(module):
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads,'NO_GRADIENTS'
    finite=torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
    norm=torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt().item()
    assert finite and math.isfinite(norm),'NONFINITE_GRADIENT'
    return norm

def train_step(pack,values,update,weight,commit=True):
    start=time.perf_counter();student=pack['student'];teacher=pack['teacher'];disc=pack['discriminator']
    p,y,pv,u,uv=values;student.train();teacher.eval();pack['optimizer'].zero_grad(set_to_none=True)
    if disc is not None:disc.train();pack['discriminator_optimizer'].zero_grad(set_to_none=True)
    gen=pack['noise']
    weak=(u+.01*torch.randn(u.shape,device=DEVICE,generator=gen))*uv
    scale=torch.empty((8,1,1,1),device=DEVICE).uniform_(.9,1.1,generator=gen)
    strong=(u*scale+.05*torch.randn(u.shape,device=DEVICE,generator=gen))*uv
    strong=strong.masked_fill(torch.rand((8,158,1,1),device=DEVICE,generator=gen)<.08,0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        lp=student(p)
        with torch.no_grad():lt=teacher(weak)
        lu=student(strong)
    sup,bce,dice=supervised(lp,y,pv,weight)
    pt=torch.sigmoid(lt.float());ps=torch.sigmoid(lu.float())
    mask=((pt>=.7)|(pt<=.3))&(uv>0)
    cons=(ps[mask]-pt[mask]).square().mean() if mask.any() else lu.float().sum()*0
    adv=p.new_zeros(())
    if disc is not None:
        with torch.autocast('cuda',dtype=torch.bfloat16):score=disc(torch.sigmoid(lp))
        adv=F.binary_cross_entropy_with_logits(score.float(),torch.ones_like(score,dtype=torch.float32))
    ramp=.2*math.exp(-5*(1-min((update-1)/180,1))**2)
    loss=sup+ramp*cons+.01*adv
    assert torch.isfinite(loss).item(),'NONFINITE_LOSS'
    loss.backward();gn=grad_record(student);dl=p.new_zeros(());dn=0.
    if disc is not None:
        pack['discriminator_optimizer'].zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):real=disc(y);fake=disc(torch.sigmoid(lp.detach()))
        dl=.5*(F.binary_cross_entropy_with_logits(real.float(),torch.ones_like(real,dtype=torch.float32))+F.binary_cross_entropy_with_logits(fake.float(),torch.zeros_like(fake,dtype=torch.float32)))
        assert torch.isfinite(dl).item(),'NONFINITE_D_LOSS'
        dl.backward();dn=grad_record(disc)
    if commit:
        pack['optimizer'].step()
        if disc is not None:pack['discriminator_optimizer'].step()
        pack['scheduler'].step()
        with torch.no_grad():
            for tp,sp in zip(teacher.parameters(),student.parameters()):tp.mul_(.99).add_(sp,alpha=.01)
            for tb,sb in zip(teacher.buffers(),student.buffers()):tb.copy_(sb)
    return dict(update=update,loss=float(loss.item()),supervised=float(sup.item()),BCE=float(bce.item()),Dice=float(dice.item()),consistency=float(cons.item()),consistency_weight=ramp,GAN_generator=float(adv.item()),GAN_discriminator=float(dl.item()),generator_gradient_l2=gn,discriminator_gradient_l2=dn,confidence_fraction=float(mask.float().mean().item()),learning_rate=pack['optimizer'].param_groups[0]['lr'],Student_committed=int(commit),EMA_committed=int(commit),GAN_committed=int(commit and disc is not None),seconds=time.perf_counter()-start)

def metric(labels,probs):
    assert len(labels)==len(probs) and np.isfinite(probs).all() and set(np.unique(labels))=={0,1}
    order=np.argsort(-probs,kind='stable');p=probs[order];y=labels[order]
    ends=np.r_[np.where(p[:-1]!=p[1:])[0],len(y)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends];fp=ends+1-tp;fn=int(y.sum())-tp
    f=2*tp/np.maximum(2*tp+fp+fn,1);index=int(np.argmax(f));threshold=float(p[ends[index]])
    pred=probs>=threshold;TP=int((pred&(labels==1)).sum());FP=int((pred&(labels==0)).sum());FN=int((~pred&(labels==1)).sum())
    return dict(AP=float(average_precision_score(labels,probs)),F1=2*TP/max(2*TP+FP+FN,1),IoU=TP/max(TP+FP+FN,1),Precision=TP/max(TP+FP,1),Recall=TP/max(TP+FN,1),threshold=threshold,TP=TP,FP=FP,FN=FN,n=len(labels),positives=int(labels.sum()),background=int((labels==0).sum()))

@torch.no_grad()
def evaluate(pack,data):
    teacher=pack['teacher'];teacher.eval();kk=[];pp=[];losses=[];w=data.valid.shape[1]
    rows=data.manifest['val_P']
    for start in range(0,len(rows),4):
        subset=rows[start:start+4];patches=[data.patch(t) for t in subset]
        vals=[torch.from_numpy(np.stack([t[i] for t in patches])).to(DEVICE) for i in range(3)]
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=teacher(vals[0])
        loss,*_=supervised(logits,vals[1],vals[2],data.manifest['positive_class_weight']);losses.append(float(loss.item()))
        probabilities=torch.sigmoid(logits.float()).cpu().numpy()[:,0]
        for t,p in zip(subset,probabilities):
            y,x=t['y'],t['x'];support=data.support[y:y+64,x:x+64];mask=support>0
            yy,xx=np.where(mask);kk.append((yy+y)*w+xx+x);pp.append(p[mask])
    keys,inv=np.unique(np.concatenate(kk),return_inverse=True)
    probabilities=np.bincount(inv,weights=np.concatenate(pp))/np.bincount(inv)
    assert np.array_equal(keys,data.expected['keys']),'EVALUATION_SUPPORT_CHANGED'
    labels=data.expected['labels'];m=metric(labels,probabilities);m['validation_loss']=float(np.mean(losses))
    return m,keys,labels,probabilities
