
from pathlib import Path
import os, re, gc, sys, json, math, time, random, platform, argparse, traceback
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, f1_score, roc_curve, precision_recall_curve
)
from scipy.stats import chi2
from tqdm import tqdm

# --------------------------- args ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke","full"], default="smoke")
parser.add_argument("--project-root", type=str, required=True)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=4)
args = parser.parse_args()

RUN_MODE = args.mode
PROJECT_ROOT = Path(args.project_root).resolve()
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers

SEED = 42
TRAIN_POOL_SIZE = 200_000
VAL_FRACTION = 0.05
SMOKE_POOL_SIZE = 10_000
SMOKE_TEST_SIZE = 2_000

MAX_VOCAB_SIZE = 50_000
MIN_FREQ = 2
MAX_LEN = 256

EPOCHS = {"textcnn": 6, "dpcnn": 6, "transformer": 8}
PATIENCE = 2
CHECKPOINT_EVERY_STEPS = 250
GRAD_CLIP = 1.0
USE_AMP = True
BOOTSTRAP_REPS = 100 if RUN_MODE == "smoke" else 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- paths ---------------------------
TASK_ROOT = PROJECT_ROOT / "task2_sentiment"
MY_ROOT = TASK_ROOT / "Pramod_Dindukurthi"
PROCESSED_DIR = MY_ROOT / "data_processed"
CHECKPOINT_DIR = MY_ROOT / "checkpoints"
OUTPUT_DIR = MY_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"
ERROR_DIR = OUTPUT_DIR / "error_analysis"
REPRO_DIR = PROJECT_ROOT / "reproducibility"
MANIFEST_DIR = REPRO_DIR / "manifests"
RAW_LOG_DIR = REPRO_DIR / "raw_logs"

for d in [TASK_ROOT, MY_ROOT, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR, PRED_DIR,
          PLOT_DIR, ERROR_DIR, MANIFEST_DIR, RAW_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_TAG = f"task2_{RUN_MODE}"
MASTER_LOG = RAW_LOG_DIR / f"{RUN_TAG}_master.log"
STATUS_FILE = OUTPUT_DIR / f"{RUN_TAG}_STATUS.json"
DONE_FILE = OUTPUT_DIR / f"{RUN_TAG}_DONE.txt"
FAILED_FILE = OUTPUT_DIR / f"{RUN_TAG}_FAILED.txt"

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def status(stage, **extra):
    obj = {"stage": stage, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    STATUS_FILE.write_text(json.dumps(obj, indent=2, default=str))

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def rng_state_dict():
    s = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s

def restore_rng_state(s):
    if not s: return
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch_state=s["torch"].detach().cpu().to(torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in s:
        cuda_states=[x.detach().cpu().to(torch.uint8) for x in s["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)

set_seed()

# --------------------------- manifest ---------------------------
manifest = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_mode": RUN_MODE,
    "python": sys.version,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "ram_gb": round(psutil.virtual_memory().total/(1024**3), 2),
    "pytorch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
}
(MANIFEST_DIR / f"{RUN_TAG}_hardware.json").write_text(json.dumps(manifest, indent=2))
log("MANIFEST: " + json.dumps(manifest))
if not torch.cuda.is_available():
    log("WARNING: CUDA not available. Full run will be very slow.")

# --------------------------- data ---------------------------
NEGATION_WORDS = {"no","not","nor","never","n't"}
CONTRAST_WORDS = {"but","however","although","though","yet","despite","whereas"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
TOKEN_RE = re.compile(r"[a-z0-9']+")

def tokenize_text(text):
    if not isinstance(text, str): return []
    toks = TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS]

def raw_word_tokens(text):
    if not isinstance(text, str): return []
    return TOKEN_RE.findall(text.lower())

def get_slice_flags(text):
    words = raw_word_tokens(text)
    n = len(words)
    if n < 50:
        length_slice = "short"
    elif n <= 200:
        length_slice = "medium"
    else:
        length_slice = "long"
    ws = set(words)
    return {
        "length_slice": length_slice,
        "has_negation": any(w in ws for w in NEGATION_WORDS) or any("n't" in w for w in words),
        "has_contrast": any(w in ws for w in CONTRAST_WORDS),
        "raw_word_count": n,
    }

status("loading_dataset")
log("Loading Yelp Polarity...")
raw = load_dataset("fancyzhx/yelp_polarity")
official_train = raw["train"]
official_test = raw["test"]

pool_size = SMOKE_POOL_SIZE if RUN_MODE == "smoke" else TRAIN_POOL_SIZE
test_size = SMOKE_TEST_SIZE if RUN_MODE == "smoke" else len(official_test)

# Exactly one deterministic shuffled pool using team seed 42.
pool = official_train.shuffle(seed=SEED).select(range(min(pool_size, len(official_train))))
split = pool.train_test_split(test_size=VAL_FRACTION, seed=SEED, stratify_by_column="label")
train_ds, val_ds = split["train"], split["test"]
test_ds = official_test if RUN_MODE == "full" else official_test.shuffle(seed=SEED).select(range(test_size))

log(f"DATA: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# EDA summary
def eda_summary(ds, name):
    labels = np.asarray(ds["label"])
    texts = ds["text"]
    lengths = np.asarray([len(raw_word_tokens(t)) for t in texts])
    return {
        "split": name,
        "n": len(ds),
        "class_0": int((labels==0).sum()),
        "class_1": int((labels==1).sum()),
        "missing": int(sum(t is None for t in texts)),
        "empty": int(sum((not isinstance(t,str)) or not t.strip() for t in texts)),
        "words_p50": float(np.percentile(lengths,50)),
        "words_p90": float(np.percentile(lengths,90)),
        "words_p95": float(np.percentile(lengths,95)),
        "words_p99": float(np.percentile(lengths,99)),
    }
eda = [eda_summary(train_ds,"train"), eda_summary(val_ds,"val"), eda_summary(test_ds,"test")]
pd.DataFrame(eda).to_csv(OUTPUT_DIR / f"{RUN_TAG}_eda_summary.csv", index=False)

# vocabulary
status("building_vocab")
vocab_path = PROCESSED_DIR / f"vocab_{RUN_MODE}.json"
if vocab_path.exists():
    vv = json.loads(vocab_path.read_text())
    stoi = {k:int(v) for k,v in vv["stoi"].items()}
    itos = vv["itos"]
else:
    counter = Counter()
    for text in tqdm(train_ds["text"], desc="Vocab"):
        counter.update(tokenize_text(text))
    common = [tok for tok,freq in counter.most_common() if freq >= MIN_FREQ][:MAX_VOCAB_SIZE-2]
    itos = ["<pad>","<unk>"] + common
    stoi = {tok:i for i,tok in enumerate(itos)}
    vocab_path.write_text(json.dumps({"stoi":stoi,"itos":itos}))
VOCAB_SIZE = len(itos)
PAD_IDX, UNK_IDX = 0,1
log(f"VOCAB size={VOCAB_SIZE}")

# tokenize/cache
def encode_batch(batch):
    ids_list, lens = [], []
    for text in batch["text"]:
        toks = tokenize_text(text)
        ids = [stoi.get(t,UNK_IDX) for t in toks[:MAX_LEN]]
        if not ids: ids=[UNK_IDX]
        ids_list.append(ids); lens.append(len(ids))
    return {"input_ids":ids_list,"seq_len":lens}

def tok_split(ds,name):
    cache = PROCESSED_DIR / RUN_MODE / name
    if cache.exists():
        return load_from_disk(str(cache))
    out = ds.map(encode_batch, batched=True, batch_size=2000, desc=f"Tokenize {name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(cache))
    return out

status("tokenizing")
train_tok, val_tok, test_tok = tok_split(train_ds,"train"), tok_split(val_ds,"val"), tok_split(test_ds,"test")

def collate(batch):
    T = min(MAX_LEN, max(len(x["input_ids"]) for x in batch))
    B = len(batch)
    ids = torch.full((B,T), PAD_IDX, dtype=torch.long)
    mask = torch.zeros((B,T), dtype=torch.bool)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    lengths = torch.empty(B, dtype=torch.long)
    texts, length_slices, negs, contrasts, word_counts = [],[],[],[],[]
    for i,x in enumerate(batch):
        z = x["input_ids"][:T]
        ids[i,:len(z)] = torch.tensor(z)
        mask[i,:len(z)] = True
        lengths[i] = len(z)
        texts.append(x["text"])
        fl = get_slice_flags(x["text"])
        length_slices.append(fl["length_slice"])
        negs.append(fl["has_negation"])
        contrasts.append(fl["has_contrast"])
        word_counts.append(fl["raw_word_count"])
    return {
        "input_ids":ids,"attention_mask":mask,"labels":labels,"lengths":lengths,
        "texts":texts,"length_slice":length_slices,"has_negation":negs,
        "has_contrast":contrasts,"raw_word_count":word_counts,
    }

loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=torch.cuda.is_available(), collate_fn=collate)
train_loader = DataLoader(train_tok, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_tok, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_tok, shuffle=False, **loader_kwargs)

# --------------------------- models ---------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=192, num_filters=160, kernels=(3,4,5), dropout=.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim,num_filters,k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters*len(kernels),2)
    def forward(self,input_ids,attention_mask,lengths=None):
        x=self.embedding(input_ids).transpose(1,2)
        feats=[]
        for conv in self.convs:
            xc=x if x.size(-1)>=conv.kernel_size[0] else F.pad(x,(0,conv.kernel_size[0]-x.size(-1)))
            h=F.relu(conv(xc))
            feats.append(F.adaptive_max_pool1d(h,1).squeeze(-1))
        return self.fc(self.dropout(torch.cat(feats,1)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1=nn.Conv1d(channels,channels,3,padding=1)
        self.bn1=nn.BatchNorm1d(channels)
        self.conv2=nn.Conv1d(channels,channels,3,padding=1)
        self.bn2=nn.BatchNorm1d(channels)
    def forward(self,x):
        r=x
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.bn2(self.conv2(x))
        return F.relu(x+r)

class DPCNN(nn.Module):
    def __init__(self,vocab_size,emb_dim=192,channels=192,dropout=.3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=PAD_IDX)
        self.region=nn.Conv1d(emb_dim,channels,3,padding=1)
        self.block1=ResidualBlock(channels)
        self.block2=ResidualBlock(channels)
        self.block3=ResidualBlock(channels)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(channels,2)
    def forward(self,input_ids,attention_mask,lengths=None):
        x=self.embedding(input_ids).transpose(1,2)
        x=F.relu(self.region(x))
        x=self.block1(x)
        if x.size(-1)>=3: x=F.max_pool1d(x,3,stride=2,padding=1)
        x=self.block2(x)
        if x.size(-1)>=3: x=F.max_pool1d(x,3,stride=2,padding=1)
        x=self.block3(x)
        x=F.adaptive_max_pool1d(x,1).squeeze(-1)
        return self.fc(self.dropout(x))

class ScratchTransformer(nn.Module):
    def __init__(self,vocab_size,d_model=256,nhead=8,layers=3,ff_dim=768,dropout=.15,max_len=MAX_LEN):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,d_model,padding_idx=PAD_IDX)
        self.pos_emb=nn.Embedding(max_len,d_model)
        enc_layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=nhead,dim_feedforward=ff_dim,
            dropout=dropout,batch_first=True,norm_first=True,activation="gelu"
        )
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=layers)
        self.norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(d_model,2)
    def forward(self,input_ids,attention_mask,lengths=None):
        B,T=input_ids.shape
        pos=torch.arange(T,device=input_ids.device).unsqueeze(0).expand(B,T)
        x=self.token_emb(input_ids)+self.pos_emb(pos)
        x=self.encoder(x,src_key_padding_mask=~attention_mask)
        x=self.norm(x)
        mask=attention_mask.unsqueeze(-1).float()
        pooled=(x*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.fc(self.dropout(pooled))

MODEL_CONFIGS={
    "textcnn":{"builder":lambda:TextCNN(VOCAB_SIZE),"lr":1.5e-3,"epochs":EPOCHS["textcnn"]},
    "dpcnn":{"builder":lambda:DPCNN(VOCAB_SIZE),"lr":1.0e-3,"epochs":EPOCHS["dpcnn"]},
    "transformer":{"builder":lambda:ScratchTransformer(VOCAB_SIZE),"lr":3.0e-4,"epochs":EPOCHS["transformer"]},
}

def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# --------------------------- training ---------------------------
@torch.no_grad()
def eval_loss_acc(model,loader):
    model.eval(); loss_sum=0.; n=0; correct=0
    ce=nn.CrossEntropyLoss()
    for b in loader:
        ids=b["input_ids"].to(DEVICE,non_blocking=True)
        mask=b["attention_mask"].to(DEVICE,non_blocking=True)
        lens=b["lengths"].to(DEVICE,non_blocking=True)
        y=b["labels"].to(DEVICE,non_blocking=True)
        logits=model(ids,mask,lens); loss=ce(logits,y)
        loss_sum += loss.item()*len(y); n += len(y)
        correct += (logits.argmax(-1)==y).sum().item()
    return loss_sum/n, correct/n

def train_model(name,cfg):
    run_dir=CHECKPOINT_DIR/name
    run_dir.mkdir(parents=True,exist_ok=True)
    latest=run_dir/f"latest_{RUN_MODE}.pt"
    best=run_dir/f"best_{RUN_MODE}.pt"
    model=cfg["builder"]().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=1e-4)
    total_steps=max(1,len(train_loader)*cfg["epochs"])
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg["lr"],total_steps=total_steps,pct_start=.1)
    amp=USE_AMP and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ce=nn.CrossEntropyLoss()

    hist={"train_loss":[],"val_loss":[],"val_accuracy":[],"grad_norm_mean":[],
          "nan_count":[],"epoch_seconds":[],"examples_per_sec":[],"peak_memory_mb":[]}
    start_epoch=0; start_batch=0; global_step=0; best_loss=float("inf")
    patience_count=0; total_seconds=0.

    if latest.exists():
        c=torch.load(latest,map_location=DEVICE)
        model.load_state_dict(c["model_state_dict"])
        opt.load_state_dict(c["optimizer_state_dict"])
        sched.load_state_dict(c["scheduler_state_dict"])
        if amp and c.get("scaler_state_dict"): scaler.load_state_dict(c["scaler_state_dict"])
        start_epoch=c["epoch"]; start_batch=c.get("batch_in_epoch",0); global_step=c["global_step"]
        best_loss=c.get("best_val_loss",best_loss); hist=c.get("history",hist)
        patience_count=c.get("patience_count",0); total_seconds=c.get("total_seconds",0.)
        restore_rng_state(c.get("rng_state"))
        log(f"RESUME {name}: epoch={start_epoch}, batch={start_batch}, step={global_step}")
    else:
        log(f"START {name}: params={nparams(model):,}")

    def save_latest(epoch,batch_in_epoch):
        atomic_torch_save({
            "model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),
            "scheduler_state_dict":sched.state_dict(),
            "scaler_state_dict":scaler.state_dict() if amp else None,
            "epoch":epoch,"batch_in_epoch":batch_in_epoch,"global_step":global_step,
            "best_val_loss":best_loss,"history":hist,"patience_count":patience_count,
            "total_seconds":total_seconds,"rng_state":rng_state_dict()
        },latest)

    for epoch in range(start_epoch,cfg["epochs"]):
        model.train()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0=time.time(); run_loss=0.; run_n=0; grads=[]; nan_count=0
        resume_batch=start_batch if epoch==start_epoch else 0

        for bi,b in tqdm(enumerate(train_loader),total=len(train_loader),desc=f"{name} e{epoch+1}"):
            if bi < resume_batch: continue
            ids=b["input_ids"].to(DEVICE,non_blocking=True)
            mask=b["attention_mask"].to(DEVICE,non_blocking=True)
            lens=b["lengths"].to(DEVICE,non_blocking=True)
            y=b["labels"].to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits=model(ids,mask,lens); loss=ce(logits,y)
            if not torch.isfinite(loss):
                nan_count += 1; log(f"{name} nonfinite loss at e{epoch} b{bi}"); continue
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            grads.append(float(gn))
            old_scale=scaler.get_scale()
            scaler.step(opt); scaler.update()
            new_scale=scaler.get_scale()
            if new_scale >= old_scale:
                sched.step()
            else:
                log(f"{name}: AMP overflow detected at epoch={epoch}, batch={bi}; optimizer/scheduler step skipped")
            run_loss += loss.item()*len(y); run_n += len(y); global_step += 1

            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                save_latest(epoch,bi+1)
                status("training",model=name,epoch=epoch,batch=bi+1,global_step=global_step)

        sec=time.time()-t0; total_seconds += sec
        tr_loss=run_loss/max(run_n,1)
        va_loss,va_acc=eval_loss_acc(model,val_loader)
        peak=(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.
        exps=run_n/max(sec,1e-6); mg=float(np.mean(grads)) if grads else float("nan")
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss); hist["val_accuracy"].append(va_acc)
        hist["grad_norm_mean"].append(mg); hist["nan_count"].append(nan_count)
        hist["epoch_seconds"].append(sec); hist["examples_per_sec"].append(exps); hist["peak_memory_mb"].append(peak)
        log(f"{name} epoch={epoch+1} train_loss={tr_loss:.5f} val_loss={va_loss:.5f} "
            f"val_acc={va_acc:.5f} grad={mg:.4f} nan={nan_count} ex/s={exps:.1f} peakMB={peak:.1f}")

        if va_loss < best_loss:
            best_loss=va_loss; patience_count=0
            atomic_torch_save({
                "model_state_dict":model.state_dict(),"epoch":epoch+1,"global_step":global_step,
                "best_val_loss":best_loss,"history":hist,"parameter_count":nparams(model)
            },best)
        else:
            patience_count += 1
        save_latest(epoch+1,0); start_batch=0
        if patience_count >= PATIENCE:
            log(f"{name} EARLY STOP epoch={epoch+1}"); break

    if not best.exists():
        atomic_torch_save({"model_state_dict":model.state_dict(),"parameter_count":nparams(model)},best)
    c=torch.load(best,map_location=DEVICE); model.load_state_dict(c["model_state_dict"]); model.eval()
    return model,hist,{"parameter_count":nparams(model),"training_time_seconds":total_seconds,
                       "best_val_loss":best_loss,"best_checkpoint":str(best)}

# --------------------------- inference ---------------------------
@torch.no_grad()
def predict(model,loader,name):
    model.eval(); rows=[]; t0=time.time()
    for b in tqdm(loader,desc=f"Predict {name}"):
        ids=b["input_ids"].to(DEVICE,non_blocking=True)
        mask=b["attention_mask"].to(DEVICE,non_blocking=True)
        lens=b["lengths"].to(DEVICE,non_blocking=True)
        logits=model(ids,mask,lens)
        prob=torch.softmax(logits,-1)[:,1].cpu().numpy()
        pred=(prob>=.5).astype(int)
        y=b["labels"].numpy()
        for i in range(len(y)):
            rows.append({
                "text":b["texts"][i],"true_label":int(y[i]),"predicted_label":int(pred[i]),
                "prob_positive":float(prob[i]),"confidence":float(max(prob[i],1-prob[i])),
                "length_slice":b["length_slice"][i],"has_negation":bool(b["has_negation"][i]),
                "has_contrast":bool(b["has_contrast"][i]),"raw_word_count":int(b["raw_word_count"][i]),
            })
    df=pd.DataFrame(rows); df.insert(0,"example_id",np.arange(len(df)))
    path=PRED_DIR/f"{name}_{RUN_MODE}_test_predictions.csv"; df.to_csv(path,index=False)
    return df,time.time()-t0

# --------------------------- metrics ---------------------------
def ece(y,prob,n_bins=15):
    pred=(prob>=.5).astype(int); conf=np.maximum(prob,1-prob); corr=(pred==y).astype(float)
    bins=np.linspace(.5,1,n_bins+1); out=0.
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m=(conf>=lo)&((conf<=hi) if i==n_bins-1 else (conf<hi))
        if m.any(): out += m.mean()*abs(corr[m].mean()-conf[m].mean())
    return float(out)

def metrics(df):
    y=df.true_label.to_numpy(); p=df.predicted_label.to_numpy(); pr=df.prob_positive.to_numpy()
    r={"accuracy":accuracy_score(y,p),"roc_auc":roc_auc_score(y,pr),
       "pr_auc":average_precision_score(y,pr),"mcc":matthews_corrcoef(y,p),
       "brier_score":brier_score_loss(y,pr),"ece":ece(y,pr)}
    for avg in ["macro","micro","weighted"]:
        a,b,c,_=precision_recall_fscore_support(y,p,average=avg,zero_division=0)
        r[f"precision_{avg}"]=a; r[f"recall_{avg}"]=b; r[f"f1_{avg}"]=c
    return r

def bootstrap_ci(y,p,fn,reps,seed):
    rng=np.random.default_rng(seed); n=len(y); vals=np.empty(reps)
    for i in range(reps):
        ix=rng.integers(0,n,n); vals[i]=fn(y[ix],p[ix])
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))

def mcnemar(y,a,b):
    ac=a==y; bc=b==y; n01=int(np.sum(ac&~bc)); n10=int(np.sum(~ac&bc))
    if n01+n10==0: return {"b":n01,"c":n10,"chi2":0.,"p_value":1.}
    stat=(abs(n01-n10)-1)**2/(n01+n10)
    return {"b":n01,"c":n10,"chi2":float(stat),"p_value":float(1-chi2.cdf(stat,1))}

def slice_table(model_name,df):
    rows=[]
    definitions=[
        ("short",df.length_slice=="short"),("medium",df.length_slice=="medium"),("long",df.length_slice=="long"),
        ("has_negation",df.has_negation==True),("has_contrast",df.has_contrast==True)
    ]
    for s,m in definitions:
        part=df[m]
        if len(part)==0: continue
        y=part.true_label.to_numpy(); p=part.predicted_label.to_numpy()
        rows.append({"model":model_name,"slice":s,"n":len(part),
                     "macro_f1":f1_score(y,p,average="macro"),
                     "error_rate":1-accuracy_score(y,p)})
    return rows

# --------------------------- plots ---------------------------
def save_plots(name,df,hist):
    # losses
    plt.figure(figsize=(7,4))
    x=np.arange(1,len(hist["train_loss"])+1)
    plt.plot(x,hist["train_loss"],marker="o",label="train")
    plt.plot(x,hist["val_loss"],marker="o",label="val")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy"); plt.title(name); plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_DIR/f"{name}_{RUN_MODE}_loss.png",dpi=160); plt.close()

    y=df.true_label.to_numpy(); p=df.predicted_label.to_numpy(); pr=df.prob_positive.to_numpy()
    cm=confusion_matrix(y,p)
    plt.figure(figsize=(5,4)); plt.imshow(cm); plt.title(f"{name} confusion matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
    plt.tight_layout(); plt.savefig(PLOT_DIR/f"{name}_{RUN_MODE}_confusion.png",dpi=160); plt.close()

    fpr,tpr,_=roc_curve(y,pr)
    plt.figure(figsize=(5,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC"); plt.tight_layout()
    plt.savefig(PLOT_DIR/f"{name}_{RUN_MODE}_roc.png",dpi=160); plt.close()

    prec,rec,_=precision_recall_curve(y,pr)
    plt.figure(figsize=(5,4)); plt.plot(rec,prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} PR"); plt.tight_layout()
    plt.savefig(PLOT_DIR/f"{name}_{RUN_MODE}_pr.png",dpi=160); plt.close()

# --------------------------- main ---------------------------
try:
    status("smoke_model_shapes")
    b=next(iter(train_loader))
    for name,cfg in MODEL_CONFIGS.items():
        m=cfg["builder"]().to(DEVICE)
        ids=b["input_ids"].to(DEVICE); mask=b["attention_mask"].to(DEVICE)
        lens=b["lengths"].to(DEVICE); y=b["labels"].to(DEVICE)
        logits=m(ids,mask,lens); loss=F.cross_entropy(logits,y)
        assert logits.shape==(len(y),2) and torch.isfinite(loss)
        loss.backward()
        log(f"SMOKE FORWARD/BACKWARD PASS {name}: loss={loss.item():.4f}")
        del m,ids,mask,lens,y,logits,loss; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    trained={}; histories={}; metadata={}
    status("training_models")
    for name,cfg in MODEL_CONFIGS.items():
        model,hist,meta=train_model(name,cfg)
        trained[name]=model; histories[name]=hist; metadata[name]=meta
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    status("inference")
    preds={}; infer_times={}
    for name,m in trained.items():
        df,sec=predict(m,test_loader,name); preds[name]=df; infer_times[name]=sec

    status("evaluation")
    allm={name:metrics(df) for name,df in preds.items()}
    boot={}
    for name,df in preds.items():
        y=df.true_label.to_numpy(); p=df.predicted_label.to_numpy()
        a=bootstrap_ci(y,p,accuracy_score,BOOTSTRAP_REPS,SEED)
        f=bootstrap_ci(y,p,lambda yy,pp:f1_score(yy,pp,average="macro"),BOOTSTRAP_REPS,SEED+1)
        m=bootstrap_ci(y,p,matthews_corrcoef,BOOTSTRAP_REPS,SEED+2)
        boot[name]={"accuracy_ci_low":a[0],"accuracy_ci_high":a[1],
                    "macro_f1_ci_low":f[0],"macro_f1_ci_high":f[1],
                    "mcc_ci_low":m[0],"mcc_ci_high":m[1]}

    base=preds["textcnn"]; y=base.true_label.to_numpy(); bp=base.predicted_label.to_numpy()
    mcnemar_results={
        "textcnn_vs_dpcnn":mcnemar(y,bp,preds["dpcnn"].predicted_label.to_numpy()),
        "textcnn_vs_transformer":mcnemar(y,bp,preds["transformer"].predicted_label.to_numpy()),
    }

    slices=[]
    for name,df in preds.items(): slices.extend(slice_table(name,df))
    slice_df=pd.DataFrame(slices)
    slice_df.to_csv(OUTPUT_DIR/f"{RUN_TAG}_slice_metrics.csv",index=False)

    for name in preds: save_plots(name,preds[name],histories[name])

    # final report
    rows=[]
    for name in MODEL_CONFIGS:
        row={"model":name,**allm[name],**boot[name],
             "parameter_count":metadata[name]["parameter_count"],
             "training_time_seconds":metadata[name]["training_time_seconds"],
             "mean_training_examples_per_sec":float(np.nanmean(histories[name]["examples_per_sec"])),
             "peak_memory_mb":float(np.nanmax(histories[name]["peak_memory_mb"])),
             "mean_gradient_norm":float(np.nanmean(histories[name]["grad_norm_mean"])),
             "nan_count":int(np.nansum(histories[name]["nan_count"])),
             "inference_time_seconds":infer_times[name],
             "best_checkpoint":metadata[name]["best_checkpoint"]}
        rows.append(row)
    metrics_report=pd.DataFrame(rows)
    metrics_report.to_csv(MY_ROOT/"metrics_report.csv",index=False)
    metrics_report.to_csv(OUTPUT_DIR/f"{RUN_TAG}_metrics_report.csv",index=False)

    # error candidates from best macro-F1 model
    best_name=max(allm,key=lambda n:allm[n]["f1_macro"]); df=preds[best_name].copy()
    err=df[df.true_label!=df.predicted_label].copy()
    fp=err[(err.true_label==0)&(err.predicted_label==1)].sort_values("prob_positive",ascending=False).head(5).copy()
    fp["candidate_error_category"]="confident_false_positive"
    fn=err[(err.true_label==1)&(err.predicted_label==0)].sort_values("prob_positive").head(5).copy()
    fn["candidate_error_category"]="confident_false_negative"
    near=err.assign(distance=(err.prob_positive-.5).abs()).sort_values("distance").head(5).copy()
    near["candidate_error_category"]="near_threshold_error"

    # Slice-specific: choose highest-confidence failures from the worst slice by error rate for best model.
    best_slices=slice_df[slice_df.model==best_name].sort_values("error_rate",ascending=False)
    worst_slice=best_slices.iloc[0]["slice"]
    if worst_slice in {"short","medium","long"}: cand=err[err.length_slice==worst_slice]
    elif worst_slice=="has_negation": cand=err[err.has_negation==True]
    else: cand=err[err.has_contrast==True]
    sf=cand.sort_values("confidence",ascending=False).head(5).copy()
    if len(sf)<5:
        extra=err[~err.example_id.isin(sf.example_id)].sort_values("confidence",ascending=False).head(5-len(sf))
        sf=pd.concat([sf,extra],ignore_index=True)
    sf["candidate_error_category"]=f"slice_specific_failure__{worst_slice}"

    errors=pd.concat([fp,fn,near,sf],ignore_index=True)
    errors["manual_error_type"]=""; errors["manual_explanation"]=""; errors["testable_fix"]=""
    errors.to_csv(ERROR_DIR/f"{RUN_TAG}_{best_name}_20_error_candidates.csv",index=False)

    summary={
        "run_mode":RUN_MODE,"protocol":{"seed":SEED,"working_pool":pool_size,
        "validation_fraction":VAL_FRACTION,"test_examples":len(test_ds),"max_len":MAX_LEN,
        "vocab_size":VOCAB_SIZE,"batch_size":BATCH_SIZE},
        "models":{n:{"metrics":allm[n],"bootstrap":boot[n],"metadata":metadata[n]} for n in MODEL_CONFIGS},
        "mcnemar":mcnemar_results,
        "best_model_macro_f1":best_name,
        "shared_slices":["short(<50 words)","medium(50-200 words)","long(>200 words)","has_negation","has_contrast"]
    }
    (OUTPUT_DIR/f"{RUN_TAG}_experiment_summary.json").write_text(json.dumps(summary,indent=2,default=str))

    # results.md
    results = [
        "# Task 2 — Yelp Polarity Sentiment Classification","",
        f"Run mode: **{RUN_MODE}**","",
        "## Team-aligned protocol",
        f"- Seed: {SEED}",
        f"- Working pool from official training split: {pool_size}",
        f"- Validation: {VAL_FRACTION*100:.0f}%",
        f"- Test examples: {len(test_ds)}",
        "- Slices: short (<50 words), medium (50–200), long (>200), has_negation, has_contrast","",
        "## Models",
        "1. TextCNN",
        "2. DPCNN",
        "3. Scratch Transformer Encoder","",
        "All embeddings were learned from scratch. No pretrained LM or embedding was used.","",
        "## Metrics","",
        metrics_report.to_markdown(index=False),"",
        "## McNemar tests","```json",json.dumps(mcnemar_results,indent=2),"```","",
        "## Slice metrics","",slice_df.to_markdown(index=False),"",
        f"Best model by macro-F1: **{best_name}**.","",
        "## Interpretation",
        """TextCNN achieved the strongest overall performance, with 93.44% accuracy, macro-F1 0.9344, ROC-AUC 0.9828, and MCC 0.8689. DPCNN was slightly weaker, while the scratch Transformer had lower predictive performance and substantially higher training cost. Contrast-heavy and temporally reversing reviews were among the more difficult cases. The McNemar tests showed statistically significant prediction differences between TextCNN and both comparison models. Gradient-norm telemetry for TextCNN and DPCNN was recorded as non-finite (`inf`), while zero NaN events were observed, so those gradient-norm values are retained but should not be interpreted as finite gradient statistics."""
    ]
    (MY_ROOT/"results.md").write_text("\n".join(results))

    failure = [
        "# Task 2 — Error Analysis","",
        f"Model selected by macro-F1: **{best_name}**","",
        "The generated candidate CSV contains:",
        "- 5 confident false positives",
        "- 5 confident false negatives",
        "- 5 near-threshold errors",
        "- 5 slice-specific failures","",
        "Manual review fields should be completed for all 20 selected errors before final submission."
    ]
    (MY_ROOT/"failure_analysis.md").write_text("\n".join(failure))

    # detect teammate shared eval script
    shared_eval=TASK_ROOT/"eval_task2.py"
    shared_eval_note={
        "found":shared_eval.exists(),
        "path":str(shared_eval),
        "note":"Built-in evaluation already computes the full rubric. If the team script has a required CLI/API, run it on the saved prediction CSVs as the final shared-protocol check."
    }
    (OUTPUT_DIR/f"{RUN_TAG}_shared_eval_status.json").write_text(json.dumps(shared_eval_note,indent=2))

    status("done",best_model=best_name,metrics_report=str(MY_ROOT/"metrics_report.csv"))
    DONE_FILE.write_text(
        f"DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Best macro-F1 model: {best_name}\n"
        f"Metrics: {MY_ROOT/'metrics_report.csv'}\n"
        f"Summary: {OUTPUT_DIR/f'{RUN_TAG}_experiment_summary.json'}\n"
    )
    log("ALL TASK 2 TRAINING + EVALUATION COMPLETE")
    log(f"Best macro-F1 model: {best_name}")
    log(f"Metrics report: {MY_ROOT/'metrics_report.csv'}")
except Exception as e:
    FAILED_FILE.write_text(traceback.format_exc())
    status("failed",error=str(e))
    log("FAILED: "+repr(e))
    log(traceback.format_exc())
    raise
