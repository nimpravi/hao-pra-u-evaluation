import csv,sys,json
from collections import defaultdict
import numpy as np
from scipy import stats
OBJ=dict(sla=0.2256,latency=0.1880,throughput=0.1692,acceptance=0.1316,
         utilization=0.1128,reliability=0.0658,isolation=0.0282,energy=0.0188,
         fragmentation=0.0390,overhead=0.0210)
BEN=["throughput","utilization","acceptance","reliability","isolation","energy_eff"]
COST=["sla_violation","latency","fragmentation","overhead"]
KEY=dict(sla_violation="sla",latency="latency",throughput="throughput",
         acceptance="acceptance",utilization="utilization",reliability="reliability",
         isolation="isolation",energy_eff="energy",fragmentation="fragmentation",overhead="overhead")
def wts(wE):
    o={k:v for k,v in OBJ.items() if k!="energy"}
    s=(1.0-wE)/sum(o.values()); w={k:v*s for k,v in o.items()}; w["energy"]=wE; return w
def comps(rows,wE):
    W=wts(wE); seeds=sorted({r["seed"] for r in rows}); ms=sorted({r["method"] for r in rows})
    out={m:[] for m in ms}
    for sd in seeds:
        sub={r["method"]:dict(r) for r in rows if r["seed"]==sd}
        acc={m:0.0 for m in ms}
        for met in BEN+COST:
            vals=np.array([float(sub[m][met]) for m in ms]); lo,hi=vals.min(),vals.max()
            n=(vals-lo)/(hi-lo+1e-12)
            if met in COST: n=1.0-n
            for m,v in zip(ms,n): acc[m]+=W[KEY[met]]*float(v)
        for m in ms: out[m].append(acc[m])
    return out
def dz(a,b):
    d=np.asarray(a)-np.asarray(b); return d.mean()/d.std(ddof=1)

rows=list(csv.DictReader(open(sys.argv[1])))
cells=defaultdict(list)
for r in rows: cells[(float(r["w_E"]),float(r["headroom"]))].append(r)
ppo_rank=[];tab_rank=[];h2h=0;h2h_p=[]
print("HAO-PPO vs HAO-PRA-U head-to-head (composite, paired)\n")
print(f"{'w_E':>6}{'h':>5}  {'winner':<22}{'PPOrk':>6}{'TABrk':>6}{'PPO-TAB dz':>11}{'p':>9}")
print("-"*68)
for cell in sorted(cells):
    wE,h=cell; c=comps(cells[cell],wE)
    mu={m:float(np.mean(v)) for m,v in c.items()}
    rank=sorted(mu,key=mu.get,reverse=True)
    pr=rank.index("HAO-PPO")+1; tr=rank.index("HAO-PRA-U")+1
    ppo_rank.append(pr); tab_rank.append(tr)
    a,b=np.array(c["HAO-PPO"]),np.array(c["HAO-PRA-U"])
    try: _,p=stats.wilcoxon(a,b)
    except ValueError: p=1.0
    h2h += (mu["HAO-PPO"]>mu["HAO-PRA-U"]); h2h_p.append(p)
    print(f"{wE:>6}{h:>5}  {rank[0]:<22}{pr:>6}{tr:>6}{dz(a,b):>11.2f}{p:>9.2e}")
print(f"\nHAO-PPO rank dist: "+", ".join(f"{k}:{ppo_rank.count(k)}" for k in sorted(set(ppo_rank))))
print(f"HAO-PRA-U rank dist: "+", ".join(f"{k}:{tab_rank.count(k)}" for k in sorted(set(tab_rank))))
print(f"PPO beats tabular on mean composite in {h2h}/20 regimes")
print(f"  of which significant (p<0.05): {sum(1 for p in h2h_p if p<0.05)}")
