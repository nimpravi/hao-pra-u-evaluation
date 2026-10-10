import csv, sys
from collections import defaultdict
import numpy as np
from scipy import stats

# PARTIAL composite: 7 of 10 metrics. NOT the registered composite.
OBJ=dict(sla=0.2256,latency=0.1880,throughput=0.1692,acceptance=0.1316,
         utilization=0.1128,energy=0.0188,fragmentation=0.0390)
BEN=["throughput","utilization","acceptance","energy_eff"]
COST=["sla_violation","latency","fragmentation"]
KEY=dict(sla_violation="sla",latency="latency",throughput="throughput",
         acceptance="acceptance",utilization="utilization",
         energy_eff="energy",fragmentation="fragmentation")

def wts(wE):
    o={k:v for k,v in OBJ.items() if k!="energy"}
    s=(1.0-wE)/sum(o.values()); w={k:v*s for k,v in o.items()}; w["energy"]=wE
    return w

def comps(rows,wE):
    W=wts(wE); seeds=sorted({r["seed"] for r in rows}); ms=sorted({r["method"] for r in rows})
    out={m:[] for m in ms}
    for sd in seeds:
        sub={r["method"]:dict(r) for r in rows if r["seed"]==sd}
        acc={m:0.0 for m in ms}
        for met in BEN+COST:
            vals=np.array([float(sub[m][met]) for m in ms])
            lo,hi=vals.min(),vals.max(); n=(vals-lo)/(hi-lo+1e-12)
            if met in COST: n=1.0-n
            for m,v in zip(ms,n): acc[m]+=W[KEY[met]]*float(v)
        for m in ms: out[m].append(acc[m])
    return out

rows=list(csv.DictReader(open(sys.argv[1])))
cells=defaultdict(list)
for r in rows: cells[(float(r["w_E"]),float(r["headroom"]))].append(r)
nwin=defaultdict(int); dec=0; pvals=[]
print(f"{'w_E':>6}{'h':>5}  {'winner':<24}{'HAOrank':>8}{'p_wilcox':>10}")
print("-"*56)
for cell in sorted(cells):
    wE,h=cell; c=comps(cells[cell],wE)
    mu={m:float(np.mean(v)) for m,v in c.items()}
    rank=sorted(mu,key=mu.get,reverse=True)
    w,ru=rank[0],rank[1]
    try: _,p=stats.wilcoxon(np.array(c[w]),np.array(c[ru]))
    except ValueError: p=float('nan')
    hr=rank.index("HAO-PRA-U")+1
    nwin[w]+=1; dec+= (p<0.05); pvals.append(p)
    print(f"{wE:>6}{h:>5}  {w:<24}{hr:>8}{p:>10.2e}")
print("\nWins by scheme:", dict(nwin))
print(f"Decisive (Wilcoxon p<0.05): {dec}/20")
