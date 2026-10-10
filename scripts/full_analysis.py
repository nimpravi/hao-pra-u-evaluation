import csv, sys
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
         isolation="isolation",energy_eff="energy",fragmentation="fragmentation",
         overhead="overhead")
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
            vals=np.array([float(sub[m][met]) for m in ms])
            lo,hi=vals.min(),vals.max(); n=(vals-lo)/(hi-lo+1e-12)
            if met in COST: n=1.0-n
            for m,v in zip(ms,n): acc[m]+=W[KEY[met]]*float(v)
        for m in ms: out[m].append(acc[m])
    return out
def dz(a,b):
    d=np.asarray(a)-np.asarray(b); return d.mean()/d.std(ddof=1)
def cliff(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return (sum(x>y for x in a for y in b)-sum(x<y for x in a for y in b))/(len(a)*len(b))

rows=list(csv.DictReader(open(sys.argv[1])))
cells=defaultdict(list)
for r in rows: cells[(float(r["w_E"]),float(r["headroom"]))].append(r)

recs=[]
for cell in sorted(cells):
    wE,h=cell; c=comps(cells[cell],wE)
    mu={m:float(np.mean(v)) for m,v in c.items()}
    rank=sorted(mu,key=mu.get,reverse=True); w,ru=rank[0],rank[1]
    a,b=np.array(c[w]),np.array(c[ru])
    try: _,p=stats.wilcoxon(a,b)
    except ValueError: p=1.0
    recs.append(dict(cell=cell,winner=w,runner=ru,p=p,
                     hao_rank=rank.index("HAO-PRA-U")+1,
                     dz=dz(a,b),cliff=cliff(a,b),
                     win=mu[w],hao=mu["HAO-PRA-U"]))

ps=np.array([r["p"] for r in recs]); n=len(ps); order=np.argsort(ps)
holm=np.empty(n); bh=np.empty(n)
for i,idx in enumerate(order): holm[idx]=min(1.0,(n-i)*ps[idx])
for i,idx in enumerate(order[::-1]):
    rank=n-i; bh[idx]=min(1.0, ps[idx]*n/rank)
bh=np.minimum.accumulate(bh[order[::-1]])[::-1] if False else bh

print("REGISTERED COMPOSITE — equal-budget nested validation\n")
print(f"{'w_E':>6}{'h':>5}  {'winner':<22}{'HAOrk':>6}{'p_wil':>9}{'Holm':>9}{'BH':>9}{'d_z':>7}{'cliff':>7}")
print("-"*82)
for r,hm,bb in zip(recs,holm,bh):
    wE,h=r["cell"]
    print(f"{wE:>6}{h:>5}  {r['winner']:<22}{r['hao_rank']:>6}{r['p']:>9.2e}"
          f"{hm:>9.2e}{bb:>9.2e}{r['dz']:>7.2f}{r['cliff']:>7.2f}")

wins=defaultdict(int); dec=defaultdict(int)
for r,hm in zip(recs,holm):
    wins[r["winner"]]+=1
    if r["p"]<0.05: dec[r["winner"]]+=1
print("\nNominal wins :",dict(wins))
print("Decisive (raw p<0.05):",dict(dec))
print(f"Surviving Holm-Bonferroni: {int((holm<0.05).sum())}/20")
print(f"Surviving Benjamini-Hochberg: {int((bh<0.05).sum())}/20")
hao=[r for r in recs if r["winner"]=="HAO-PRA-U"]
print(f"\nHAO-PRA-U wins: {len(hao)}/20")
for r in hao:
    print(f"  w_E={r['cell'][0]}, h={r['cell'][1]}: p={r['p']:.2e}, d_z={r['dz']:.2f}, composite {r['win']:.3f} vs runner")
ranks=[r["hao_rank"] for r in recs]
print(f"\nHAO-PRA-U rank distribution: "+", ".join(f"{k}:{ranks.count(k)}" for k in sorted(set(ranks))))
