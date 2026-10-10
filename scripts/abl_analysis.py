import csv,sys
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
rows=list(csv.DictReader(open(sys.argv[1])))
cfgs=["FULL","no_reserve","no_risk","no_negotiate","no_edge"]
# composite per (cell,seed) min-max across CONFIGS
cells=defaultdict(list)
for r in rows: cells[(float(r["w_E"]),float(r["headroom"]))].append(r)
comp={c:[] for c in cfgs}
for cell in sorted(cells):
    wE,h=cell; W=wts(wE); sub=cells[cell]
    for sd in sorted({r["seed"] for r in sub}):
        d={r["config"]:r for r in sub if r["seed"]==sd}
        acc={c:0.0 for c in cfgs}
        for met in BEN+COST:
            vals=np.array([float(d[c][met]) for c in cfgs]); lo,hi=vals.min(),vals.max()
            n=(vals-lo)/(hi-lo+1e-12)
            if met in COST: n=1.0-n
            for c,v in zip(cfgs,n): acc[c]+=W[KEY[met]]*float(v)
        for c in cfgs: comp[c].append(acc[c])
def dz(a,b):
    d=np.asarray(a)-np.asarray(b); s=d.std(ddof=1)
    return d.mean()/s if s>1e-15 else 0.0
print("LEAVE-ONE-OUT ABLATION (n=200 paired cells)\n")
print(f"{'removed':<14}{'sla':>9}{'d_sla':>10}{'composite':>11}{'d_comp':>9}{'p_wilcox':>10}{'d_z':>7}")
print("-"*70)
full_sla=np.mean([float(r['sla_violation']) for r in rows if r['config']=='FULL'])
fc=np.array(comp["FULL"])
for c in cfgs:
    s=np.mean([float(r['sla_violation']) for r in rows if r['config']==c])
    cc=np.array(comp[c])
    if c=="FULL":
        print(f"{c:<14}{s:>9.5f}{'':>10}{cc.mean():>11.4f}{'':>9}{'':>10}{'':>7}")
        continue
    try: _,p=stats.wilcoxon(fc,cc)
    except ValueError: p=1.0
    print(f"{c:<14}{s:>9.5f}{s-full_sla:>+10.5f}{cc.mean():>11.4f}{cc.mean()-fc.mean():>+9.4f}{p:>10.2e}{dz(fc,cc):>7.2f}")
# per-headroom breakdown for reserve/edge
print("\nBy capacity headroom (mean SLA violation):")
print(f"{'h':>5}  "+"".join(f"{c:>13}" for c in cfgs))
for h in sorted({float(r['headroom']) for r in rows}):
    line=f"{h:>5}  "
    for c in cfgs:
        v=np.mean([float(r['sla_violation']) for r in rows if r['config']==c and float(r['headroom'])==h])
        line+=f"{v:>13.5f}"
    print(line)
