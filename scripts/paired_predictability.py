import csv, glob, os
import numpy as np
from scipy import stats

BENEFIT=["throughput","utilization","acceptance","reliability","isolation","energy_eff"]
COST=["sla_violation","latency","fragmentation","overhead"]
KEYMAP=dict(sla_violation="sla",latency="latency",throughput="throughput",
            acceptance="acceptance",utilization="utilization",reliability="reliability",
            isolation="isolation",energy_eff="energy",fragmentation="fragmentation",
            overhead="overhead")
W=dict(sla=0.2256,latency=0.1880,throughput=0.1692,acceptance=0.1316,
       utilization=0.1128,reliability=0.0658,isolation=0.0282,energy=0.0188,
       fragmentation=0.0390,overhead=0.0210)

def composites(rows):
    seeds=sorted({r["seed"] for r in rows}); methods=sorted({r["method"] for r in rows})
    out={m:[] for m in methods}
    for sd in seeds:
        sub={r["method"]:r for r in rows if r["seed"]==sd}
        comp={m:0.0 for m in methods}
        for metric in BENEFIT+COST:
            vals=np.array([float(sub[m][metric]) for m in methods])
            lo,hi=vals.min(),vals.max()
            norm=(vals-lo)/(hi-lo+1e-12)
            if metric in COST: norm=1.0-norm
            for m,v in zip(methods,norm): comp[m]+=W[KEYMAP[metric]]*float(v)
        for m in methods: out[m].append(comp[m])
    return out

base="/home/claude/repo/hao-pra-u-evaluation/results/predictability"
print(f"{'p':>5} {'winner':<20} {'HAOcomp':>8} {'p_Welch':>9} {'p_paired':>9} {'p_wilcox':>9}  HAOwin")
print("-"*80)
for f in sorted(glob.glob(os.path.join(base,"results_per_seed__p*.csv"))):
    p=os.path.basename(f).split("__p")[1].replace(".csv","")
    rows=list(csv.DictReader(open(f)))
    comp=composites(rows)
    means={m:float(np.mean(v)) for m,v in comp.items()}
    ranked=sorted(means,key=means.get,reverse=True); w,ru=ranked[0],ranked[1]
    a=np.array(comp[w]); b=np.array(comp[ru])
    _,pw=stats.ttest_ind(a,b,equal_var=False); _,pp=stats.ttest_rel(a,b)
    try: _,pl=stats.wilcoxon(a,b)
    except ValueError: pl=float('nan')
    print(f"{p:>5} {w:<20} {means['HAO-PRA-U']:>8.3f} {pw:>9.2e} {pp:>9.2e} {pl:>9.2e}  {w=='HAO-PRA-U'}")
