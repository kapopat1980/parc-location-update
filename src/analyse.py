"""analyse.py -- aggregate full_results.json into tables and figures."""

import json, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TB_GRID = [0.5, 1, 2, 3, 5, 8, 12, 20, 30]
DB_GRID = [1, 2, 3, 4, 5, 6]
MB_GRID = [1, 2, 3, 4, 6, 8, 12]
PB_GRID = [0.80, 0.90, 0.95, 0.99]
GRIDS = dict(TB=TB_GRID, DB=DB_GRID, MB=MB_GRID, PB=PB_GRID)
CMRS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SCHEMES = ['TB', 'DB', 'MB', 'PB']

rows = json.load(open('/home/claude/locsim/full_results.json'))
seeds = sorted({r['seed'] for r in rows})
kinds = sorted({r['kind'] for r in rows})
KIND_LABEL = {'RandomWalk': 'Random Walk', 'GaussMarkov': 'Gauss-Markov',
              'Activity': 'Activity (routine)'}


def sel(cmr=None, kind=None, seed=None):
    out = rows
    if cmr is not None:  out = [r for r in out if r['cmr'] == cmr]
    if kind is not None: out = [r for r in out if r['kind'] == kind]
    if seed is not None: out = [r for r in out if r['seed'] == seed]
    return out


# --------------------------------------------------------------------------
# FRAMING B: one globally optimal parameter per scheme (across everything)
# --------------------------------------------------------------------------
global_param = {}
for s in SCHEMES:
    tot = {pv: sum(r[f'{s}@{pv}'] for r in rows) for pv in GRIDS[s]}
    global_param[s] = min(tot, key=tot.get)

# --------------------------------------------------------------------------
# Aggregate tables
# --------------------------------------------------------------------------
print("=" * 78)
print("GLOBAL (deployable) PARAMETERS:", global_param)
print("=" * 78)

tableA, tableB = {}, {}
for cmr in CMRS:
    sub = sel(cmr=cmr)
    n = len(sub)
    row = {}
    for s in SCHEMES:
        costs = {pv: sum(r[f'{s}@{pv}'] for r in sub) for pv in GRIDS[s]}
        pstar = min(costs, key=costs.get)
        row[s] = (costs[pstar] / n, pstar)
    parc = sum(r['PARC'] for r in sub) / n
    tableA[cmr] = (row, parc)

    rowB = {s: sum(r[f'{s}@{global_param[s]}'] for r in sub) / n
            for s in SCHEMES}
    tableB[cmr] = (rowB, parc)

print("\nFRAMING A -- baselines re-tuned at every CMR (oracle comparator)")
print(f"{'CMR':>6} {'TB*':>10} {'DB*':>10} {'MB*':>10} {'PB*':>10} "
      f"{'PARC':>10} {'gain%':>8}")
for cmr in CMRS:
    row, parc = tableA[cmr]
    best = min(row[s][0] for s in SCHEMES)
    print(f"{cmr:>6} " + " ".join(f"{row[s][0]:>10.0f}" for s in SCHEMES) +
          f" {parc:>10.0f} {100*(best-parc)/best:>7.1f}%")

print("\nFRAMING B -- baselines with one globally tuned parameter (deployable)")
print(f"{'CMR':>6} {'TB':>10} {'DB':>10} {'MB':>10} {'PB':>10} "
      f"{'PARC':>10} {'gain%':>8}")
for cmr in CMRS:
    row, parc = tableB[cmr]
    best = min(row.values())
    print(f"{cmr:>6} " + " ".join(f"{row[s]:>10.0f}" for s in SCHEMES) +
          f" {parc:>10.0f} {100*(best-parc)/best:>7.1f}%")

# --------------------------------------------------------------------------
# Per-mobility-class breakdown (Framing A, the hostile one)
# --------------------------------------------------------------------------
print("\nPER-CLASS, FRAMING A (oracle baselines)")
print(f"{'class':>20} {'CMR':>6} {'best base':>11} {'PARC':>10} {'gain%':>8}")
per_class = {}
for k in kinds:
    per_class[k] = {}
    for cmr in CMRS:
        sub = sel(cmr=cmr, kind=k)
        if not sub: continue
        n = len(sub)
        bb = min(min(sum(r[f'{s}@{pv}'] for r in sub) for pv in GRIDS[s])
                 for s in SCHEMES) / n
        parc = sum(r['PARC'] for r in sub) / n
        per_class[k][cmr] = (bb, parc)
        print(f"{KIND_LABEL[k]:>20} {cmr:>6} {bb:>11.0f} {parc:>10.0f} "
              f"{100*(bb-parc)/bb:>7.1f}%")

# --------------------------------------------------------------------------
# Paired statistics (per-user paired differences, Framing B)
# --------------------------------------------------------------------------
print("\nPAIRED TESTS (Framing B, per-user, pooled over CMR)")
for s in SCHEMES:
    a = np.array([r[f'{s}@{global_param[s]}'] for r in rows], float)
    b = np.array([r['PARC'] for r in rows], float)
    d = (a - b) / a * 100.0
    t, p = stats.ttest_rel(a, b)
    w, pw = stats.wilcoxon(a, b)
    ci = stats.t.interval(0.95, len(d)-1, loc=d.mean(),
                          scale=stats.sem(d))
    dz = (a-b).mean() / (a-b).std(ddof=1)
    print(f"  PARC vs {s}: mean reduction {d.mean():6.2f}% "
          f"(95% CI {ci[0]:.2f}..{ci[1]:.2f}), t={t:.1f}, p={p:.2e}, "
          f"Wilcoxon p={pw:.2e}, Cohen dz={dz:.2f}, n={len(a)}")

# --------------------------------------------------------------------------
# Component metrics
# --------------------------------------------------------------------------
print("\nCOMPONENT METRICS (Framing B, aggregated)")
print(f"{'scheme':>8} {'upd/user':>10} {'polled/call':>12} {'miss rate':>11} "
      f"{'rounds/call':>12}")
for s in SCHEMES:
    pv = global_param[s]
    u = np.mean([r[f'{s}@{pv}#upd'] for r in rows])
    pol = np.sum([r[f'{s}@{pv}#pol'] for r in rows]) / np.sum([r['n_calls'] for r in rows])
    ms = np.sum([r[f'{s}@{pv}#miss'] for r in rows]) / np.sum([r['n_calls'] for r in rows])
    rd = np.sum([r[f'{s}@{pv}#rnd'] for r in rows]) / np.sum([r['n_calls'] for r in rows])
    print(f"{s:>8} {u:>10.1f} {pol:>12.2f} {ms:>11.4f} {rd:>12.2f}")
u = np.mean([r['PARC#upd'] for r in rows])
pol = np.sum([r['PARC#pol'] for r in rows]) / np.sum([r['n_calls'] for r in rows])
ms = np.sum([r['PARC#miss'] for r in rows]) / np.sum([r['n_calls'] for r in rows])
rd = np.sum([r['PARC#rnd'] for r in rows]) / np.sum([r['n_calls'] for r in rows])
print(f"{'PARC':>8} {u:>10.1f} {pol:>12.2f} {ms:>11.4f} {rd:>12.2f}")

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
plt.rcParams.update({'font.size': 9, 'figure.dpi': 160})
COL = dict(TB='#8c8c8c', DB='#1f77b4', MB='#2ca02c', PB='#d62728',
           PARC='#0f766e')
MK = dict(TB='o', DB='s', MB='^', PB='v', PARC='D')

# Fig 1: total cost vs CMR, both framings
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, tab, title in ((axes[0], tableA, 'Framing A: baselines re-tuned per CMR'),
                       (axes[1], tableB, 'Framing B: one global parameter')):
    for s in SCHEMES:
        y = [tab[c][0][s][0] if title.startswith('Framing A') else tab[c][0][s]
             for c in CMRS]
        ax.plot(CMRS, y, marker=MK[s], color=COL[s], label=s, lw=1.3, ms=4)
    ax.plot(CMRS, [tab[c][1] for c in CMRS], marker=MK['PARC'],
            color=COL['PARC'], label='PARC', lw=2.2, ms=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Call-to-Mobility Ratio'); ax.set_title(title, fontsize=9)
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7.5)
axes[0].set_ylabel('Mean signalling cost per user')
plt.tight_layout(); plt.savefig('/home/claude/locsim/fig1_cost.png')
plt.close()

# Fig 2: per-class gain, Framing A
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for k in kinds:
    y = [100*(per_class[k][c][0]-per_class[k][c][1])/per_class[k][c][0]
         for c in CMRS if c in per_class[k]]
    ax.plot([c for c in CMRS if c in per_class[k]], y, marker='o',
            lw=1.6, ms=4, label=KIND_LABEL[k])
ax.axhline(0, color='k', lw=.8, ls='--')
ax.set_xscale('log'); ax.set_xlabel('Call-to-Mobility Ratio')
ax.set_ylabel('PARC cost reduction vs best oracle baseline (%)')
ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('/home/claude/locsim/fig2_class.png')
plt.close()

# Fig 3: update vs paging decomposition, Framing B
fig, ax = plt.subplots(figsize=(6.2, 3.4))
names = SCHEMES + ['PARC']
upd, pag = [], []
n_users_tot = len(rows)
for s in names:
    if s == 'PARC':
        upd.append(10*np.mean([r['PARC#upd'] for r in rows]))
        pag.append(np.mean([r['PARC#pol'] for r in rows]))
    else:
        pv = global_param[s]
        upd.append(10*np.mean([r[f'{s}@{pv}#upd'] for r in rows]))
        pag.append(np.mean([r[f'{s}@{pv}#pol'] for r in rows]))
x = np.arange(len(names))
ax.bar(x, upd, .55, label='Location update cost', color='#0f766e')
ax.bar(x, pag, .55, bottom=upd, label='Paging cost', color='#c9a227')
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel('Mean cost per user'); ax.grid(alpha=.3, axis='y')
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('/home/claude/locsim/fig3_decomp.png')
plt.close()

json.dump(dict(global_param=global_param,
               tableA={str(k): (
                   {s: list(v[0][s]) for s in SCHEMES}, v[1]) for k, v in tableA.items()},
               tableB={str(k): (v[0], v[1]) for k, v in tableB.items()},
               per_class={k: {str(c): list(v) for c, v in d.items()}
                          for k, d in per_class.items()}),
          open('/home/claude/locsim/summary.json', 'w'), indent=1)
print("\nfigures + summary.json written")
