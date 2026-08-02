# PARC — Adaptive Predictive Location Updating in Mobile Computing Networks

Simulation code and results for the paper
**"Exploiting Subscriber Mobility Regularity for Adaptive Location Updating in Mobile Computing Networks"**.

This repository contains everything needed to reproduce every number, table and figure in the
paper. No result reported in the paper was produced by any other means.

---

## What this is

Location management trades the signalling cost of **location updates** against the cost of
**paging**. Four dynamic update strategies dominate the literature — time-based, distance-based,
movement-based and profile-based — and each commits to a single scalar trigger with a threshold
that must be provisioned in advance.

This code implements all four, plus two proposed algorithms:

| Scheme | Description |
|---|---|
| `run_TB` | Time-based — update every *T* time units |
| `run_DB` | Distance-based — update at *D* cells from last report |
| `run_MB` | Movement-based — update after *M* cell crossings |
| `run_PB` | Profile-based — update on departure from time-of-day profile |
| `run_PARC` | **Proposed.** Shared predictive model, online radius selection, membership guarantee |
| `run_PARC(..., rw_kernels=…)` | **PARC-H.** Hybrid: selects between learned and random-walk models by predicted cost rate |

Cost model: `C_total = C_U · N_update + C_P · N_polled`, with `C_U = 10`, `C_P = 1`.

## Headline results

Over **560 paired per-subscriber observations** (4 seeds × 20 subscribers × 7 CMR points):

| Comparison | Cost reduction | Wilcoxon p | Cohen's d_z |
|---|---|---|---|
| PARC vs time-based | 53.84% | 2.0 × 10⁻⁹³ | 1.12 |
| PARC vs movement-based | 22.55% | 1.5 × 10⁻⁴⁸ | 0.61 |
| PARC vs profile-based | 73.95% | 1.0 × 10⁻⁹² | 0.69 |
| PARC vs distance-based (global) | 5.21% | **0.32 — not significant** | 0.24 |
| **PARC-H vs PARC** | 3.54% | 4.2 × 10⁻⁴⁹ | 0.77 |
| **PARC-H vs distance-based (global)** | **8.70%** | **5.0 × 10⁻¹²** | 0.32 |

**Read the fourth row.** Plain PARC does *not* significantly outperform a correctly tuned
distance-based scheme. This is reported in the paper as a substantive negative result, consistent
with the classical optimality theory for distance-based tracking. The hybrid PARC-H resolves it
against the deployable comparator but not against an oracle re-tuned at every operating point.

Both PARC and PARC-H achieve a **zero paging-miss rate** across all 560 observations.

## Fairness protocol

Three provisions guard against an inflated result, and reviewers are invited to check them:

1. **Fully paired.** All schemes see identical mobility traces and identical call-arrival traces
   for a given subscriber and seed.
2. **Baselines tuned to their own optimum.** Every classical scheme is grid-searched at every
   operating point. The comparison is against the best configuration of each, not a default.
3. **Dual framing.** *Framing A* re-tunes baselines at every CMR (an oracle no network can
   realise). *Framing B* gives one globally fitted parameter, as parameters are actually
   provisioned. Both are reported.

PARC receives no reciprocal advantage: one fixed configuration across every operating point and
mobility class, never tuned per CMR.

---

## Reproducing the results

```bash
pip install -r requirements.txt
```

### Main results (Tables 3–7, Figures 2–4)

Runs are split by seed and CMR range to keep individual invocations short:

```bash
cd src
for s in 0 1 2 3; do
  python run_full.py 20 $s lo      # CMR 0.1, 0.25, 0.5
  python run_full.py 20 $s hi      # CMR 1, 2, 4, 8
done
python -c "
import json,glob
rows=[]
for f in sorted(glob.glob('part_*.json')): rows+=json.load(open(f))
json.dump(rows,open('full_results.json','w'))
print('merged',len(rows),'records')"
python analyse.py
```

Expect roughly 25–30 minutes total on a modern CPU. Output: aggregated tables printed to stdout,
plus `fig1_cost.png`, `fig2_class.png`, `fig3_decomp.png` and `summary.json`.

### Hybrid results (Tables 8–10)

`run_hybrid.py` regenerates the **identical** subscriber population and call traces by replaying
the same RNG draw order, so hybrid results are paired with the main results at per-subscriber level.

```bash
for s in 0 1 2 3; do
  python run_hybrid.py 20 $s lo
  python run_hybrid.py 20 $s hi
done
```

### Architecture figure

```bash
python arch.py        # writes fig0_arch.png
```

---

## Layout

```
src/
  engine.py        hex topology, mobility models, paging cost, belief propagation
  schemes.py       the six schemes and the shared cost model
  run_full.py      main experiment with per-CMR baseline tuning
  run_hybrid.py    PARC-H under the identical protocol
  analyse.py       aggregation, paired statistics, figures
  arch.py          architecture diagram
results/
  full_results.json    560 per-subscriber records, all schemes, all parameters
  hybrid_results.json  560 per-subscriber records, PARC and PARC-H
  summary.json         aggregated tables
figures/           all figures as published
```

## Simulation parameters

| Parameter | Value |
|---|---|
| Topology | Hexagonal mesh, axial coordinates, ring radius 6 (127 cells) |
| Cell residence time | Gamma, shape 2, mean normalised to 1 |
| Call arrivals | Poisson, CMR ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8} |
| Mobility models | Random Walk; Gauss-Markov (α = 0.75); Activity (24-slot commute, 15% noise) |
| Population mix | Approximately equal thirds |
| Training window | 960 time units per subscriber |
| Evaluation window | 240 time units, disjoint from training |
| Paging | Probability-ordered, ≤ 3 rounds, blanket page on miss |
| Seeds | 4 (RNG seeded at `2000 + seed`) |

## Known limitations

Stated in the paper and repeated here so they are not discovered by surprise:

- **Synthetic mobility.** No measured call detail records. The Activity model imposes a regularity
  real commuters only approximate, so every routine-mobility figure is an **upper bound**.
- **Single cost ratio.** All results at `C_U/C_P = 10`. Scheme ranking at other ratios is not established.
- **Radius selector miscalibration.** With the learned model deselected, PARC-H should reduce
  exactly to distance-based updating, but its selector converges to `R = 1` where the tuned
  optimum is `D = 3`. Not a kernel-horizon artefact (verified at 20, 40 and 60 steps). The defect
  is in the cost-rate objective, likely the expected-cycle-length term. **This is the most
  worthwhile open item in the codebase.**
- **Implicit registration not modelled.** Would reduce measured update cost for every scheme,
  disproportionately at high CMR.
- **Perfect model sharing assumed.** Divergence under loss or delayed resync would weaken the
  zero-miss guarantee.

## Citation

```bibtex
@article{PLACEHOLDER,
  title  = {Exploiting Subscriber Mobility Regularity for Adaptive Location
            Updating in Mobile Computing Networks},
  author = {[AUTHOR NAME]},
  year   = {2026},
  note   = {Code: https://github.com/[USERNAME]/parc-location-update}
}
```

## License

MIT — see `LICENSE`. Replace `[AUTHOR NAME]` before publishing.
