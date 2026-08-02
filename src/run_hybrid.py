"""run_hybrid.py -- PARC-H under the full evaluation protocol.

Regenerates the IDENTICAL subscriber population and call traces used by
run_full.py (same seeds, same rng draw order) so that PARC-H results are
paired with the Section 6 results at the per-subscriber level.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/home/claude/locsim')
from engine import HexGrid, RandomWalk, GaussMarkov, ActivityMobility, rw_kernel
import schemes as S

GRID_R, KMAX, MEAN_RES = 6, 20, 1.0
SIM_T, TRAIN_T, DAY = 240.0, 960.0, 24.0
CMR_LIST = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
KINDS = ['RandomWalk', 'GaussMarkov', 'Activity']
MIXW = [0.34, 0.33, 0.33]


def arr(tr):
    return (np.array([x[0] for x in tr]), np.array([x[1] for x in tr]))


def make(grid, rng, kind):
    if kind == 'RandomWalk':
        return RandomWalk(grid, rng, MEAN_RES)
    if kind == 'GaussMarkov':
        return GaussMarkov(grid, rng, MEAN_RES, alpha=0.75)
    return ActivityMobility(grid, rng, MEAN_RES, day_len=DAY)


def main(n_users, seed, cmrs):
    grid = HexGrid(GRID_R)
    kern = rw_kernel(grid, KMAX)
    rng = np.random.default_rng(2000 + seed)     # identical to run_full.py
    users = []
    for _ in range(n_users):
        kind = KINDS[rng.choice(3, p=MIXW)]
        mob = make(grid, rng, kind)
        train = arr(mob.trace(TRAIN_T))
        test = arr(mob.trace(SIM_T, start=int(train[1][-1])))
        users.append((kind, test,
                      S.build_profile(grid, train, day_len=DAY),
                      S.build_user_kernel(grid, train, KMAX)))

    rows, t0 = [], time.time()
    for cmr in CMR_LIST:                      # loop over ALL to keep rng aligned
        lam = cmr / MEAN_RES
        calls = [np.sort(rng.uniform(0, SIM_T,
                 max(rng.poisson(lam * SIM_T), 1))) for _ in range(n_users)]
        if cmr not in cmrs:
            continue
        for u, (kind, tr, prof, ku) in enumerate(users):
            p = S.run_PARC(grid, tr, calls[u], ku, prof, MEAN_RES, lam,
                           resync_period=DAY, resync_cost=1.0)
            h = S.run_PARC(grid, tr, calls[u], ku, prof, MEAN_RES, lam,
                           resync_period=DAY, resync_cost=1.0, rw_kernels=kern)
            rows.append(dict(seed=seed, cmr=cmr, user=u, kind=kind,
                             n_calls=len(calls[u]),
                             PARC=S.total_cost(p), PARCH=S.total_cost(h),
                             PARCH_upd=h['updates'], PARCH_pol=h['polled'],
                             PARCH_miss=h['misses'], PARCH_rnd=h['rounds'],
                             PARCH_R=h['mean_R'],
                             PARCH_learned=h['frac_learned']))
        print(f"  seed={seed} cmr={cmr} done ({time.time()-t0:.0f}s)", flush=True)

    tag = f"{seed}_{'lo' if min(cmrs) < 1.0 else 'hi'}"
    with open(f'/home/claude/locsim/hyb_{tag}.json', 'w') as f:
        json.dump(rows, f)
    print("saved", len(rows), "records")


if __name__ == '__main__':
    half = sys.argv[3]
    cm = {'lo': [0.1, 0.25, 0.5], 'hi': [1.0, 2.0, 4.0, 8.0]}[half]
    main(int(sys.argv[1]), int(sys.argv[2]), cm)
