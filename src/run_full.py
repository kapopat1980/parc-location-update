"""
run_full.py -- Full comparative evaluation under two framings.

FRAMING A (oracle baselines):
    each classical scheme is re-tuned to its own optimum at EVERY operating
    point (CMR). This is a deliberately hostile comparator -- no deployed
    network re-optimises its update parameter per user per call-rate -- and
    it establishes a lower bound on the gain.

FRAMING B (deployable baselines):
    each classical scheme is given ONE globally optimal parameter, fitted
    across the whole population and all operating points, as in practice.
    PARC is unchanged between framings: it adapts online and is never tuned.

Reported per mobility class and in aggregate, over multiple seeds, with
paired statistics.
"""

import json, sys, time
import numpy as np
sys.path.insert(0, '/home/claude/locsim')

from engine import HexGrid, RandomWalk, GaussMarkov, ActivityMobility, rw_kernel
import schemes as S

GRID_R, KMAX, MEAN_RES = 6, 20, 1.0
SIM_T, TRAIN_T, DAY = 240.0, 960.0, 24.0
CMR_LIST = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TB_GRID = [0.5, 1, 2, 3, 5, 8, 12, 20, 30]
DB_GRID = [1, 2, 3, 4, 5, 6]
MB_GRID = [1, 2, 3, 4, 6, 8, 12]
PB_GRID = [0.80, 0.90, 0.95, 0.99]
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


def main(n_users=30, seed_list=(0,), cmrs=None, out=None):
    grid = HexGrid(GRID_R)
    kern = rw_kernel(grid, KMAX)
    rows = []
    t0 = time.time()

    for seed in seed_list:
        rng = np.random.default_rng(2000 + seed)
        users = []
        for _ in range(n_users):
            kind = KINDS[rng.choice(3, p=MIXW)]
            mob = make(grid, rng, kind)
            train = arr(mob.trace(TRAIN_T))
            test = arr(mob.trace(SIM_T, start=int(train[1][-1])))
            users.append((kind, test,
                          S.build_profile(grid, train, day_len=DAY),
                          S.build_user_kernel(grid, train, KMAX)))

        for cmr in (cmrs or CMR_LIST):
            lam = cmr / MEAN_RES
            calls = [np.sort(rng.uniform(0, SIM_T,
                     max(rng.poisson(lam * SIM_T), 1))) for _ in range(n_users)]

            # ---- evaluate every scheme at every parameter, per user -------
            # cube[scheme][param][user] = (cost, updates, polled, rounds, miss)
            cube = {}
            for name, gridv in (('TB', TB_GRID), ('DB', DB_GRID),
                                ('MB', MB_GRID), ('PB', PB_GRID)):
                cube[name] = {}
                for pv in gridv:
                    per_user = []
                    for u, (kind, tr, prof, ku) in enumerate(users):
                        if name == 'TB':
                            r = S.run_TB(grid, tr, calls[u], pv, kern, MEAN_RES)
                        elif name == 'DB':
                            r = S.run_DB(grid, tr, calls[u], pv, kern, MEAN_RES)
                        elif name == 'MB':
                            r = S.run_MB(grid, tr, calls[u], pv, kern, MEAN_RES)
                        else:
                            r = S.run_PB(grid, tr, calls[u], prof, pv, kern,
                                         MEAN_RES, day_len=DAY)
                        per_user.append((S.total_cost(r), r))
                    cube[name][pv] = per_user

            parc = []
            for u, (kind, tr, prof, ku) in enumerate(users):
                r = S.run_PARC(grid, tr, calls[u], ku, prof, MEAN_RES, lam,
                               resync_period=DAY, resync_cost=1.0)
                parc.append((S.total_cost(r), r))

            for u, (kind, tr, prof, ku) in enumerate(users):
                rec = dict(seed=seed, cmr=cmr, user=u, kind=kind,
                           n_calls=len(calls[u]), n_moves=len(tr[0]))
                for name in ('TB', 'DB', 'MB', 'PB'):
                    for pv, per_user in cube[name].items():
                        c, r = per_user[u]
                        rec[f'{name}@{pv}'] = c
                        rec[f'{name}@{pv}#upd'] = r['updates']
                        rec[f'{name}@{pv}#pol'] = r['polled']
                        rec[f'{name}@{pv}#miss'] = r['misses']
                        rec[f'{name}@{pv}#rnd'] = r['rounds']
                c, r = parc[u]
                rec['PARC'] = c
                rec['PARC#upd'] = r['updates']
                rec['PARC#pol'] = r['polled']
                rec['PARC#miss'] = r['misses']
                rec['PARC#rnd'] = r['rounds']
                rec['PARC#R'] = r['mean_R']
                rows.append(rec)

            print(f"seed={seed} cmr={cmr} done  ({time.time()-t0:.0f}s)",
                  flush=True)

    tag = f"{seed_list[0]}_{'lo' if (cmrs and cmrs[0]<1.0) else 'hi' if cmrs else 'all'}"
    out = out or f'/home/claude/locsim/part_{tag}.json'
    with open(out, 'w') as f:
        json.dump(rows, f)
    print("saved", len(rows), "user-records to", out)


if __name__ == '__main__':
    half = sys.argv[3] if len(sys.argv) > 3 else 'all'
    cm = {'lo': [0.1, 0.25, 0.5], 'hi': [1.0, 2.0, 4.0, 8.0]}.get(half)
    main(n_users=int(sys.argv[1]), seed_list=(int(sys.argv[2]),), cmrs=cm)
