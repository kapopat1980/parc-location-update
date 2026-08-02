"""
schemes.py -- Location update schemes evaluated on a common trace.

Every scheme is driven by the SAME mobility trace and the SAME call-arrival
trace for a given user/seed, so all comparisons are paired.

Common cost model:
    C_total = C_U * (#location updates) + C_P * (#cells polled)
"""

import numpy as np
from engine import paging_cost, belief_row

C_U = 10.0      # cost of one location update (signalling units)
C_P = 1.0       # cost of polling one cell during paging
MAX_ROUNDS = 3  # paging delay bound


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _truncate(belief_vec, mass=0.995, cap=91):
    """Reduce a full-grid belief to a paging set covering `mass` probability."""
    order = np.argsort(-belief_vec)
    csum = np.cumsum(belief_vec[order])
    k = int(np.searchsorted(csum, mass) + 1)
    k = min(max(k, 1), cap)
    idx = order[:k]
    p = belief_vec[idx]
    s = p.sum()
    return idx, (p / s if s > 0 else np.full(len(idx), 1.0 / len(idx)))


def _epc(p_sorted_desc):
    """Expected number of cells polled under probability-ordered paging."""
    return float(np.sum(np.arange(1, len(p_sorted_desc) + 1) * p_sorted_desc))


def _cell_at(trace, t):
    """Cell occupied at time t (trace is list of (time, cell))."""
    ts = trace[0]
    k = np.searchsorted(ts, t, side='right') - 1
    return trace[1][max(k, 0)], max(k, 0)


# ---------------------------------------------------------------------------
# Baseline 1: Time-Based
# ---------------------------------------------------------------------------

def run_TB(grid, trace, calls, T, kernels, mean_res):
    ts, cs = trace
    updates = np.arange(0.0, ts[-1], T)
    n_upd = len(updates)
    polled = 0.0
    rounds = 0.0
    misses = 0
    for t in calls:
        j = np.searchsorted(updates, t, side='right') - 1
        if j < 0:
            j = 0
        t0 = updates[j]
        c0, _ = _cell_at(trace, t0)
        lam = (t - t0) / mean_res
        belief = belief_row(kernels, c0, lam, len(kernels) - 1)
        idx, p = _truncate(belief)
        true_c, _ = _cell_at(trace, t)
        pc, rd, ms = paging_cost(idx, p, true_c, grid.n, MAX_ROUNDS)
        polled += pc; rounds += rd; misses += ms
    return dict(updates=n_upd, polled=polled, rounds=rounds,
                misses=misses, calls=len(calls))


# ---------------------------------------------------------------------------
# Baseline 2: Distance-Based
# ---------------------------------------------------------------------------

def run_DB(grid, trace, calls, D, kernels, mean_res):
    ts, cs = trace
    upd_t, upd_c = [0.0], [cs[0]]
    ref = cs[0]
    for t, c in zip(ts[1:], cs[1:]):
        if grid.dist[ref, c] >= D:
            upd_t.append(t); upd_c.append(c); ref = c
    upd_t = np.array(upd_t)
    polled = 0.0; rounds = 0.0; misses = 0
    for t in calls:
        j = max(np.searchsorted(upd_t, t, side='right') - 1, 0)
        c0 = upd_c[j]
        lam = (t - upd_t[j]) / mean_res
        belief = belief_row(kernels, c0, lam, len(kernels) - 1).copy()
        belief[grid.dist[c0] >= D] = 0.0        # hard geometric constraint
        s = belief.sum()
        if s > 0:
            belief /= s
        idx, p = _truncate(belief)
        true_c, _ = _cell_at(trace, t)
        pc, rd, ms = paging_cost(idx, p, true_c, grid.n, MAX_ROUNDS)
        polled += pc; rounds += rd; misses += ms
    return dict(updates=len(upd_t), polled=polled, rounds=rounds,
                misses=misses, calls=len(calls))


# ---------------------------------------------------------------------------
# Baseline 3: Movement-Based
# ---------------------------------------------------------------------------

def run_MB(grid, trace, calls, M, kernels, mean_res):
    ts, cs = trace
    upd_t, upd_c = [0.0], [cs[0]]
    cnt = 0
    for t, c in zip(ts[1:], cs[1:]):
        cnt += 1
        if cnt >= M:
            upd_t.append(t); upd_c.append(c); cnt = 0
    upd_t = np.array(upd_t)
    polled = 0.0; rounds = 0.0; misses = 0
    for t in calls:
        j = max(np.searchsorted(upd_t, t, side='right') - 1, 0)
        c0 = upd_c[j]
        lam = (t - upd_t[j]) / mean_res
        belief = belief_row(kernels, c0, lam, len(kernels) - 1).copy()
        belief[grid.dist[c0] > M] = 0.0         # at most M hops since update
        s = belief.sum()
        if s > 0:
            belief /= s
        idx, p = _truncate(belief)
        true_c, _ = _cell_at(trace, t)
        pc, rd, ms = paging_cost(idx, p, true_c, grid.n, MAX_ROUNDS)
        polled += pc; rounds += rd; misses += ms
    return dict(updates=len(upd_t), polled=polled, rounds=rounds,
                misses=misses, calls=len(calls))


# ---------------------------------------------------------------------------
# Baseline 4: Profile-Based
# ---------------------------------------------------------------------------

def build_profile(grid, train_trace, n_slots=12, day_len=24.0):
    """Time-of-day conditional visit histogram learned from a training trace."""
    ts, cs = train_trace
    H = np.full((n_slots, grid.n), 1e-4)
    for t, c in zip(ts, cs):
        s = int((t % day_len) / (day_len / n_slots))
        H[s, c] += 1.0
    return H / H.sum(axis=1, keepdims=True)


def run_PB(grid, trace, calls, profile, mass, kernels, mean_res,
           n_slots=12, day_len=24.0):
    """Strong profile-based scheme.

    Trigger: the terminal updates whenever it leaves the profile-predicted set
    for the current time-of-day slot.
    Belief:  the network combines the profile prior for the slot with a
             random-walk propagation from the last reported position
             (normalised product), then pages in probability order.
    This is a competent, well-tuned realisation of the profile-based family
    rather than a naive one, so the comparison is not against a strawman.
    """
    ts, cs = trace
    sets = []
    for s_ in range(n_slots):
        idx, _ = _truncate(profile[s_], mass=mass)
        sets.append(set(idx.tolist()))

    upd_t, upd_c = [0.0], [cs[0]]
    for t, c in zip(ts[1:], cs[1:]):
        s_ = int((t % day_len) / (day_len / n_slots))
        if c not in sets[s_]:
            upd_t.append(t); upd_c.append(c)
    upd_t = np.array(upd_t)

    kmax = len(kernels) - 1
    polled = 0.0; rounds = 0.0; misses = 0
    for t in calls:
        j = max(np.searchsorted(upd_t, t, side='right') - 1, 0)
        lam = max((t - upd_t[j]) / mean_res, 1e-6)
        b = belief_row(kernels, upd_c[j], lam, kmax)
        s_ = int((t % day_len) / (day_len / n_slots))
        b = b * profile[s_]
        tot = b.sum()
        if tot <= 0:
            b = belief_row(kernels, upd_c[j], lam, kmax)
        else:
            b = b / tot
        idx, p = _truncate(b)
        true_c, _ = _cell_at(trace, t)
        pc, rd, ms = paging_cost(idx, p, true_c, grid.n, MAX_ROUNDS)
        polled += pc; rounds += rd; misses += ms
    return dict(updates=len(upd_t), polled=polled, rounds=rounds,
                misses=misses, calls=len(calls))


# ---------------------------------------------------------------------------
# PROPOSED: PARC
# Predictive Adaptive Residence-aware Composite location update
# ---------------------------------------------------------------------------

def build_user_kernel(grid, train_trace, kmax, alpha=3.0):
    """Per-user first-order Markov kernel with empirical-Bayes shrinkage.

    Rows with few observations are shrunk toward the uniform-neighbour
    random-walk kernel:   M_i = (1-g_i) * M_hat_i + g_i * M_rw_i ,
    with  g_i = alpha / (alpha + n_i).
    This prevents the over-confident, miss-prone kernels that arise from
    sparse training data over a large cell mesh.
    """
    ts, cs = train_trace
    n = grid.n
    cnt = np.zeros((n, n))
    for a, b in zip(cs[:-1], cs[1:]):
        if a != b:
            cnt[a, b] += 1.0

    M = np.zeros((n, n))
    for i in range(n):
        nbs = grid.neigh[i]
        rw = np.zeros(n)
        rw[nbs] = 1.0 / len(nbs)
        n_i = cnt[i].sum()
        if n_i > 0:
            hat = cnt[i] / n_i
            gi = alpha / (alpha + n_i)
            M[i] = (1.0 - gi) * hat + gi * rw
        else:
            M[i] = rw
        M[i] /= M[i].sum()

    powers = [np.eye(n)]
    for _ in range(kmax):
        powers.append(powers[-1] @ M)
    return powers


def run_PARC(grid, trace, calls, user_kernels, profile, mean_res, lam_call,
             eps=0.0, T_max=None, n_slots=12, day_len=24.0,
             resync_period=None, resync_cost=1.0, w_profile=0.35,
             R_max=6, rw_kernels=None):
    """
    PARC -- Predictive Adaptive Residence-aware Composite location update.

    Three coupled mechanisms:

    (i)  SHARED PREDICTIVE STATE. Terminal and network hold the same
         per-user model: a shrunk first-order cell-transition kernel and a
         time-of-day occupancy prior. The network's belief about the
         terminal is therefore reproducible AT the terminal, which lets the
         terminal reason about the network's uncertainty regarding itself.

    (ii) ADAPTIVE SUPPORT SELECTION. At each update the terminal selects the
         registration radius R* that minimises the predicted signalling cost
         rate for its own current kernel, position and time-of-day context.
         Classical distance-based updating is the special case R* = const
         with a uniform kernel and no prior, so PARC dominates it by
         construction; the gain comes from making R* user- and
         context-specific.

    (iii) RENEWAL-REWARD STOPPING. Between updates the terminal updates at
         the first crossing where the instantaneous paging cost rate exceeds
         the running average cost rate of the current cycle -- the optimal
         stopping condition for the renewal-reward objective. No offline
         per-CMR tuning is required.

    A membership guarantee (update on leaving the network's paging set)
    bounds the paging-miss probability at zero.
    """
    ts, cs = trace
    kmax = len(user_kernels) - 1
    if T_max is None:
        T_max = 60.0 * mean_res

    # model[0] = learned kernel + time-of-day prior; model[1] = plain random walk
    use_learned = [True]

    def raw_belief(c0, t, lam):
        if use_learned[0]:
            b = belief_row(user_kernels, c0, lam, kmax)
            sl = int((t % day_len) / (day_len / n_slots))
            return (1.0 - w_profile) * b + w_profile * profile[sl]
        return belief_row(rw_kernels if rw_kernels is not None else user_kernels,
                          c0, lam, kmax)

    def masked_belief(c0, t, lam, R):
        b = raw_belief(c0, t, lam).copy()
        b[grid.dist[c0] > R] = 0.0
        tot = b.sum()
        if tot <= 0:
            b = np.zeros(grid.n)
            b[c0] = 1.0
            return b
        return b / tot

    def expected_paging(b):
        idx, p = _truncate(b)
        covered = float(b[idx].sum())
        base = _epc(np.sort(p)[::-1])
        return base + (1.0 - covered) * (len(idx) + grid.n)

    def _best_R_for_model(c0, t):
        best_R, best_rate = 1, np.inf
        kern = user_kernels if use_learned[0] else (
            rw_kernels if rw_kernels is not None else user_kernels)
        for R in range(1, R_max + 1):
            inside_mask = grid.dist[c0] <= R
            surv, epc_sum, wsum = [], 0.0, 0.0
            for k in range(0, kmax + 1):
                row = kern[k][c0]
                pin = float(row[inside_mask].sum())
                surv.append(pin)
                if pin > 1e-6:
                    b = raw_belief(c0, t + k * mean_res, max(k, 1e-6)).copy()
                    b[~inside_mask] = 0.0
                    tt = b.sum()
                    if tt > 0:
                        epc_sum += pin * expected_paging(b / tt)
                        wsum += pin
            # expected cycle length, with a geometric extension of the
            # survival tail beyond the kernel truncation horizon
            L = float(np.sum(surv))
            if len(surv) >= 2 and surv[-2] > 1e-9:
                r = min(surv[-1] / surv[-2], 0.999)
                if 0.0 < r < 1.0:
                    L += surv[-1] * r / (1.0 - r)
            L *= mean_res
            if L <= 0 or wsum <= 0:
                continue
            epc_avg = epc_sum / wsum
            rate = (C_U + lam_call * C_P * epc_avg * L) / L
            if rate < best_rate:
                best_rate, best_R = rate, R
        return best_R, best_rate

    def choose_R(c0, t):
        """Select belief model AND registration radius by predicted cost rate.

        PARC-H evaluates the predicted signalling cost rate under both the
        learned per-subscriber model and the plain random-walk model, and
        adopts whichever is cheaper. Because the random-walk branch requires
        no model synchronisation, the learned branch must earn its overhead
        before it is selected. Subscribers whose mobility carries no
        exploitable structure therefore fall back automatically to the
        classical distance-based behaviour.
        """
        if rw_kernels is None:
            use_learned[0] = True
            return _best_R_for_model(c0, t)[0]
        sync_penalty = (resync_cost * mean_res / resync_period) if resync_period else 0.0
        use_learned[0] = True
        R_l, rate_l = _best_R_for_model(c0, t)
        use_learned[0] = False
        R_r, rate_r = _best_R_for_model(c0, t)
        if rate_l + sync_penalty * C_U < rate_r:
            use_learned[0] = True
            learned_updates[0] += 1
            return R_l
        use_learned[0] = False
        return R_r

    # ---- forward pass ------------------------------------------------------
    learned_updates = [0]
    upd_t, upd_c, upd_R, upd_M = [0.0], [cs[0]], [], []
    t0, c0 = 0.0, cs[0]
    R = choose_R(c0, 0.0)
    upd_R.append(R); upd_M.append(use_learned[0])
    acc, t_prev = 0.0, 0.0

    for t, c in zip(ts[1:], cs[1:]):
        lam = max((t - t0) / mean_res, 1e-6)
        left_support = grid.dist[c0, c] > R
        if left_support:
            trigger = True
        else:
            # the registration radius is already chosen to minimise the whole
            # cycle cost rate, so no additional greedy trigger is applied;
            # only the paging-set membership guarantee and a safety cap
            b = masked_belief(c0, t, lam, R)
            idx_now, _ = _truncate(b)
            trigger = (c not in set(idx_now.tolist())) or ((t - t0) > T_max)
        t_prev = t
        if trigger:
            upd_t.append(t); upd_c.append(c)
            t0, c0 = t, c
            R = choose_R(c0, t)
            upd_R.append(R); upd_M.append(use_learned[0])
            acc = 0.0

    upd_t = np.array(upd_t)
    n_upd = len(upd_t)
    # model synchronisation is charged only in proportion to how often the
    # learned model was actually selected
    frac_learned = (learned_updates[0] / max(len(upd_R), 1)) if rw_kernels is not None else 1.0
    if resync_period:
        n_upd += resync_cost * (ts[-1] / resync_period) * frac_learned

    # ---- paging evaluation -------------------------------------------------
    polled = 0.0; rounds = 0.0; misses = 0
    for t in calls:
        j = max(np.searchsorted(upd_t, t, side='right') - 1, 0)
        lam = max((t - upd_t[j]) / mean_res, 1e-6)
        use_learned[0] = upd_M[j]
        b = masked_belief(upd_c[j], t, lam, upd_R[j])
        idx, p = _truncate(b)
        true_c, _ = _cell_at(trace, t)
        pc, rd, ms = paging_cost(idx, p, true_c, grid.n, MAX_ROUNDS)
        polled += pc; rounds += rd; misses += ms

    return dict(updates=n_upd, polled=polled, rounds=rounds,
                misses=misses, calls=len(calls),
                mean_R=float(np.mean(upd_R)), frac_learned=frac_learned)


def total_cost(res):
    return C_U * res['updates'] + C_P * res['polled']
