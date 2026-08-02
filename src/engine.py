"""
engine.py -- Simulation engine for comparative evaluation of location update
strategies in mobile computing.

Implements:
  * Hexagonal cell topology (axial coordinates)
  * Three mobility models: Random Walk, Gauss-Markov, Activity/Profile-driven
  * Five location update schemes:
        TB   - Time-Based
        DB   - Distance-Based
        MB   - Movement-Based
        PB   - Profile-Based
        PARC - Predictive Adaptive Residence-aware Composite (proposed)

Cost model follows the standard location-management formulation:
        C_total = C_u * N_update + C_p * N_cells_polled
"""

import numpy as np
from collections import defaultdict

# ----------------------------------------------------------------------------
# Hexagonal topology (axial coordinates q, r)
# ----------------------------------------------------------------------------

HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


class HexGrid:
    """Hexagonal cell mesh of given ring radius, with wrap-free boundary."""

    def __init__(self, radius):
        self.radius = radius
        self.cells = []
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                self.cells.append((q, r))
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.n = len(self.cells)
        # neighbour index table
        self.neigh = [[] for _ in range(self.n)]
        for c, i in self.index.items():
            for d in HEX_DIRS:
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in self.index:
                    self.neigh[i].append(self.index[nb])
        # all-pairs hex distance
        self.dist = np.zeros((self.n, self.n), dtype=np.int16)
        for i, a in enumerate(self.cells):
            for j, b in enumerate(self.cells):
                self.dist[i, j] = self._hexdist(a, b)

    @staticmethod
    def _hexdist(a, b):
        aq, ar = a
        bq, br = b
        return (abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) // 2

    def disc(self, centre_idx, radius):
        """Indices of all cells within `radius` hops of centre."""
        return np.where(self.dist[centre_idx] <= radius)[0]


# ----------------------------------------------------------------------------
# Mobility models -- each produces a trace of (time, cell_index)
# ----------------------------------------------------------------------------

class Mobility:
    """Base: generates cell-crossing events with Gamma-distributed residence."""

    def __init__(self, grid, rng, mean_residence=1.0, shape=2.0):
        self.g = grid
        self.rng = rng
        self.mean_res = mean_residence
        self.shape = shape

    def residence(self):
        # Gamma with given shape, mean = mean_res
        return self.rng.gamma(self.shape, self.mean_res / self.shape)

    def trace(self, T_end, start=None):
        raise NotImplementedError


class RandomWalk(Mobility):
    """Memoryless walk: uniform choice among neighbours."""

    def trace(self, T_end, start=None):
        i = start if start is not None else self.g.index[(0, 0)]
        t = 0.0
        out = [(0.0, i)]
        while t < T_end:
            t += self.residence()
            nbs = self.g.neigh[i]
            i = nbs[self.rng.integers(len(nbs))]
            out.append((t, i))
        return out


class GaussMarkov(Mobility):
    """Directionally persistent walk (discrete Gauss-Markov analogue on hex).

    alpha controls memory: alpha->1 gives straight-line (vehicular) motion,
    alpha->0 degenerates to a random walk.
    """

    def __init__(self, grid, rng, mean_residence=1.0, shape=2.0, alpha=0.75):
        super().__init__(grid, rng, mean_residence, shape)
        self.alpha = alpha

    def trace(self, T_end, start=None):
        i = start if start is not None else self.g.index[(0, 0)]
        d = int(self.rng.integers(6))
        t = 0.0
        out = [(0.0, i)]
        while t < T_end:
            t += self.residence()
            if self.rng.random() > self.alpha:
                d = (d + int(self.rng.integers(-1, 2))) % 6
            cand = (self.g.cells[i][0] + HEX_DIRS[d][0],
                    self.g.cells[i][1] + HEX_DIRS[d][1])
            if cand in self.g.index:
                i = self.g.index[cand]
            else:                      # reflect at boundary
                d = (d + 3) % 6
                nbs = self.g.neigh[i]
                i = nbs[self.rng.integers(len(nbs))]
            out.append((t, i))
        return out


class ActivityMobility(Mobility):
    """Profile-driven daily-cycle model: home -> work -> home with dwell.

    Produces strongly time-of-day-correlated, low-entropy movement, which is
    the regime profile-based schemes are designed for.
    """

    def __init__(self, grid, rng, mean_residence=1.0, shape=2.0,
                 day_len=24.0, noise=0.15):
        super().__init__(grid, rng, mean_residence, shape)
        self.day_len = day_len
        self.noise = noise
        # anchor points
        self.home = int(rng.integers(grid.n))
        far = np.where(grid.dist[self.home] >= max(3, grid.radius // 2))[0]
        self.work = int(far[rng.integers(len(far))]) if len(far) else self.home
        self.path_hw = self._path(self.home, self.work)
        self.path_wh = self.path_hw[::-1]

    def _path(self, a, b):
        """Greedy shortest hop path on the hex mesh."""
        path = [a]
        cur = a
        while cur != b:
            nbs = self.g.neigh[cur]
            cur = min(nbs, key=lambda x: self.g.dist[x, b])
            path.append(cur)
        return path

    def trace(self, T_end, start=None):
        t = 0.0
        i = self.home
        out = [(0.0, i)]
        while t < T_end:
            tod = t % self.day_len
            # commute windows: 8-10h and 17-19h
            if 8.0 <= tod < 10.0:
                seq = self.path_hw
            elif 17.0 <= tod < 19.0:
                seq = self.path_wh
            else:
                seq = None

            if seq is None:
                # dwell at anchor, occasional local excursion
                anchor = self.work if 10.0 <= tod < 17.0 else self.home
                t += self.residence()
                if self.rng.random() < self.noise:
                    nbs = self.g.neigh[i]
                    i = nbs[self.rng.integers(len(nbs))]
                else:
                    i = anchor
                out.append((t, i))
            else:
                for c in seq:
                    t += self.residence() * 0.5
                    if self.rng.random() < self.noise:
                        nbs = self.g.neigh[c]
                        c = nbs[self.rng.integers(len(nbs))]
                    i = c
                    out.append((t, i))
                    if t >= T_end:
                        break
        return out


# ----------------------------------------------------------------------------
# Paging cost model (shared by all schemes -- ensures a fair comparison)
# ----------------------------------------------------------------------------

def paging_cost(belief_idx, belief_p, true_cell, n_total, max_rounds=3):
    """Sequential probability-ordered paging.

    Returns (cells_polled, rounds_used, miss_flag).
    Cells in the belief set are polled in descending probability order,
    partitioned into at most `max_rounds` groups. If the terminal is not
    found, a blanket page over the whole location area is issued.
    """
    if len(belief_idx) == 0:
        return n_total, max_rounds + 1, 1

    order = np.argsort(-belief_p)
    cells = belief_idx[order]
    hit = np.where(cells == true_cell)[0]

    if len(hit) == 0:                       # paging miss -> blanket page
        return len(cells) + n_total, max_rounds + 1, 1

    rank = hit[0] + 1
    # partition set into max_rounds groups; round index of the hit
    grp = int(np.ceil(rank / max(1, len(cells) / max_rounds)))
    return int(rank), min(grp, max_rounds), 0


# ----------------------------------------------------------------------------
# Random-walk propagation kernel (network belief without user profile)
# ----------------------------------------------------------------------------

def rw_kernel(grid, max_hops):
    """P[k][i, j] = prob. of being in j after exactly k uniform-neighbour hops."""
    n = grid.n
    M = np.zeros((n, n))
    for i in range(n):
        nbs = grid.neigh[i]
        M[i, nbs] = 1.0 / len(nbs)
    out = [np.eye(n)]
    for _ in range(max_hops):
        out.append(out[-1] @ M)
    return out


_LOGFACT = np.cumsum(np.concatenate(([0.0], np.log(np.arange(1, 200)))))


def poisson_weights(lam, kmax):
    ks = np.arange(0, kmax + 1)
    logw = -lam + ks * np.log(max(lam, 1e-12)) - _LOGFACT[:kmax + 1]
    w = np.exp(logw - logw.max())
    return w / w.sum()


def belief_row(kernels, c0, lam, kmax):
    """Row-wise mixture: belief over cells given start cell c0 and elapsed lam.

    Operates on single rows instead of full matrices -- this is the hot path.
    """
    w = poisson_weights(lam, kmax)
    acc = np.zeros(kernels[0].shape[1])
    for k in range(kmax + 1):
        if w[k] > 1e-6:
            acc += w[k] * kernels[k][c0]
    s = acc.sum()
    return acc / s if s > 0 else acc
