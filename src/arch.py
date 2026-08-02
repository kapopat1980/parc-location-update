"""arch.py -- PARC system architecture figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

TEAL, GOLD, INK, GREY = '#0F766E', '#C9A227', '#1A1A1A', '#6B7280'
BG_T, BG_N = '#E6F2F1', '#FBF4E0'

fig, ax = plt.subplots(figsize=(9.4, 5.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')

def box(x, y, w, h, fc, ec, lw=1.4, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

def txt(x, y, s, size=9, w='normal', c=INK, ha='center', style='normal'):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c,
            ha=ha, va='center', zorder=4, style=style)

def arrow(x1, y1, x2, y2, c, style='-|>', lw=1.6, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=13, linewidth=lw, color=c, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=3))

# ---------------- containers ----------------
box(2, 12, 40, 44, BG_T, TEAL, 1.8)
box(58, 12, 40, 44, BG_N, GOLD, 1.8)
txt(22, 53, 'MOBILE TERMINAL', 11, 'bold', TEAL)
txt(78, 53, 'NETWORK', 11, 'bold', '#8A6D14')

# ---------------- shared model ----------------
box(6, 41, 32, 9, 'white', TEAL, 1.2)
txt(22, 47.3, 'Shared predictive model', 8.6, 'bold', TEAL)
txt(22, 44.4, 'shrunk transition kernel  M', 7.8)
txt(22, 42.2, 'time-of-day prior  H', 7.8)

box(62, 41, 32, 9, 'white', GOLD, 1.2)
txt(78, 47.3, 'Shared predictive model', 8.6, 'bold', '#8A6D14')
txt(78, 44.4, 'identical copy of  M, H', 7.8)
txt(78, 42.2, 'last report  (c₀, t₀, R)', 7.8)

# sync link
arrow(38.5, 45.5, 61.5, 45.5, GREY, '<|-|>', 1.4, (0, (4, 2)))
txt(50, 47.6, 'model resync', 7.4, 'bold', GREY)
txt(50, 43.4, 'once per profile\nperiod (costed)', 6.6, 'normal', GREY)

# ---------------- terminal mechanisms ----------------
box(6, 30.5, 32, 8, 'white', TEAL, 1.0)
txt(22, 36.4, '① Adaptive radius selection', 8.2, 'bold', TEAL)
txt(22, 33.0, 'R* = arg min  [C_U + λC_P·EPC(R)·L(R)] / L(R)', 7.2, 'normal', INK)

box(6, 20.5, 32, 8, 'white', TEAL, 1.0)
txt(22, 26.4, '② Membership guarantee', 8.2, 'bold', TEAL)
txt(22, 23.6, 'reproduce network belief; update if', 7.2)
txt(22, 21.7, 'own cell would leave the paging set', 7.2)

box(6, 14, 32, 5.2, 'white', TEAL, 1.0)
txt(22, 16.6, '③ Trigger: leave disc(c₀,R) · not in set · T_max', 7.2, 'bold')

# ---------------- network mechanisms ----------------
box(62, 30.5, 32, 8, 'white', GOLD, 1.0)
txt(78, 36.4, 'Belief reconstruction', 8.2, 'bold', '#8A6D14')
txt(78, 33.0, 'propagate M from (c₀,t₀), blend H,', 7.2)
txt(78, 31.6, 'mask to disc(c₀, R)', 7.2)

box(62, 20.5, 32, 8, 'white', GOLD, 1.0)
txt(78, 26.4, 'Probability-ordered paging', 8.2, 'bold', '#8A6D14')
txt(78, 23.6, 'poll cells in descending belief order,', 7.2)
txt(78, 21.7, '≤ 3 rounds · zero miss by construction', 7.2)

box(62, 14, 32, 5.2, 'white', GOLD, 1.0)
txt(78, 16.6, 'Incoming call arrives → page', 7.2, 'bold')

# ---------------- signalling arrows ----------------
arrow(38.5, 17.5, 61.5, 17.5, TEAL, '-|>', 2.0)
txt(50, 19.4, 'LOCATION UPDATE  (c, id)', 7.6, 'bold', TEAL)
txt(50, 15.6, 'uplink · cost C_U', 6.8, 'normal', GREY)

arrow(61.5, 24.5, 38.5, 24.5, '#8A6D14', '-|>', 2.0)
txt(50, 26.4, 'PAGING', 7.6, 'bold', '#8A6D14')
txt(50, 22.7, 'downlink · cost C_P per cell', 6.8, 'normal', GREY)

# internal flow arrows
arrow(22, 41, 22, 38.5, TEAL, '-|>', 1.2)
arrow(22, 30.5, 22, 28.5, TEAL, '-|>', 1.2)
arrow(22, 20.5, 22, 19.2, TEAL, '-|>', 1.2)
arrow(78, 41, 78, 38.5, GOLD, '-|>', 1.2)
arrow(78, 30.5, 78, 28.5, GOLD, '-|>', 1.2)

# ---------------- key property banner ----------------
box(2, 2, 96, 7.5, '#F7F7F5', GREY, 1.0)
txt(50, 7.6, 'KEY PROPERTY', 8.2, 'bold', GREY)
txt(50, 4.6, 'Because terminal and network hold the same model, the terminal can evaluate the network\'s belief about its own position.\n'
             'The update decision becomes an economic one — update when predicted uncertainty becomes more expensive than an update.',
    7.6, 'normal', INK)

plt.tight_layout()
plt.savefig('/home/claude/locsim/fig0_arch.png', dpi=190, bbox_inches='tight',
            facecolor='white')
print("saved")
