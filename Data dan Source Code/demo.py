"""Demo langsung: jalankan GA, ACO, dan SA pada satu instans, tampilkan rute
dan kurva konvergensi yang bergerak.

    python demo.py                      # default berlin52
    python demo.py --instance eil51 --budget 20000
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.tsp import load_tsp, distance_matrix, Counter, OPTIMUM
from src.ga import run_ga
from src.aco import run_aco
from src.baselines import run_sa

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="berlin52")
ap.add_argument("--budget", type=int, default=30000)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

xy = load_tsp(f"data/{args.instance}.tsp")
D = distance_matrix(xy)
opt = OPTIMUM[args.instance]
step = args.budget // 30                       # gambar ulang tiap 1/30 anggaran

ALGOS = {
    "GA-OX": lambda c, r: run_ga(c, r, crossover="ox"),
    "ACO":   lambda c, r: run_aco(c, r),
    "SA (baseline)": lambda c, r: run_sa(c, r),
}


class LiveCounter(Counter):
    """Counter yang juga menyimpan cuplikan rute terbaik untuk animasi."""
    def __init__(self, *a):
        super().__init__(*a)
        self.snapshots = []

    def record(self, length, tour):
        super().record(length, tour)
        if self.n_eval % step == 0:
            self.snapshots.append((self.n_eval, self.best, self.best_tour.copy()))


print("Menghitung... (sekitar 10-30 detik)")
results = {}
for name, fn in ALGOS.items():
    c = LiveCounter(D, args.budget, step)
    fn(c, np.random.default_rng(args.seed))
    results[name] = c
    print(f"  {name:14s} panjang terbaik = {c.best:.0f}  gap = {(c.best-opt)/opt*100:.2f}%")

plt.ion()
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
n_frames = min(len(c.snapshots) for c in results.values())
for f in range(n_frames):
    for ax, (name, c) in zip(axes[:3], results.items()):
        ev, best, tour = c.snapshots[f]
        t = list(tour) + [tour[0]]
        ax.clear()
        ax.plot(xy[t, 0], xy[t, 1], "-o", ms=3, lw=1)
        ax.set_title(f"{name}\neval {ev}  L={best:.0f}  gap {(best-opt)/opt*100:.1f}%", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[3]
    ax.clear()
    for name, c in results.items():
        e = [s[0] for s in c.snapshots[:f + 1]]
        g = [(s[1] - opt) / opt * 100 for s in c.snapshots[:f + 1]]
        ax.plot(e, g, label=name)
    ax.set_xlim(0, args.budget); ax.set_yscale("log")
    ax.set_xlabel("evaluasi"); ax.set_ylabel("gap (%)"); ax.legend(fontsize=8)
    ax.set_title(f"Konvergensi ({args.instance}, optimum {opt})", fontsize=9)
    fig.tight_layout()
    plt.pause(0.25)

plt.ioff()
plt.show()
