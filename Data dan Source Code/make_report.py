"""Buat tabel, kurva konvergensi, uji statistik, dan grafik sensitivitas dari results/*.csv.

    python make_report.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums

runs = pd.read_csv("results/runs.csv")
conv = pd.read_csv("results/convergence.csv")
sens = pd.read_csv("results/sensitivity.csv")
ORDER = ["GA-OX", "GA-PMX", "ACO", "HC-2opt", "SA"]
instances = list(dict.fromkeys(runs.instance))

# 1) Tabel ringkasan: terbaik, rata-rata, std, terburuk, gap
summary = (runs.groupby(["instance", "algorithm"])
           .agg(best=("best_length", "min"), mean=("best_length", "mean"),
                std=("best_length", "std"), worst=("best_length", "max"),
                optimum=("optimum", "first"), mean_time_s=("time_sec", "mean"))
           .reset_index())
summary["gap_best_%"] = (summary.best - summary.optimum) / summary.optimum * 100
summary["gap_mean_%"] = (summary["mean"] - summary.optimum) / summary.optimum * 100
summary["algorithm"] = pd.Categorical(summary.algorithm, ORDER)
summary["instance"] = pd.Categorical(summary.instance, instances)
summary = summary.sort_values(["instance", "algorithm"]).round(2)
summary.to_csv("results/summary_table.csv", index=False)
print("=== Tabel hasil ===")
print(summary.to_string(index=False))

# 2) Uji Wilcoxon rank-sum antara dua algoritma terbaik (rata-rata) per instans
tests = []
for inst in instances:
    s = summary[summary.instance == inst].sort_values("mean")
    a, b = s.algorithm.iloc[0], s.algorithm.iloc[1]
    xa = runs[(runs.instance == inst) & (runs.algorithm == a)].best_length
    xb = runs[(runs.instance == inst) & (runs.algorithm == b)].best_length
    stat, p = ranksums(xa, xb)
    tests.append(dict(instance=inst, best_algo=a, second_algo=b,
                      mean_best=round(xa.mean(), 1), mean_second=round(xb.mean(), 1),
                      statistic=round(stat, 3), p_value=round(p, 4),
                      significant_5pct="ya" if p < 0.05 else "tidak"))
tests = pd.DataFrame(tests)
tests.to_csv("results/wilcoxon.csv", index=False)
print("\n=== Uji Wilcoxon rank-sum (dua algoritma terbaik) ===")
print(tests.to_string(index=False))

# 3) Kurva konvergensi (rata-rata 5 run)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, inst in zip(axes.flat, instances):
    c = conv[conv.instance == inst]
    opt = runs[runs.instance == inst].optimum.iloc[0]
    for alg in ORDER:
        m = c[c.algorithm == alg].groupby("evaluations").best.mean()
        ax.plot(m.index, (m - opt) / opt * 100, label=alg)
    ax.set_title(inst)
    ax.set_xlabel("Jumlah evaluasi")
    ax.set_ylabel("Gap rata-rata (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
axes.flat[0].legend()
fig.suptitle("Kurva konvergensi (rata-rata 5 run)")
fig.tight_layout()
fig.savefig("figures/convergence.png", dpi=150)

# 4) Boxplot gap per instans
fig, axes = plt.subplots(1, len(instances), figsize=(14, 4))
for ax, inst in zip(axes, instances):
    r = runs[runs.instance == inst]
    ax.boxplot([r[r.algorithm == a].gap_percent for a in ORDER])
    ax.set_xticks(range(1, len(ORDER) + 1), ORDER)
    ax.set_title(inst)
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("Gap (%)")
fig.tight_layout()
fig.savefig("figures/boxplot_gap.png", dpi=150)

# 5) Analisis sensitivitas
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, alg in zip(axes, ["GA-OX", "ACO"]):
    s = sens[sens.algorithm == alg].groupby("value").gap_percent.agg(["mean", "std"])
    ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=4)
    ax.set_xlabel(sens[sens.algorithm == alg].param.iloc[0])
    ax.set_ylabel("Gap rata-rata (%)")
    ax.set_title(f"Sensitivitas {alg} ({sens.instance.iloc[0]})")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/sensitivity.png", dpi=150)
sens_tab = sens.groupby(["algorithm", "param", "value"]).gap_percent.agg(["mean", "std", "min", "max"]).round(2)
sens_tab.to_csv("results/sensitivity_table.csv")
print("\n=== Sensitivitas parameter ===")
print(sens_tab.to_string())

# 6) Gambar rute terbaik tiap algoritma pada berlin52
from src.tsp import load_tsp
inst = "berlin52"
xy = load_tsp(f"data/{inst}.tsp")
fig, axes = plt.subplots(1, len(ORDER), figsize=(18, 3.8))
for ax, alg in zip(axes, ORDER):
    r = runs[(runs.instance == inst) & (runs.algorithm == alg)].sort_values("best_length").iloc[0]
    t = list(map(int, r.tour.split())) + [int(r.tour.split()[0])]
    ax.plot(xy[t, 0], xy[t, 1], "-o", ms=3, lw=1)
    ax.set_title(f"{alg}\nL={r.best_length:.0f} (gap {r.gap_percent:.1f}%)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("figures/best_tours_berlin52.png", dpi=150)
print("\nGambar disimpan di folder figures/")
