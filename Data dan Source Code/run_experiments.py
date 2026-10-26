"""Jalankan semua eksperimen sesuai config.json dan simpan hasil mentah ke results/.

    python run_experiments.py

Run-run dijalankan paralel di semua core CPU supaya cepat (tiap run tetap
punya seed sendiri, jadi hasilnya sama persis berapa pun jumlah core-nya).
"""
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.tsp import load_tsp, distance_matrix, Counter, OPTIMUM
from src.ga import run_ga
from src.aco import run_aco
from src.baselines import run_hill_climbing, run_sa

RUNNERS = {"ga": run_ga, "aco": run_aco, "hc": run_hill_climbing, "sa": run_sa}


def run_one(D, params, seed, budget, log_every):
    """Jalankan satu algoritma satu kali; kembalikan Counter (berisi hasil) dan waktu."""
    params = dict(params)
    algo = RUNNERS[params.pop("type")]
    counter = Counter(D, budget, log_every)
    t0 = time.time()
    algo(counter, np.random.default_rng(seed), **params)
    return counter, time.time() - t0


def job(task):
    """Satu pekerjaan untuk proses paralel."""
    inst, name, params, seed, budget, log_every, tag = task
    D = distance_matrix(load_tsp(f"data/{inst}.tsp"))
    c, sec = run_one(D, params, seed, budget, log_every)
    assert sorted(c.best_tour) == list(range(len(D)))       # cek solusi valid
    gap = (c.best - OPTIMUM[inst]) / OPTIMUM[inst] * 100
    print(f"{tag:5s} {inst:9s} {name:8s} seed={seed}  L={c.best:8.0f}  gap={gap:6.2f}%  ({sec:.1f}s)", flush=True)
    return dict(instance=inst, algorithm=name, seed=seed, best_length=c.best,
                optimum=OPTIMUM[inst], gap_percent=round(gap, 3), time_sec=round(sec, 2),
                tour=" ".join(map(str, c.best_tour)), history=c.history, tag=tag, params=params)


def main():
    cfg = json.load(open("config.json"))
    B, log_every = cfg["budget_evaluations"], cfg["log_every"]

    tasks = [(inst, name, p, s, B, log_every, "main")
             for inst in cfg["instances"] for name, p in cfg["algorithms"].items()
             for s in cfg["seeds"]]
    sens = cfg["sensitivity"]
    for name in ["GA-OX", "ACO"]:
        p, values = sens[name]["param"], sens[name]["values"]
        for v in values:
            params = dict(cfg["algorithms"][name], **{p: v})
            tasks += [(sens["instance"], name, params, s, B, log_every, "sens") for s in cfg["seeds"]]

    # tugas ACO paling lama -> jalankan duluan agar core terpakai merata
    tasks.sort(key=lambda t: t[2]["type"] != "aco")
    t0 = time.time()
    with Pool() as pool:
        out = pool.map(job, tasks, chunksize=1)

    main_rows = [r for r in out if r["tag"] == "main"]
    curves = [dict(instance=r["instance"], algorithm=r["algorithm"], seed=r["seed"],
                   evaluations=(k + 1) * log_every, best=v)
              for r in main_rows for k, v in enumerate(r["history"])]
    cols = ["instance", "algorithm", "seed", "best_length", "optimum", "gap_percent", "time_sec", "tour"]
    pd.DataFrame(main_rows)[cols].to_csv("results/runs.csv", index=False)
    pd.DataFrame(curves).to_csv("results/convergence.csv", index=False)

    srows = []
    for r in out:
        if r["tag"] == "sens":
            p = sens[r["algorithm"]]["param"]
            srows.append(dict(instance=r["instance"], algorithm=r["algorithm"], param=p,
                              value=r["params"][p], seed=r["seed"], best_length=r["best_length"],
                              gap_percent=r["gap_percent"]))
    pd.DataFrame(srows).to_csv("results/sensitivity.csv", index=False)
    print(f"Selesai dalam {time.time() - t0:.0f} detik. Hasil di folder results/")


if __name__ == "__main__":
    main()
