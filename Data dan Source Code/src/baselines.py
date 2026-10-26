"""Baseline klasik: Hill Climbing 2-opt (random restart) dan Simulated Annealing.

Tetangga (neighbor) = langkah 2-opt: pilih dua posisi i<j lalu balik segmen tour[i..j].
Perubahan panjang dihitung cepat (delta) dari 4 sisi saja, tapi tetap dihitung
sebagai 1 evaluasi agar adil dibanding GA dan ACO.
"""
import math
import numpy as np


def two_opt_delta(tour, i, j, D):
    """Selisih panjang jika segmen tour[i..j] dibalik."""
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def random_move(n, rng):
    i, j = sorted(rng.choice(n, 2, replace=False))
    if i == 0 and j == n - 1:          # membalik seluruh rute = rute sama
        i = 1
    return i, j


def run_hill_climbing(counter, rng, patience=2000):
    """Terima langkah hanya jika lebih baik. Restart acak jika macet `patience` langkah."""
    D, n = counter.D, len(counter.D)
    while not counter.done():
        tour = rng.permutation(n)
        L = counter.evaluate(tour)
        stuck = 0
        while stuck < patience and not counter.done():
            i, j = random_move(n, rng)
            delta = two_opt_delta(tour, i, j, D)
            if delta < 0:
                tour[i:j + 1] = tour[i:j + 1][::-1]
                L += delta
                stuck = 0
            else:
                stuck += 1
            counter.record(L if delta < 0 else L + delta, tour)
    return counter


def run_sa(counter, rng, t_start=None, t_end=1e-3):
    """Simulated Annealing: langkah buruk diterima dengan peluang exp(-delta/T)."""
    D, n = counter.D, len(counter.D)
    tour = rng.permutation(n)
    L = counter.evaluate(tour)
    T0 = t_start if t_start else D.mean()          # suhu awal ~ rata-rata jarak
    cooling = (t_end / T0) ** (1.0 / counter.budget)   # pendinginan geometrik
    T = T0
    while not counter.done():
        i, j = random_move(n, rng)
        delta = two_opt_delta(tour, i, j, D)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            L += delta
        counter.record(L, tour)
        T *= cooling
    return counter
