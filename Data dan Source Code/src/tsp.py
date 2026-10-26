"""Utilitas TSP: membaca file TSPLIB, matriks jarak, dan panjang rute."""
import numpy as np

# Nilai optimum resmi dari TSPLIB (Reinelt, 1991)
OPTIMUM = {"eil51": 426, "berlin52": 7542, "st70": 675, "kroA100": 21282}


def load_tsp(path):
    """Baca koordinat kota dari file .tsp (format EUC_2D)."""
    coords, reading = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading = True
                continue
            if line.startswith("EOF") or not line:
                if reading and line.startswith("EOF"):
                    break
                continue
            if reading:
                _, x, y = line.split()[:3]
                coords.append((float(x), float(y)))
    return np.array(coords)


def distance_matrix(coords):
    """Jarak EUC_2D sesuai aturan TSPLIB: dibulatkan ke integer terdekat."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.floor(np.sqrt((diff ** 2).sum(-1)) + 0.5)


def tour_length(tour, D):
    """Panjang rute tertutup (kembali ke kota awal)."""
    return float(D[tour, np.roll(tour, -1)].sum())


class Counter:
    """Menghitung jumlah evaluasi fungsi cost, supaya perbandingan adil."""

    def __init__(self, D, budget, log_every):
        self.D, self.budget, self.log_every = D, budget, log_every
        self.n_eval, self.best, self.best_tour, self.history = 0, float("inf"), None, []

    def done(self):
        return self.n_eval >= self.budget

    def record(self, length, tour):
        """Catat satu evaluasi; simpan solusi terbaik dan kurva konvergensi."""
        self.n_eval += 1
        if length < self.best:
            self.best, self.best_tour = length, np.array(tour)
        if self.n_eval % self.log_every == 0:
            self.history.append(self.best)

    def evaluate(self, tour):
        L = tour_length(tour, self.D)
        self.record(L, tour)
        return L
