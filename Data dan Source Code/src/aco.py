import numpy as np


class AntColony:
    def __init__(self, counter, rng, n_ants=30, n_best=5, decay=0.9,
                 alpha=1.0, beta=2.0, rho=None):
        if rho is not None:         
            decay = 1.0 - rho #decay feromon yang tersisa, rho laju penguapan
        self.counter = counter
        self.rng = rng
        self.D = counter.D
        self.n = len(self.D)
        self.eta = 1.0 / (self.D + np.eye(self.n)) #kota yang dekat lebih menarik, np.eye mencegah pembagian nol di diagonal
        tau0 = n_ants / (self.n * self.D.mean())
        self.pheromone = np.full((self.n, self.n), tau0)
        self.tau_min = tau0 * 1e-2 #batas bawah feromon agar sisi tidak pernah peluangnya nol, jadi koloni tidak macet di satu rute
        self.n_ants = n_ants
        self.n_best = n_best #jumlah semut yang boleh menambah feromon
        self.decay = decay
        self.alpha = alpha #mengatur feromon
        self.beta = beta #pengaruh jarak


    def run(self): #ini loop utama 
        best_tour, best_len = None, np.inf
        while not self.counter.done():
            all_paths = self.gen_all_paths()
            if not all_paths:
                break

            # catat rute terbaik
            tour, L = min(all_paths, key=lambda x: x[1])
            if L < best_len:
                best_tour, best_len = tour, L #simpan rute terbaik

            # penguapan, lalu jaga agar tidak turun di bawah batas
            self.pheromone *= self.decay #ini feromon menguap supaya jejak lama yang buruk terlupakan
            np.maximum(self.pheromone, self.tau_min, out=self.pheromone) #dipotong di tau_min
            self.spread_pheromone(all_paths)
        return best_tour, best_len

    def spread_pheromone(self, all_paths): #pembelajaran koloni
        sorted_paths = sorted(all_paths, key=lambda x: x[1]) #rute diurutkan dari yang terpendek
        for tour, L in sorted_paths[:self.n_best]: #disini hanya n_best yang menambah feromon (ambil 5 teratas saja)
            nxt = np.roll(tour, -1) 
            self.pheromone[tour, nxt] += 1.0 / L #sebesar 1/L
            self.pheromone[nxt, tour] += 1.0 / L

    def gen_all_paths(self): #seluruh koloni bekerja
        all_paths = []
        for _ in range(self.n_ants):
            if self.counter.done():
                break
            tour = self.gen_path(self.rng.integers(self.n))
            L = self.counter.evaluate(tour) #tiap semut membuat rute, lalu panjangnya dihitung dengan counter.evaluae
            #setiap evaluasi mengurangi budget, kalau habis semut berhenti
            all_paths.append((tour, L))
        return all_paths

    def gen_path(self, start): #satu semut membuat satu rute
        tour = [start] 
        visited = np.zeros(self.n, dtype=bool)
        visited[start] = True
        for _ in range(self.n - 1):
            nxt = self.pick_move(tour[-1], visited)
            tour.append(nxt)
            visited[nxt] = True #Untuk menjamin kalau tiap kota muncul tepat sekali agar rute valid
        return np.array(tour)

    def pick_move(self, cur, visited):
        row = (self.pheromone[cur] ** self.alpha) * (self.eta[cur] ** self.beta) #rumus pembilang, daya tiap kota
        row[visited] = 0  # kota yang sudah dikunjungi tidak bisa dipilih, diberi bobot 0
        total = row.sum() #penyebut
        if total <= 0:  # jaga-jaga kalau semua bobot 0
            return int(np.flatnonzero(~visited)[0])
        return int(self.rng.choice(self.n, p=row / total)) #bobot dinormalkan menjadi peluang dan rng.choice mengacak sesuai peluang
#roulette wheel selection -> ada unsur eksplorasi, tidak hanya di kota paling baik
# jadi semakin dekat dan banyak feromon, semakin menarik

def run_aco(counter, rng, **kwargs):
    AntColony(counter, rng, **kwargs).run()
    return counter