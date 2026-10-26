"""Genetic Algorithm untuk TSP (ditulis sendiri, tanpa DEAP).

Kromosom  : permutasi kota, misal [3, 0, 4, 1, 2]  -> selalu valid
Fitness   : panjang rute (makin kecil makin baik)
Seleksi   : tournament (ukuran k)
Crossover : OX (Order Crossover) atau PMX (Partially Mapped Crossover)
Mutasi    : inversion (membalik satu segmen rute)
Elitisme  : sejumlah individu terbaik disalin langsung ke generasi berikut
"""
import numpy as np


def order_crossover(p1, p2, rng):
    """OX: salin segmen dari p1, sisanya diisi urutan kota dari p2."""
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    used = set(child[a:b + 1])
    rest = [c for c in np.roll(p2, -(b + 1)) if c not in used]
    pos = [(b + 1 + i) % n for i in range(n - (b - a + 1))]
    child[pos] = rest
    return child


def pmx_crossover(p1, p2, rng):
    """PMX: salin segmen dari p1, sisanya dari p2 dengan pemetaan agar tidak ada duplikat."""
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = p2.copy()
    child[a:b + 1] = p1[a:b + 1]
    mapping = {p1[i]: p2[i] for i in range(a, b + 1)}
    for i in list(range(0, a)) + list(range(b + 1, n)):
        c = p2[i]
        while c in mapping:
            c = mapping[c]
        child[i] = c
    return child


def inversion_mutation(tour, rng):
    """Balik urutan satu segmen acak (setara satu langkah 2-opt)."""
    a, b = sorted(rng.choice(len(tour), 2, replace=False))
    tour = tour.copy()
    tour[a:b + 1] = tour[a:b + 1][::-1]
    return tour


def run_ga(counter, random_generator, pop_size=100, crossover="ox", pc=0.9, pm=0.2,
           tournament_k=3, elite_count=2):
    #1. Tentukan jumlah kota
    num_cities = len(counter.D)

    #2. Pilih metode crossover
    if crossover == "ox":
        crossover_function = order_crossover
    else:
        crossover_function = pmx_crossover

    #3. Membuat populasi berisi rute-rute acak
    population = []
    for _ in range(pop_size):
        route = random_generator.permutation(num_cities)
        population.append(route)

    #4. Evaluasi semua rute dalam populasi awal

    route_distances=[]

    for route in population:
        distance = counter.evaluate(route)
        route_distances.append(distance)

    route_distances=np.array(route_distances)

    #5. Fungsi untuk memilih satu induk

    def select_parent():
        candidate_indices = random_generator.choice(
            len(population),
            tournament_k,
            replace=False
        )

        candidate_distances = route_distances[candidate_indices]

        best_position = np.argmin(candidate_distances)
        best_candidate_index = candidate_indices[best_position]

        return population[best_candidate_index]
    
    #6. Evolusi populasi sampai anggaran evaluasi habis
    while not counter.done():

        #urutkan individu dari jarak terkecil
        sorted_indices = np.argsort(route_distances)

        #salin individu terbaik sebagai elit
        new_population = [
            population[index].copy()
            for index in sorted_indices[:elite_count]
        ]

        new_route_distances=[
            route_distances[index]
            for index in sorted_indices[:elite_count]
        ]

        #buat anak sampai populasi baru penuh
        while(
            len(new_population)<pop_size and not counter.done()
        ):
            #pilih 2 induk
            parent1=select_parent()
            parent2=select_parent()

            #lakukan crossover atau salin induk pertama
            if random_generator.random()<pc:
                child = crossover_function(
                    parent1,
                    parent2,
                    random_generator
                )
            else:
                child=parent1.copy()

            #lakukan mutasi jika terpilih
            if random_generator.random()<pm:
                child=inversion_mutation(
                    child, 
                    random_generator
                )

            #Evaluasi anak dan masukkan ke populasi baru
            child_distance=counter.evaluate(child)

            new_population.append(child)
            new_route_distances.append(child_distance)

        #ganti populasi lama dengan populasi baru
        population = new_population
        route_distances = np.array(new_route_distances)

    return counter

        

