# Problem Solving Agent untuk Traveling Salesman Problem

Perbandingan eksperimental GA-OX, GA-PMX, ACO, Hill Climbing (2-opt),
dan Simulated Annealing pada instance TSPLIB (eil51, berlin52, st70, kroA100).
Tugas Besar Fundamental AI.

## Struktur folder
- `src/`  : implementasi algoritma (aco.py, ga.py, baselines.py, tsp.py)
- `data/` : file instance TSPLIB (.tsp)
- `results/` : hasil eksperimen (CSV)
- `config.json` : parameter algoritma, seed, dan budget evaluasi
- `run_experiments.py` : menjalankan perbandingan utama
- `run_all_params.py`, `run_ga_params.py` : uji sensitivitas parameter ACO dan GA

## Cara menjalankan
pip install numpy pandas
python run_experiments.py
python run_all_params.py kroA100
python run_ga_params.py kroA100

## Hasil singkat
ACO menghasilkan rata-rata gap terendah pada keempat instance
(berlin52 optimum pada semua seed), tetapi memerlukan waktu komputasi
jauh lebih lama dibanding Hill Climbing dan SA.

## Laporan
Lihat `laporan_ieee.pdf`.
