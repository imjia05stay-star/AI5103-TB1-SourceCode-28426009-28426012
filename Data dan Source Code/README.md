# Problem Solving Agent untuk TSP: GA vs ACO vs Baseline

Tugas AI5103 — membandingkan **Genetic Algorithm**, **Ant Colony Optimization**, dan dua baseline
(**Hill Climbing 2-opt random restart** dan **Simulated Annealing**) pada Traveling Salesman Problem.

## Data
Instans dari **TSPLIB** (Reinelt, 1991), jenis jarak `EUC_2D`, sudah ada di folder `data/`.

| Instans | Jumlah kota | Optimum |
|---|---|---|
| eil51 | 51 | 426 |
| berlin52 | 52 | 7542 |
| st70 | 70 | 675 |
| kroA100 | 100 | 21282 |

Sumber: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ (salinan: https://github.com/mastqe/tsplib).
Unduh ulang: `python download_data.py`.

## Instalasi
```bash
pip install -r requirements.txt
```

## Cara mereproduksi semua tabel & gambar (2 perintah)
```bash
python run_experiments.py   # semua run (paralel di semua core CPU) -> results/*.csv
python make_report.py       # tabel, uji Wilcoxon, kurva, sensitivitas -> results/ & figures/
```
Semua parameter dan seed ada di `config.json`.

## Demo untuk presentasi
```bash
python demo.py                       # berlin52, animasi rute + kurva konvergensi
python demo.py --instance eil51
```

## Struktur
```
src/tsp.py         baca data, jarak, panjang rute, penghitung evaluasi
src/ga.py          Genetic Algorithm (OX / PMX, tournament, inversion mutation, elitisme)
src/aco.py         Ant System
src/baselines.py   Hill Climbing 2-opt dan Simulated Annealing
run_experiments.py eksperimen utama + analisis sensitivitas
make_report.py     ringkasan statistik dan gambar
demo.py            demo animasi
results/           runs.csv, convergence.csv, sensitivity.csv, summary_table.csv, wilcoxon.csv
figures/           gambar untuk laporan
```

## Keadilan perbandingan
Semua algoritma diberi anggaran yang sama: **50.000 evaluasi solusi** per run
(GA: 1 anak = 1 evaluasi; ACO: 1 rute semut = 1 evaluasi; HC/SA: 1 tetangga 2-opt = 1 evaluasi).
Setiap algoritma × instans dijalankan 5 kali (seed 1–5).
