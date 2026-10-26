"""Unduh ulang instans TSPLIB ke folder data/ (file .tsp sudah disertakan, ini hanya cadangan)."""
import urllib.request

BASE = "https://raw.githubusercontent.com/mastqe/tsplib/master/"
for name in ["eil51", "berlin52", "st70", "kroA100"]:
    urllib.request.urlretrieve(BASE + name + ".tsp", f"data/{name}.tsp")
    print("ok", name)
