import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
import ligo.skymap.plot

# center = "10deg -40deg"
# center galactic coordinates 312deg -76deg

def read_summary_file(fit_base_path):
    with open(f"{fit_base_path}/fit_summary.txt", "r") as f:
        lines = f.readlines()
        ra, dec, K = [], [], []
        for line in lines:
            if line.startswith("Position"):
                ra.append(float(line.split()[1].strip(",")))
                dec.append(float(line.split()[2].strip(";")))
            elif line.startswith("K"):
                K.append(float(line.split()[1]))
    return ra, dec, K

def plot_result(fit_base_path):
    ra, dec, K = read_summary_file(fit_base_path)

    fig, ax = plt.subplots(figsize=(5,4), subplot_kw={'projection': 'galactic degrees zoom', 'center': '312deg -76deg', 'radius': '8deg'})

    pos = ax.scatter(ra, dec, c=K,transform=ax.get_transform('fk5'), cmap="viridis", s=15)
    ax.grid()

    ax.plot(
            10, -40,
            transform=ax.get_transform('fk5'),
            marker=ligo.skymap.plot.reticle(),
            markersize=10,
            markeredgewidth=2,
            c="tab:orange")

    fig.colorbar(pos, ax=ax, label="K")
    fig.savefig(f"{fit_base_path}/search_result.png")


if __name__ == "__main__":
    fit_base_path = "/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2"
    plot_result(fit_base_path)