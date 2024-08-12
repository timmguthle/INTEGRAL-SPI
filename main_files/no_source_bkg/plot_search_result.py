import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import ligo.skymap.plot


# center = "10deg -40deg"
# center galactic coordinates 312deg -

def create_positions(n, delta=0.5):
    """
    Create a list of positions in a grid around (10, -40). Delta is the vertical and horizontal distance between the positions.
    the grid is n x n. To center the grid n should be odd.
    """
    positions = []
    for i in range(n//2, -n//2, -1):
        for j in range(n//2, -n//2, -1):
            positions.append([10+(i*delta), -40+(j*delta)])
    return positions


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

def plot_result(fit_base_path, galactic=True):
    ra, dec, K = read_summary_file(fit_base_path)

    class HandlerMarker(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            markerline = plt.Line2D([width / 2], [height / 2], ls="", marker=ligo.skymap.plot.reticle(), markersize=10, markeredgewidth=3, 
                                    color=orig_handle.get_color(), transform=trans)
            return [markerline]

    if galactic:
        fig, ax = plt.subplots(figsize=(6,5), subplot_kw={'projection': 'galactic degrees zoom', 'center': '312deg -76deg', 'radius': '8deg'})
    else:
        fig, ax = plt.subplots(figsize=(6,5), subplot_kw={'projection': 'astro degrees zoom', 'center': '10deg -40deg', 'radius': '8
                                                          deg'})



    pos = ax.scatter(ra, dec, c=K,transform=ax.get_transform('fk5'), cmap="viridis", s=15)
    ax.grid()

    crab_marker, = ax.plot(
            10, -40,
            transform=ax.get_transform('fk5'),
            marker=ligo.skymap.plot.reticle(),
            markersize=10,
            markeredgewidth=3,
            c="tab:orange", 
            label="center")
    
    ax.legend(handler_map={crab_marker: HandlerMarker()})

    fig.colorbar(pos, ax=ax, label="K")
    fig.savefig(f"{fit_base_path}/search_result.pdf")


def plot_positions(positions, galactic=True, fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2"):

    if not os.path.exists(fit_base_path):
        os.makedirs(fit_base_path)

    if not galactic:
        fig, ax = plt.subplots(figsize=(5,4), subplot_kw={'projection': 'astro degrees zoom', 'center': '10deg -40deg', 'radius': '20deg'})
    else:
        fig, ax = plt.subplots(figsize=(5,4), subplot_kw={'projection': 'galactic degrees zoom', 'center': '312deg -76deg', 'radius': '20deg'})

    ax.grid()
    ax.set_title('Positions of Potential sources')

    for ra, dec in positions:
        ax.scatter(ra, dec, transform=ax.get_transform('fk5'), c='tab:blue', s=40)

    fig.savefig(f"{fit_base_path}/positions.png")


def sweep_search_3():
    positions = create_positions(15, 2.25)
    plot_positions(positions=positions, fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_3")


if __name__ == "__main__":
    fit_base_path = "/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2"
    plot_result(fit_base_path, galactic=False)
    # sweep_search_3()