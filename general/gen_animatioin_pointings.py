import numpy as np
import astropy.io.fits as fits
import ligo.skymap.plot

import healpy as hp
from matplotlib.colors import TwoSlopeNorm
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

with fits.open('/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/dataset_skymap374-2/spi/pointing.fits') as hdul:
    hdul.info()
    pointings_43 = hdul[1].data

ra = pointings_43['RA_SPIX']
dec = pointings_43['DEC_SPIX']
p_id = pointings_43["PTID_ISOC"]
ind = np.arange(len(ra))

fig, ax = plt.subplots(figsize=(5,4), subplot_kw={'projection': 'astro degrees zoom', 'center': '10deg -40deg', 'radius': '7deg'})

ax.grid()
ax.set_title('Pointings of observation 374')

def animate(i):
    points = []
    ax.set_title(f'Pointing {p_id[i]}')
    if i >= 1:
        points.append(ax.scatter(ra[i-1], dec[i-1], transform=ax.get_transform('fk5'), c='tab:blue', s=40, alpha=0.5))
        points.append(ax.plot(ra[i-1:i+1], dec[i-1:i+1], transform=ax.get_transform('fk5'), c='tab:blue', linestyle=':', alpha=0.7)[0])
    points.append(ax.scatter(ra[i], dec[i], transform=ax.get_transform('fk5'), c='tab:orange', s=50))
    return points

ani = FuncAnimation(fig, animate, frames=len(ra), interval=200, blit=True)
ani.save('pointings_374-2.mp4', writer='ffmpeg', fps=1)

#HTML(ani.to_jshtml())