from astropy.time import Time
import numpy as np
from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI


grbtime = Time('2012-07-11T02:44:53', format='isot', scale='utc')

ebounds = np.geomspace(20, 8000, 100)

det = 0

tsb = TimeSeriesBuilderSPI(f"SPIDet{det}", det, grbtime, ebounds=ebounds, sgl_type="both")