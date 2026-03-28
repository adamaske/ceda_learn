# Cardiac analysis
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import cedalion
import cedalion.data
import cedalion.nirs
import cedalion.vis.anatomy
import cedalion.nirs.cw as cw
import cedalion.vis.colors as colors
import cedalion.vis.blocks as vbx
import cedalion.io.snirf as snirf_io

import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
from cedalion.vis.quality import plot_quality_mask

from cedalion import units

xr.set_options(display_expand_data=False)

# Load data
filepath = (
    r"C:\nirs\data\RH-data\Patient02\2026-01-21\2026-01-21_002\2026-01-21_002.snirf"
)

outpath = r"cardiac_data_autonomic.snirf"


rec = snirf_io.read_snirf(filepath)[0]
print(f"===={filepath}====")
print(rec)

print(f"====Auxiliary Signals====")
aux = rec.aux_ts
print(aux.keys())

print(f"====Stim====")
rec.stim.cd.rename_events(
    {
        "1": "baseline_start",
        "2": "baseline_end",
        "3": "count_start",
        "4": "count_end",
        "5": "pcl_start",
        "6": "pcl_end",
        "7": "ves_start",
        "8": "ves_end",
        "11": "induction_start",
        "12": "induction_end",
        "13": "arythmia_start",
        "14": "arythmia_end",
        "15": "changed_arythmia",
        "99": "experiment_end",
    }
)
print(rec.stim)

print("====Probe====")
print(rec.geo2d)
print(rec.geo3d)
# cedalion.vis.anatomy.plot_montage3D(rec["amp"], rec.geo3d)
# plt.show()

print("====Timeseries====")
amp = rec["amp"]
print(amp)

print("====Motion Correction====")
rec["od"] = cw.int2od(rec["amp"])
rec["od_tddr"] = motion.tddr(rec["od"])
rec["od_wl"] = motion.wavelet(rec["od_tddr"])

print("====Temporal Filtering====")
fmin = 0.01 * units.Hz
fmax = 0.2 * units.Hz
order = 5
rec["od"] = frequency.freq_filter(rec["od"], fmin, fmax)
rec["od_wl"] = frequency.freq_filter(rec["od_wl"], fmin, fmax)

print("====Wavelengths====")
wls = rec["od_wl"].wavelength.values
print(wls)

print("====MBLL====")
dpf = xr.DataArray(
    [6.0] * len(wls),
    dims=["wavelength"],
    coords={"wavelength": wls},
)

rec["conc"] = cw.od2conc(rec["od"], rec.geo3d, dpf)
rec["conc_corrected"] = cw.od2conc(rec["od_wl"], rec.geo3d, dpf)

print("====Output====")
snirf_io.write_snirf(outpath, rec)
