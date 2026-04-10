# Cardiac analysis
import xarray as xr
import matplotlib.pyplot as plt

import cedalion.io.snirf as snirf_io
import cedalion.nirs.cw as cw
import cedalion.sigproc.frequency as frequency
from cedalion import units

import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality

from ceda_correction import correct_and_prune

xr.set_options(display_expand_data=False)

# Load data
filepath = (
    r"C:\nirs\data\RH-data\Patient02\2026-01-21\2026-01-21_002\2026-01-21_002.snirf"
)
outpath = r"cardiac_pruned.snirf"

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

print("====Timeseries====")
print(rec["amp"])

print("====Motion Correction & Channel Pruning====")
rec, pruned_channels = correct_and_prune(
    rec,
    sci_threshold=0.75,
    psp_threshold=0.03,
    window_length=10 * units.s,
    perc_time_clean_threshold=0.65,
    example_channels=["S1D1", "S4D5"],
    visualize=False,
)
# plt.show()

print(f"Pruned channels: {pruned_channels}")


print("====Temporal Filtering====")
fmin = 0.01 * units.Hz
fmax = 0.2 * units.Hz
order = 5
rec["od_corrected"] = frequency.freq_filter(rec["od_corrected"], fmin, fmax)

print("====Wavelengths====")
wls = rec["od_corrected"].wavelength.values
print(wls)

print("====MBLL====")
dpf = xr.DataArray(
    [6.0] * len(wls),
    dims=["wavelength"],
    coords={"wavelength": wls},
)

rec["conc_corrected"] = cw.od2conc(rec["od_corrected"], rec.geo3d, dpf)

print("====Output====")
snirf_io.write_snirf(outpath, rec)
print(f"Wrote .snirf to {outpath}")
