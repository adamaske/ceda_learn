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

print("====Signal Quality====")
window_length = 10 * units.s

sci_threshold = 0.75
sci, sci_mask = quality.sci(rec["amp"], window_length, sci_threshold)

psp_threshold = 0.03
psp, psp_mask = quality.psp(rec["amp"], window_length, psp_threshold)

combined_mask = sci_mask & psp_mask

print(sci.rename("sci"))
print(sci_mask.rename("sci_mask"))

sci_norm, sci_cmap = colors.threshold_cmap("sci_cmap", 0.0, 1.0, sci_threshold)
psp_norm, psp_cmap = colors.threshold_cmap("psp_cmap", 0.0, 0.30, psp_threshold)


def plot_sci(sci):
    f, ax = plt.subplots(1, 1, figsize=(12, 10))
    m = ax.pcolormesh(
        sci.time,
        np.arange(len(sci.channel)),
        sci,
        shading="nearest",
        cmap=sci_cmap,
        norm=sci_norm,
    )
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("SCI")
    ax.set_xlabel("time / s")
    plt.tight_layout()
    ax.yaxis.set_ticks(np.arange(len(sci.channel)))
    ax.yaxis.set_ticklabels(sci.channel.values, fontsize=7)


def plot_psp(psp):
    f, ax = plt.subplots(1, 1, figsize=(12, 10))
    m = ax.pcolormesh(
        psp.time,
        np.arange(len(sci.channel)),
        psp,
        shading="nearest",
        cmap=psp_cmap,
        norm=psp_norm,
    )
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("PSP")
    ax.set_xlabel("time / s")
    plt.tight_layout()
    ax.yaxis.set_ticks(np.arange(len(psp.channel)))
    ax.yaxis.set_ticklabels(sci.channel.values, fontsize=7)


print("====Motion Correction====")
rec["od"] = cw.int2od(rec["amp"])
rec["od_tddr"] = motion.tddr(rec["od"])
rec["od_wl"] = motion.wavelet(rec["od_tddr"])
rec["amp_corrected"] = cw.od2int(rec["od_wl"], rec["amp"].mean("time"))

print("====Recalculate Quality & Compare====")
sci_corr, sci_corr_mask = quality.sci(
    rec["amp_corrected"], window_length, sci_threshold
)
psp_corr, psp_corr_mask = quality.psp(
    rec["amp_corrected"], window_length, psp_threshold
)
combined_corr_mask = sci_corr_mask & psp_corr_mask

# ==== Plot combined masks before & after correction
# plot_quality_mask(combined_mask, "combined mask")
# plot_quality_mask(combined_corr_mask, "combined corrected mask")

# ==== Individual Plots ====
# plot_sci(sci)"],
# plot_quality_mask(sci > sci_threshold, f"SCI > {sci_threshold}")
# plot_psp(psp)
# plot_quality_mask(psp > psp_threshold, f"PSP > {psp_threshold}")

# ==== Compare masks before and after correction ====
# What was improved by motion correction?
worsened_windows = (combined_mask == quality.CLEAN) & (
    combined_corr_mask == quality.TAINTED
)
improved_windows = (combined_mask == quality.TAINTED) & (
    combined_corr_mask == quality.CLEAN
)
# plot_quality_mask(
#    improved_windows,
#    "mask of time windows cleaned by motion correction",
#    bool_labels=["unchanged", "improved"],
# )
# plt.title("Improved Windows")

# plot_quality_mask(
#    worsened_windows,
#    "mask of time windows corrupted by motion correction",
#    bool_labels=["unchanged", "worsened"],
# )
# plt.title("Worsened Windows")

# ==== Calculate percentage of time clean ====
perc_time_clean = combined_mask.sum(dim="time") / len(sci.time)
perc_time_clean_corr = combined_corr_mask.sum(dim="time") / len(sci.time)
perc_threshold_low = 0.7
perc_threshold_high = 1.0
f, ax = plt.subplots(1, 2, figsize=(14, 6.5))
cedalion.vis.anatomy.scalp_plot(
    rec["amp"],
    rec.geo3d,
    perc_time_clean,
    ax[0],
    cmap="RdYlGn",
    vmin=perc_threshold_low,
    vmax=perc_threshold_high,
    title="Before Correction",
    cb_label="Percentage of clean time",
    channel_lw=2,
    optode_labels=True,
)

cedalion.vis.anatomy.scalp_plot(
    rec["amp"],
    rec.geo3d,
    perc_time_clean_corr,
    ax[1],
    cmap="RdYlGn",
    vmin=perc_threshold_low,
    vmax=perc_threshold_high,
    title="After Correction",
    cb_label="Percentage of clean time",
    channel_lw=2,
    optode_labels=True,
)
f.tight_layout()
plt.show()

# ====Global Variance of Time Derivative====
gvtd, gvtd_mask = quality.gvtd(rec["amp"])
gvtd_corr, gvtd_corr_mask = quality.gvtd(rec["amp_corrected"])
top10_bad_segments = sorted(
    [seg for seg in quality.mask_to_segments(combined_mask.all("channel"))],
    key=lambda t: gvtd.sel(time=slice(t[0], t[1])).max(),
    reverse=True,
)[:10]

f, ax = plt.subplots(4, 1, figsize=(16, 6), sharex=True)
ax[0].plot(gvtd.time, gvtd)
ax[1].plot(combined_mask.time, combined_mask.all("channel"))

ax[2].plot(gvtd_corr.time, gvtd_corr)
ax[3].plot(combined_corr_mask.time, combined_corr_mask.all("channel"))

ax[0].set_ylim(0, 0.02)
ax[2].set_ylim(0, 0.02)
ax[0].set_ylabel("GVTD")
ax[2].set_ylabel("GVTD")
ax[1].set_ylabel("all channels clean\n (before)")
ax[3].set_ylabel("all channels clean\n (after)")
ax[3].set_xlabel("time / s")

for i in range(4):
    vbx.plot_segments(ax[i], top10_bad_segments)
plt.show()

# ====Visualize motion correction in selected segments====
example_channels = ["S1D1", "S4D5"]

f, ax = plt.subplots(5, 4, figsize=(16, 16), sharex=False)
ax = ax.T.flatten()
padding = 15
i = 0
for ch in example_channels:
    for start, end in top10_bad_segments:
        ax[i].set_prop_cycle(color=["#e41a1c", "#ff7f00", "#377eb8", "#984ea3"])
        for wl in rec["od"].wavelength.values:
            sel = rec["od"].sel(
                time=slice(start - padding, end + padding), channel=ch, wavelength=wl
            )
            ax[i].plot(sel.time, sel, label=f"{wl:.0f} nm orig")
            sel = rec["od_wl"].sel(
                time=slice(start - padding, end + padding), channel=ch, wavelength=wl
            )
            ax[i].plot(sel.time, sel, label=f"{wl:.0f} nm corr")
            ax[i].set_title(ch)
        ax[i].legend(ncol=2, loc="upper center")
        ylim = ax[i].get_ylim()
        ax[i].set_ylim(
            ylim[0], ylim[1] + 0.25 * (ylim[1] - ylim[0])
        )  # make space for legend

        i += 1

plt.tight_layout()
plt.show()

print("====Channel Pruning====")
perc_time_clean_threshold = 0.5
num_bad_channels = perc_time_clean_corr[
    perc_time_clean_corr < perc_time_clean_threshold
]

signal_quality_selection_masks = [perc_time_clean >= perc_time_clean_threshold]
rec["amp_pruned"], pruned_channels = quality.prune_ch(
    rec["amp"], signal_quality_selection_masks, "all"
)

print(f"Time Clean Threshold: {perc_time_clean_threshold}")
print(f"Number of Bad Channels: {num_bad_channels}")
exit()
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
