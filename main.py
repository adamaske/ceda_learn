import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

import cedalion
import cedalion.data
import cedalion.nirs
import cedalion.vis.anatomy
import cedalion.vis.colors as colors
import cedalion.vis.blocks as vbx

import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion.vis.quality import plot_quality_mask
import cedalion.sigproc.frequency as frequency

from cedalion import units


xr.set_options(display_expand_data=False)

# Get fingertapping dataset
rec = cedalion.data.get_fingertappingDOT()

# rec is a container of ordered dictornaries
# The fNIRS timeseries is stored in "amp"
print(rec.timeseries.keys())

amp = rec["amp"]

# Auxiliary sensors
print(rec.aux_ts.keys())

# Events
# stim is a pandas DataFrame
# .cd is a custom Cedalion accessor
rec.stim.cd.rename_events(
    {
        "1": "Rest",
        "2": "FTapping/Left",
        "3": "FTapping/Right",
        "4": "BallSqueezing/Left",
        "5": "BallSqueezing/Right",
    }
)
print(rec.stim)


# with pd.option_context("display.max_rows", 5):
#    print(rec.stim[rec.stim.trial_type.str.startswith("BallSqueezing")])
#
#
# with pd.option_context("display.max_rows", 5):
#    print(rec.stim[rec.stim.trial_type.str.endswith("Left")])

# Timeseries Data
amp = rec["amp"]
print(amp)

# Probe Geometry
print(rec.geo3d)
# cedalion.vis.anatomy.plot_montage3D(rec["amp"], rec.geo3d)
# Further reading: cedalion.typing.LabeledPoints


# Channel Data
def example_time_trace():
    amp = rec["amp"]
    ch = "S12D25"
    f, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.set_prop_cycle("color", cedalion.vis.colors.COLORBREWER_Q8)

    ax.plot(amp.time, amp.sel(channel=ch, wavelength=760), label="amp. 760 nm")
    ax.plot(amp.time, amp.sel(channel=ch, wavelength=850), label="amp. 850 nm")

    vbx.plot_stim_markers(ax, rec.stim, y=1)

    ax.set_xlabel("time / s")
    ax.set_ylabel("amplitude / V")
    ax.set_xlim(0, 150)
    ax.legend(loc="upper right")
    ax.set_title(ch)
    plt.show()


# example_time_trace()

# Quality Metrics
print("\n\n================= QUALITY METRICS =====================")

sci_threshold = 0.75
window_length = 10 * units.s
sci, sci_mask = quality.sci(rec["amp"], window_length, sci_threshold)

psp_threshold = 0.03
psp, psp_mask = quality.psp(rec["amp"], window_length, psp_threshold)

# print(sci.rename("sci"))
# print(sci_mask.rename("sci_mask"))


# Visualization of metrics and quality masks
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
        sci.time,
        np.arange(len(psp.channel)),
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
    ax.yaxis.set_ticklabels(psp.channel.values, fontsize=7)


# plot_sci(sci)
# plot_quality_mask(sci > sci_threshold, f"SCI > {sci_threshold}")
# plot_psp(psp)
# plot_quality_mask(psp > psp_threshold, f"PSP > {psp_threshold}")


combined_mask = sci_mask & psp_mask
# print(combined_mask)
# plot_quality_mask(combined_mask, "combined_mask")


# Percentage of the time
perc_time_clean = combined_mask.sum(dim="time") / len(sci.time)

print("Channels clean less that 95% of the recording:")
print(perc_time_clean[perc_time_clean < 0.95])


def visualize_percentage_clean():
    f, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    cedalion.vis.anatomy.scalp_plot(
        rec["amp"],
        rec.geo3d,
        perc_time_clean,
        ax,
        cmap="RdYlGn",
        vmin=0.7,
        vmax=1,
        title=None,
        cb_label="Percentage of clean time",
        channel_lw=2,
        optode_labels=True,
    )

    f.tight_layout()


visualize_percentage_clean()


# Motion Correction
rec["od"] = cedalion.nirs.cw.int2od(rec["amp"])
rec["od_tddr"] = motion.tddr(rec["od"])
rec["od_wavelet"] = motion.wavelet(rec["od_tddr"])
rec["amp_corrected"] = cedalion.nirs.cw.od2int(
    rec["od_wavelet"], rec["amp"].mean("time")
)

sci_corr, sci_corr_mask = quality.sci(
    rec["amp_corrected"], window_length, sci_threshold
)

psp_corr, psp_corr_mask = quality.psp(
    rec["amp_corrected"], window_length, psp_threshold
)

combined_corr_mask = sci_corr_mask & psp_corr_mask

plot_quality_mask(combined_mask, "combined mask")
plot_quality_mask(combined_corr_mask, "combined corrected mask")

plt.show()

# Compare Masks
changed_windows = (combined_mask == quality.TAINTED) & (
    combined_corr_mask == quality.CLEAN
)
plot_quality_mask(
    changed_windows,
    "mask of time winows cleaned by motion correction",
    bool_labels=["unchaged", "improved"],
)

changed_windows = (combined_mask == quality.CLEAN) & (
    combined_corr_mask == quality.TAINTED
)
plot_quality_mask(
    changed_windows,
    "mask of time windows corrupted by motion correction",
    bool_labels=["unchanged", "worsened"],
)


perc_time_clean_corr = combined_corr_mask.sum(dim="time") / len(sci.time)
f, ax = plt.subplots(1, 2, figsize=(14, 6.5))

cedalion.vis.anatomy.scalp_plot(
    rec["amp"],
    rec.geo3d,
    perc_time_clean,
    ax[0],
    cmap="RdYlGn",
    vmin=0.80,
    vmax=1,
    title="before correction",
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
    vmin=0.80,
    vmax=1,
    title="before correction",
    cb_label="Percentage of clean time",
    channel_lw=2,
    optode_labels=True,
)
f.tight_layout()
plt.show()

# Global Variance of the Temporal Derivate (GVTD)
gvtd, gvtd_mask = quality.gvtd(rec["amp"])
gvtd_corr, gvtd_prr_mask = quality.gvtd(rec["amp_corrected"])

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
example_channels = ["S4D10", "S13D26"]

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
            sel = rec["od_wavelet"].sel(
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
