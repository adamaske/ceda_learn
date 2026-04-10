"""
ceda_correction.py

Reusable motion correction and channel pruning for cedalion fNIRS recordings.

Usage:
    from ceda_correction import correct_and_prune

    rec, pruned_channels = correct_and_prune(rec, visualize=True)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cedalion.vis.anatomy
import cedalion.nirs.cw as cw
import cedalion.vis.colors as colors
import cedalion.vis.blocks as vbx

import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion import units


def compute_quality(amp, window_length, sci_threshold, psp_threshold):
    """Calculate SCI and PSP quality metrics and masks.

    Parameters
    ----------
    amp : xarray.DataArray
        Amplitude timeseries.
    window_length : pint Quantity
        Sliding window length (e.g. ``10 * units.s``).
    sci_threshold : float
        SCI threshold for clean/tainted classification.
    psp_threshold : float
        PSP threshold for clean/tainted classification.

    Returns
    -------
    sci, sci_mask, psp, psp_mask, combined_mask : xarray.DataArrays
    sci_norm, sci_cmap, psp_norm, psp_cmap : colormap helpers for plotting
    """
    sci, sci_mask = quality.sci(amp, window_length, sci_threshold)
    psp, psp_mask = quality.psp(amp, window_length, psp_threshold)
    combined_mask = sci_mask & psp_mask

    sci_norm, sci_cmap = colors.threshold_cmap("sci_cmap", 0.0, 1.0, sci_threshold)
    psp_norm, psp_cmap = colors.threshold_cmap("psp_cmap", 0.0, 0.30, psp_threshold)

    return (
        sci,
        sci_mask,
        psp,
        psp_mask,
        combined_mask,
        sci_norm,
        sci_cmap,
        psp_norm,
        psp_cmap,
    )


def run_motion_correction(rec):
    """Apply TDDR + wavelet motion correction to a cedalion recording.

    Adds the following keys to ``rec``:
    - ``od``            — optical density
    - ``od_tddr``       — OD after TDDR
    - ``od_wl``         — OD after TDDR + wavelet filtering
    - ``amp_corrected`` — corrected amplitude (back-converted from OD)

    Parameters
    ----------
    rec : cedalion Recording
        Recording with an ``amp`` timeseries.

    Returns
    -------
    rec : cedalion Recording
        Same object with new keys added in-place.
    """
    rec["od"] = cw.int2od(rec["amp"])
    rec["od_tddr"] = motion.tddr(rec["od"])
    rec["od_wl"] = motion.wavelet(rec["od_tddr"])
    rec["amp_corrected"] = cw.od2int(rec["od_wl"], rec["amp"].mean("time"))
    return rec


def prune_channels(rec, perc_time_clean_corr, threshold):
    """Remove channels below a minimum percentage of clean time.

    Parameters
    ----------
    rec : cedalion Recording
        Recording with an ``amp`` timeseries.
    perc_time_clean_corr : xarray.DataArray
        Per-channel fraction of clean time after motion correction.
    threshold : float
        Channels with ``perc_time_clean_corr < threshold`` are pruned.

    Returns
    -------
    rec : cedalion Recording
        Same object with ``amp_pruned`` added.
    pruned_channels : list
        Channel labels that were removed.
    """
    selection_masks = [perc_time_clean_corr >= threshold]
    rec["amp_pruned"], pruned_channels = quality.prune_ch(
        rec["amp"], selection_masks, "all"
    )
    return rec, pruned_channels


def plot_sci(sci, sci_norm, sci_cmap):
    """Plot SCI as a channel × time heatmap."""
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


def plot_psp(psp, psp_norm, psp_cmap):
    """Plot PSP as a channel × time heatmap."""
    f, ax = plt.subplots(1, 1, figsize=(12, 10))
    m = ax.pcolormesh(
        psp.time,
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


def _plot_correction_results(
    rec,
    combined_mask,
    combined_corr_mask,
    perc_time_clean,
    perc_time_clean_corr,
    perc_threshold_low=0.7,
    perc_threshold_high=1.0,
    example_channels=None,
    padding=15,
):
    """Internal: render all correction visualizations."""
    # ---- Scalp plots: % clean time before/after ----
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

    # ---- GVTD + clean mask panel ----
    gvtd, _ = quality.gvtd(rec["amp"])
    gvtd_corr, _ = quality.gvtd(rec["amp_corrected"])
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
    ax[1].set_ylabel("all channels clean\n(before)")
    ax[3].set_ylabel("all channels clean\n(after)")
    ax[3].set_xlabel("time / s")
    for i in range(4):
        vbx.plot_segments(ax[i], top10_bad_segments)

    # ---- Per-channel OD traces around bad segments ----
    if example_channels:
        f, ax = plt.subplots(
            5, len(example_channels) * 2, figsize=(16, 16), sharex=False
        )
        ax = ax.T.flatten()
        i = 0
        for ch in example_channels:
            for start, end in top10_bad_segments:
                ax[i].set_prop_cycle(color=["#e41a1c", "#ff7f00", "#377eb8", "#984ea3"])
                for wl in rec["od"].wavelength.values:
                    sel = rec["od"].sel(
                        time=slice(start - padding, end + padding),
                        channel=ch,
                        wavelength=wl,
                    )
                    ax[i].plot(sel.time, sel, label=f"{wl:.0f} nm orig")
                    sel = rec["od_wl"].sel(
                        time=slice(start - padding, end + padding),
                        channel=ch,
                        wavelength=wl,
                    )
                    ax[i].plot(sel.time, sel, label=f"{wl:.0f} nm corr")
                    ax[i].set_title(ch)
                ax[i].legend(ncol=2, loc="upper center")
                ylim = ax[i].get_ylim()
                ax[i].set_ylim(ylim[0], ylim[1] + 0.25 * (ylim[1] - ylim[0]))
                i += 1
        plt.tight_layout()


def correct_and_prune(
    rec,
    sci_threshold=0.75,
    psp_threshold=0.03,
    window_length=None,
    perc_time_clean_threshold=0.5,
    example_channels=None,
    visualize=True,
):
    """Apply motion correction and channel pruning to a cedalion recording.

    Parameters
    ----------
    rec : cedalion Recording
        Recording with an ``amp`` timeseries. Modified in-place.
    sci_threshold : float
        SCI threshold for clean/tainted classification (default 0.75).
    psp_threshold : float
        PSP threshold for clean/tainted classification (default 0.03).
    window_length : pint Quantity, optional
        Quality metric window length. Defaults to ``10 * units.s``.
    perc_time_clean_threshold : float
        Channels with less than this fraction of clean time are pruned (default 0.5).
    example_channels : list of str, optional
        Channel names to use for the per-segment OD visualisation. If ``None``
        that plot is skipped even when ``visualize=True``.
    visualize : bool
        Whether to show plots (default True). Set to False for batch processing.

    Returns
    -------
    rec : cedalion Recording
        Same object with added keys: ``od``, ``od_tddr``, ``od_wl``,
        ``amp_corrected``, ``amp_pruned``.
    pruned_channels : list
        Channel labels removed during pruning.
    """
    if window_length is None:
        window_length = 10 * units.s

    print("Computing pre-correction signal quality...")
    sci, sci_mask, psp, psp_mask, combined_mask, *_ = compute_quality(
        rec["amp"], window_length, sci_threshold, psp_threshold
    )

    print("Applying motion correction (TDDR + wavelet)...")
    rec = run_motion_correction(rec)

    print("Computing post-correction signal quality...")
    sci_corr, sci_corr_mask, psp_corr, psp_corr_mask, combined_corr_mask, *_ = (
        compute_quality(
            rec["amp_corrected"], window_length, sci_threshold, psp_threshold
        )
    )

    perc_time_clean = combined_mask.sum(dim="time") / len(sci.time)
    perc_time_clean_corr = combined_corr_mask.sum(dim="time") / len(sci_corr.time)

    if visualize:
        print("Plotting results...")
        _plot_correction_results(
            rec,
            combined_mask,
            combined_corr_mask,
            perc_time_clean,
            perc_time_clean_corr,
            example_channels=example_channels,
        )

    print("Pruning channels...")
    rec, pruned_channels = prune_channels(
        rec, perc_time_clean_corr, perc_time_clean_threshold
    )

    num_pruned = len(pruned_channels) if hasattr(pruned_channels, "__len__") else "?"
    print(
        f"Done. Pruned {num_pruned} channel(s) below {perc_time_clean_threshold:.0%} clean time."
    )

    return rec, pruned_channels
