import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cedalion.io.snirf as snirf_io
import cedalion
import cedalion.dot
import cedalion.nirs.cw as cw
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.physio as physio
from cedalion import units

import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
import cedalion.vis.anatomy as anat_vis


from ceda_correction import correct_and_prune

xr.set_options(display_expand_data=False)

filepath = r"C:\dev\NIRWizard\examples\example_snirf_data\2025-05-19_004.snirf"

HEAD_MODEL = "colin27"
DATASET = "fingeratppingDOT"
PRECOMPUTED_SENSITIVITY = True
FORWARD_MODEL = "MCX"

INTERACTIVE_PLOTS = False

from pathlib import Path
from tempfile import TemporaryDirectory

temp_dir = TemporaryDirectory()
cwd = Path(temp_dir.name)

rec = snirf_io.read_snirf(filepath, time_units="s")[0]

anat_vis.plot_montage3D(rec["amp"], rec.geo3d)
plt.show()


rec.stim.cd.rename_events(
    {
        "0": "Rest",
        "1": "Left",
        "2": "Right",
    }
)

# Preprocessing
rec["od"] = cw.int2od(rec["amp"])
rec["od_tddr"] = motion.tddr(rec["od"])
rec["od_wavelet"] = motion.wavelet(rec["od_tddr"])

print("====Temporal Filtering====")
fmin = 0.01 * units.Hz
fmax = 0.5 * units.Hz
rec["od_filtered"] = frequency.freq_filter(rec["od_wavelet"], fmin, fmax)

od_var = quality.measurement_variance(rec["od_wavelet"], calc_covariance=False)

rec["od_mean_subtracted"], global_comp = physio.global_component_subtract(
    rec["od_filtered"], ts_weights=1 / od_var, k=0
)

rec["od_corrected"] = rec["od_mean_subtracted"].cd.freq_filter(
    fmin=0.1, fmax=0.5, butter_order=4
)

epochs = rec["od_corrected"].cd.to_epochs(
    rec.stim,
    ["Left", "Right"],
    before=5 * units.s,
    after=30 * units.s,
)

baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
# Subtract baseline
epochs_blcorrected = epochs - baseline

blockaverage = epochs_blcorrected.groupby("trial_type").mean("epoch")

# Plot block averages.
noPlts2 = int(np.ceil(np.sqrt(len(blockaverage.channel))))
f, ax = plt.subplots(noPlts2, noPlts2, figsize=(12, 8))
ax = ax.flatten()
for i_ch, ch in enumerate(blockaverage.channel):
    for ls, trial_type in zip(["-", "--"], blockaverage.trial_type):
        ax[i_ch].plot(
            blockaverage.reltime,
            blockaverage.sel(wavelength=760, trial_type=trial_type, channel=ch),
            "r",
            lw=2,
            ls=ls,
        )
        ax[i_ch].plot(
            blockaverage.reltime,
            blockaverage.sel(wavelength=850, trial_type=trial_type, channel=ch),
            "b",
            lw=2,
            ls=ls,
        )

    ax[i_ch].grid(1)
    ax[i_ch].set_title(ch.values)
    ax[i_ch].set_ylim(-0.01, 0.01)
    ax[i_ch].set_axis_off()
    ax[i_ch].axhline(0, c="k")
    ax[i_ch].axvline(0, c="k")

plt.suptitle("760nm: r | 850nm: b | rest: - | stim: --")
plt.tight_layout()
plt.show()

if HEAD_MODEL in ["colin27", "icbm152"]:
    head_ijk = cedalion.dot.get_standard_headmodel(HEAD_MODEL)
elif HEAD_MODEL == "custom":
    segm_datadir = Path("/path/to/dir/with/segmentation_masks")
    mask_files = {
        "csf": "mask_csf.nii",
        "gm": "mask_gray.nii",
        "scalp": "mask_skin.nii",
        "skull": "mask_bone.nii",
        "wm": "mask_white.nii",
    }
    # The landmarks must be in scannar RAS space.
    # For example Slicer3D can be used to pick landmarks
    landmarks_file = Path("path/to/landmarks.mrk.json")

    # if available provide a mapping between vertices and labels
    parcel_file = None

    # Construct a head model from segmentation mask.
    head_ijk = cedalion.dot.TwoSurfaceHeadModel.from_segmentation(
        segmentation_dir=segm_datadir,
        mask_files=mask_files,
        landmarks_ras_file=landmarks_file,
        parcel_file=parcel_file,
        # adjust these to control mesh parameters
        brain_face_count=None,
        scalp_face_count=None,
        smoothing=0.0,
    )

    # Likely, better brain and scalp surfaces are achievable from
    # specialized segmentation tools.
    head_ijk = cedalion.dot.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=segm_datadir,
        mask_files=mask_files,
        landmarks_ras_file=landmarks_file,
        parcel_file=parcel_file,
        brain_surface_file=segm_datadir / "mask_brain.obj",
        scalp_surface_file=segm_datadir / "mask_scalp.obj",
    )
else:
    raise ValueError("Unkown head model")
