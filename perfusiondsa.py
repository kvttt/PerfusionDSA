import nibabel as nib
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d

import os
import argparse

from non_parametric_deconvolution import modelfree_deconv

parser = argparse.ArgumentParser()
parser.add_argument('--dsa', '-d', type=str, required=True, help='path to the DSA series (DICOM, shape = [t, x, y])')
parser.add_argument('--aif', '-a', type=str, required=True, help='path to the AIF mask (NIFTI, shape = [x, y, t])')
parser.add_argument('--output', '-o', type=str, required=True, help='path to the output folder')
parser.add_argument('--fps', type=int, default=6, help='frame per second (default = 6)')
parser.add_argument('--cbf_threshold', type=float, default=0.02, help='CBF threshold (default = 0.02)')
parser.add_argument('--hct', type=float, default=0.45, help='hematocrit (default = 0.45)')
parser.add_argument('--show_aif', action='store_true', help='show AIF')
parser.add_argument('--show_results', action='store_true', help='show result maps')
args = parser.parse_args()


'''
DSA
'''
# Load DSA
dsa_seq = dicom.read_file(args.dsa)
data = dsa_seq.pixel_array
data = data[..., np.newaxis]  # dsa_seq.shape = (t, x, y, 1)
data = np.transpose(data, axes=[1, 2, 3, 0])  # data.shape = (x, y, 1, t)

# Preprocess DSA
data[data == 0] = np.median(data)

m = np.min(data)
M = np.max(data)
data = 255. - (data - m) / (M - m) * 255.

data = data - np.median(data)
data[data < 0.] = 0.

'''
AIF
'''
# Load AIF mask
aif_mask = nib.load(args.aif)
aif_mask = aif_mask.get_fdata()
aif_mask = np.sum(aif_mask, axis=2)  # aif.shape = (x, y)

aif = np.mean(data[aif_mask > 0], axis=0)  # aif.shape = (1, t)
aif = np.squeeze(aif)  # aif.shape = (t,)

# Plot AIF
if args.show_aif:
    timesteps = np.arange(len(aif), dtype=float)
    plt.title('Arterial Input Function')
    plt.xlabel('Time (frames)')
    plt.ylabel('Intensity (a.u.)')
    plt.plot(timesteps, aif, label='AIF')
    plt.legend()

'''
Perfusion
'''
# Compute perfusion
CBF, CBV, MTT, Tmax = modelfree_deconv(data,
                                       aif,
                                       dt=1. / args.fps,
                                       hct=args.hct,
                                       epsilon=1e-9,
                                       )

# Post-process perfusion maps
CBF[CBF <= 0] = 0.
CBV[CBV <= 0] = 0.
MTT[MTT <= 0] = 0.
Tmax[Tmax <= 0] = 0.

MTT[CBF <= args.cbf_threshold] = 0.
Tmax[CBF <= args.cbf_threshold] = 0.

MTT = medfilt2d(MTT, kernel_size=3)
Tmax = medfilt2d(Tmax, kernel_size=3)

# Display perfusion maps
if args.show_aif:
    # making subplots
    fig, ax = plt.subplots(2, 2)

    # set data with subplots and plot
    ax[0, 0].imshow(CBF, cmap='gray')
    ax[0, 0].set_title("CBF")
    ax[0, 1].imshow(CBV, cmap='gray')
    ax[0, 1].set_title("CBV")
    ax[1, 0].imshow(MTT, cmap='gray')
    ax[1, 0].set_title("MTT")
    ax[1, 1].imshow(Tmax, cmap='gray')
    ax[1, 1].set_title("Tmax")

    fig.tight_layout()
    plt.show()

# Save perfusion maps
if not os.path.exists(args.output):
    os.makedirs(args.output)

nib.save(nib.Nifti1Image(CBF, np.eye(4)), os.path.join(args.output, 'CBF.nii'))
nib.save(nib.Nifti1Image(CBV, np.eye(4)), os.path.join(args.output, 'CBV.nii'))
nib.save(nib.Nifti1Image(MTT, np.eye(4)), os.path.join(args.output, 'MTT.nii'))
nib.save(nib.Nifti1Image(Tmax, np.eye(4)), os.path.join(args.output, 'Tmax.nii'))
