import nibabel as nib
import pydicom as dicom
# from pydicom.uid import ImplicitVRLittleEndian
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d, lfilter
from scipy.signal.windows import gaussian
import cv2

import os
import argparse

from non_parametric_deconvolution import modelfree_deconv

parser = argparse.ArgumentParser()
parser.add_argument('--dsa', '-d', type=str, required=True, help='path to the DSA series (DICOM, shape = [t, x, y])')
parser.add_argument('--aif', '-a', type=str, help='path to the AIF mask (NIFTI, shape = [x, y, t])')
parser.add_argument('--output', '-o', type=str, required=True, help='path to the output folder')
parser.add_argument('--fps', type=int, default=6, help='frame per second (default = 6)')
parser.add_argument('--cbf_threshold', type=float, default=0.02, help='CBF threshold (default = 0.02)')
parser.add_argument('--hct', type=float, default=0.45, help='hematocrit (default = 0.45)')
parser.add_argument('--show_aif', action='store_true', help='show AIF')
parser.add_argument('--show_results', action='store_true', help='show result maps')
parser.add_argument('--prefilter', '-f', action='store_true', help='apply gaussian filter to smooth AIF')
args = parser.parse_args()

'''
DSA
'''
IS_DICOM = True
# Load DSA
print('Reading DSA...')
try:
	dsa = dicom.read_file(args.dsa)
	dsa_seq = dsa.pixel_array  # dsa_seq.shape = (t, x, y)
	data = dsa_seq[..., np.newaxis]  # data.shape = (t, x, y, 1)
	data = np.transpose(data, axes=[1, 2, 3, 0])  # data.shape = (x, y, 1, t)
except dicom.errors.InvalidDicomError:
	try:
		dsa_seq = nib.load(args.dsa)
		dsa_seq = dsa_seq.get_fdata()  # dsa_seq.shape = (x, y, t)
		data = dsa_seq[..., np.newaxis]  # data.shape = (x, y, t, 1)
		data = np.transpose(data, axes=[0, 1, 3, 2])  # data.shape = (x, y, 1, t)
		IS_DICOM = False
	except nib.filebasedimages.ImageFileError:
		raise ValueError('Invalid input file type. Please specify a valid DICOM or NIFTI file.')
print(f'Done. Loaded data shape = {data.shape}.')

print('Preprocessing DSA...')
data[data == 0] = np.median(data)

m = np.min(data)
M = np.max(data)
data = 255. - (data - m) / (M - m) * 255.

data = data - np.median(data)
data[data < 0.] = 0.
print('Done.')

'''
AIF
'''

if args.aif is None:
	# Get AIF mask from user input
	# initialize flag, ix, iy, roi
	cv2.destroyAllWindows()
	drawing = False
	start, end = (-1, -1), (-1, -1)
	roi = np.empty((2, 2), dtype=int)

	# display first half of DSA summed together along axis=0
	img = np.sum(dsa_seq, axis=0)
	img = (img - np.min(img)) / (np.max(img) - np.min(img))

	# Mouse event
	def draw_roi(event, x, y, flags, param):
		global drawing, start, end, ix, iy, roi

		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			start = (x, y)
			ix, iy = x, y

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing:
				end = (x, y)

		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			cv2.rectangle(img, start, (x, y), (255, 0, 0), -1)
			start, end = (-1, -1), (-1, -1)
			roi = np.array([[ix, iy], [x, y]])
			print(f'ROI specified at\n{roi}. \nPress Enter to confirm...')


	# initialize window
	cv2.namedWindow('DSA', cv2.WINDOW_NORMAL)
	cv2.setMouseCallback('DSA', draw_roi)
	print('Please specify ROI for AIF (press Enter to confirm)...')

	while True:
		# cv2.resizeWindow('DSA', img.shape[0], img.shape[1])
		temp = np.copy(img)
		if drawing and end != (-1, -1):
			cv2.rectangle(temp, start, end, (255, 0, 0), -1)
		cv2.resizeWindow('DSA', 800, 800)
		cv2.imshow('DSA', temp)
		if cv2.waitKey(20) == ord('\r'):
			break
	cv2.destroyAllWindows()

	# Create ndarray from saved ROI vertices
	aif_mask = np.zeros_like(img)
	aif_mask[roi[0, 0]:roi[1, 0], roi[0, 1]:roi[1, 1]] = 1

	aif = np.mean(data[aif_mask > 0], axis=0)  # aif.shape = (1, t)
	aif = np.squeeze(aif)  # aif.shape = (t,)

else:
	# Load AIF mask from file
	aif_mask = nib.load(args.aif)
	aif_mask = aif_mask.get_fdata()
	aif_mask = np.sum(aif_mask, axis=2)  # aif.shape = (x, y)

	aif = np.mean(data[aif_mask > 0], axis=0)  # aif.shape = (1, t)
	aif = np.squeeze(aif)  # aif.shape = (t,)

# Apply gaussian filter to smooth AIF and data
if args.prefilter:
	m = max(3, args.fps)
	a = 1.66
	print('Applying gaussian filter to smooth AIF...')
	w = gaussian(m, (m - 1) / (2 * a))
	w = w / w.sum()
	aif = lfilter(w, 1, aif)
	print('Done.')

	print('Applying gaussian filter to smooth DSA...')

	data = lfilter(w, 1, data, axis=-1)
	print('Done.')

# Plot AIF
if args.show_aif:
	if args.aif is None:
		print(f'Displaying AIF at \n{roi}. \nClose the window to continue...')
	else:
		print(f'Displaying AIF from file. Close the window to continue...')
	timesteps = np.arange(len(aif), dtype=float)
	plt.title('Arterial Input Function')
	plt.xlabel('Time (frames)')
	plt.ylabel('Intensity (a.u.)')
	plt.plot(timesteps, aif, label='AIF')
	plt.legend()
	plt.show()

'''
Perfusion
'''
# Compute perfusion
print('Calculating perfusion maps...')
CBF, CBV, MTT, Tmax = modelfree_deconv(data, aif, dt=1. / args.fps, hct=args.hct, epsilon=1e-9, )
print('Done.')

# Post-process perfusion maps
print('Post-processing perfusion maps...')
CBF[CBF <= 0] = 0.
CBV[CBV <= 0] = 0.
MTT[MTT <= 0] = 0.
Tmax[Tmax <= 0] = 0.

MTT[CBF <= args.cbf_threshold] = 0.
Tmax[CBF <= args.cbf_threshold] = 0.

MTT = medfilt2d(MTT, kernel_size=3)
Tmax = medfilt2d(Tmax, kernel_size=3)
print('Done.')

# Save perfusion maps
if not os.path.exists(args.output):
	os.makedirs(args.output)

# if IS_DICOM:
# 	print('Saving perfusion maps as DICOM files...')
# 	dsa.FrameTimeVector = None
# 	dsa.NumberOfFrames = 1
# 	dsa.RepresentativeFrameNumber = None
#
# 	dsa.BitsAllocated = 64
# 	dsa.BitsStored = 64
# 	dsa.HighBit = dsa.BitsStored - 1
# 	del dsa.PixelData
#
# 	dsa.StudyDescription = 'CBF'
# 	dsa.SeriesNumber = 1001
# 	dsa.filename = 'CBF.dcm'
# 	cbf_uid = dicom.uid.generate_uid()
# 	dsa.StudyInstanceUID = cbf_uid
# 	# CBF1 = CBF[np.newaxis, ...]
# 	dsa.DoubleFloatPixelData = CBF.tobytes()
# 	dicom.dcmwrite(os.path.join(args.output, 'CBF.dcm'), dsa)
#
# 	dsa.StudyDescription = 'CBV'
# 	dsa.SeriesNumber = 1002
# 	dsa.filename = 'CBV.dcm'
# 	cbv_uid = dicom.uid.generate_uid()
# 	dsa.StudyInstanceUID = cbv_uid
# 	# CBV1 = CBV[np.newaxis, ...]
# 	dsa.DoubleFloatPixelData = CBV.tobytes()
# 	dicom.dcmwrite(os.path.join(args.output, 'CBV.dcm'), dsa)
#
# 	dsa.StudyDescription = 'MTT'
# 	dsa.SeriesNumber = 1003
# 	dsa.filename = 'MTT.dcm'
# 	mtt_uid = dicom.uid.generate_uid()
# 	dsa.StudyInstanceUID = mtt_uid
# 	# MTT1 = MTT[np.newaxis, ...]
# 	dsa.DoubleFloatPixelData = MTT.tobytes()
# 	dicom.dcmwrite(os.path.join(args.output, 'MTT.dcm'), dsa)
#
# 	dsa.StudyDescription = 'Tmax'
# 	dsa.SeriesNumber = 1004
# 	dsa.filename = 'Tmax.dcm'
# 	tmax_uid = dicom.uid.generate_uid()
# 	dsa.StudyInstanceUID = tmax_uid
# 	# Tmax1 = Tmax[np.newaxis, ...]
# 	dsa.DoubleFloatPixelData = Tmax.tobytes()
# 	dicom.dcmwrite(os.path.join(args.output, 'Tmax.dcm'), dsa)
#
# else:
print('Saving perfusion maps as NIFTI files...')
nib.save(nib.Nifti1Image(CBF, np.eye(4)), os.path.join(args.output, 'CBF.nii'))
nib.save(nib.Nifti1Image(CBV, np.eye(4)), os.path.join(args.output, 'CBV.nii'))
nib.save(nib.Nifti1Image(MTT, np.eye(4)), os.path.join(args.output, 'MTT.nii'))
nib.save(nib.Nifti1Image(Tmax, np.eye(4)), os.path.join(args.output, 'Tmax.nii'))
print('Done.')

# Display perfusion maps
if args.show_results:
	print('Displaying perfusion maps...')
	# making subplots
	fig, ax = plt.subplots(2, 2)

	# set data with subplots and plot
	norm = plt.Normalize(vmin=np.percentile(CBF, 0), vmax=np.percentile(CBF, 98))
	im00 = ax[0, 0].imshow(CBF, cmap='jet', norm=norm)
	ax[0, 0].set_title("CBF")
	plt.colorbar(im00, ax=ax[0, 0])

	norm = plt.Normalize(vmin=np.percentile(CBV, 0), vmax=np.percentile(CBV, 98))
	im01 = ax[0, 1].imshow(CBV, cmap='jet', norm=norm)
	ax[0, 1].set_title("CBV")
	plt.colorbar(im01, ax=ax[0, 1])

	norm = plt.Normalize(vmin=np.percentile(MTT, 0), vmax=np.percentile(MTT, 98))
	im10 = ax[1, 0].imshow(MTT, cmap='jet', norm=norm)
	ax[1, 0].set_title("MTT")
	plt.colorbar(im10, ax=ax[1, 0])

	norm = plt.Normalize(vmin=np.percentile(Tmax, 0), vmax=np.percentile(Tmax, 98))
	im11 = ax[1, 1].imshow(Tmax, cmap='jet', norm=norm)
	ax[1, 1].set_title("Tmax")
	plt.colorbar(im11, ax=ax[1, 1])

	fig.tight_layout()
	plt.show()
