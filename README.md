PerfusionDSA
============

This simple programs calculates perfusion parametric images including 

* Cerebral Blood Flow (CBF),
* Cerebral Blood Volume (CBV),
* Mean Transit Time (MTT), and 
* Time to max flow-scaled residue function (Tmax)

from a series of 2D+t images produced by cerebral X-ray angiography.

The program is adopted from [perfusionDSA](https://github.com/RuishengSu/perfusionDSA).
The main function `non_parametric_deconvolution.py` is modified from 
[non_parametric_deconvolution](https://github.com/liob/non-parametric_deconvolution).

Dependencies
------------
* NumPy
* Nibabel
* Pydicom
* Scipy
* Matplotlib

Usage
-----
For a Digital Subtraction Angiogram (DSA) series of shape `[t, x, y]` saved as a DICOM file `./dsa.dcm`, 
an (optional) arterial input function (AIF) mask of shape `[x, y, t]` saved as a NIfTI file `./aif.nii`, 
the following script saves the parametric images as NIfTI files `CBF.nii`, `CBV.nii`, `MTT.nii`, and `Tmax.nii`
under the directory specified by `--output` or `-o`.

    python perfusiondsa.py -d ./dsa.dcm -a ./aif.nii -o ./output

If an AIF mask is not specified by `--aif` or `-a`, 
the user will be prompted to draw a region of interest (ROI) from which the AIF will be obtained.
Note: only the last ROI drawn will be recorded.

The `--show_aif` and `--show_results` flags 
can be used to plot the AIF and display the parametric maps, respectively.

To apply a Gaussian filter to the AIF and the DSA time series, 
use the option `--prefilter` or `-f`.



