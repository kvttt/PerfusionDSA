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
For a DSA series of shape `[t, x, y]` saved as a DICOM file `./dsa.dcm`, 
an arterial input function mask of shape `[x, y, t]` saved as a NIfTI file `./aif.nii`, 
the following script saves the parametric images as NIfTI files `CBF.nii`, `CBV.nii`, `MTT.nii`, and `Tmax.nii`
under the directory specified by `--output` or `-o`.

    python perfusiondsa.py -d ./dsa.dcm -a ./aif.nii -o ./output

The `--show_aif` and `--show_results` flags 
can be used to plot the arterial input function and display the parametric maps, respectively.



