# Gold Atlas - Male Pelvis MRI and CT Scans

This dataset contains paired samples of MRIs and CT scans from the pelvic region of 19 male patients. Additionally, there is a CT image which is deformed and registered to the MRI data. The data is collected from 3 different machines across 3 location.

If using this data, please cite:

Nyholm, Tufve, Stina Svensson, Sebastian Andersson, Joakim Jonsson, Maja Sohlin, Christian Gustafsson, Elisabeth Kjellén, et al. 2018. “MR and CT Data with Multi Observer Delineations of Organs in the Pelvic Area - Part of the Gold Atlas Project.” Medical Physics 12 (10): 3218–21. doi:10.1002/mp.12748.

## Access

Data is accessed from <https://zenodo.org/records/583096>. Request to access can also be made at the same link.

## Processing

Each patient archive is extracted into its own folder. Each of these folders contain a number of `.dcm` files. Two python scripts are used to process this data. 

The scripts are run using python v3.10.12. The first, does all the pre-processing and stores the results into a directory with numpy files. The second converts the numpy files into paired PNG files.

To get the processed data, run the following:

```
python3 data_preprocess.py /path/to/dcm/folders /path/to/numpy
python3 joint_png.py /path/to/numpy /path/to/png
```

## Train/Test/Validation set split

The splitting of sets is done using absolute names, and are therefore hard-coded. Details are available in the processing script.

