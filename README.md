SEISMIC HORIZON PICKER GUI - VERSION 1
Copyright (c) 2025, E. K. Tripoliti, ucfbetr@ucl.ac.uk


Features: 
Load Seismic Data (SEG-Y Format): Load and visualize seismic data in the SEG-Y format.
Waveform Plotting: Visualize seismic waveforms and make manual horizon picks.
Automatic Horizon Picking: Automatically pick reflection horizons using envelope thresholding, Gaussian smoothing, and DBSCAN clustering.
Dip and Azimuth Estimation: Visualize dip, azimuth, and rose diagrams for horizon analysis.
Export Results: Export manual and automatic picks to Excel files for further analysis.
Interactive Interface: User-friendly interface for controlling picking parameters and visualizations.

Requirements:
Python 3.6+

Required Libraries:
PyQt5
segyio
numpy
pandas
scipy
matplotlib
openpyxl
sklearn (for DBSCAN)
multiprocessing (for parallel processing

Installation
Clone or download this repository.
Install the required dependencies (see above).
Run the script seismic_horizon_picker.py to launch the GUI.

Example Workflow:
Load a SEG-Y file with seismic data.
Adjust the envelope threshold ratio and smoothing parameters.
Pick horizons manually by clicking on the waveform plot.
Optionally, run the automatic picking algorithm.
Visualize the results on the seismic section, including dip and azimuth diagrams.
Export the picks to an Excel file for further analysis.

CITATION REQUIREMENT:
If you use this software in your research or publication, please cite it as follows:
E. K. Tripoliti, SEISMIC HORIZON PICKER GUI - VERSION 1, 2025, DOI:10.5281/zenodo.15647161

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and NONINFRINGEMENT. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the Software or the use or other dealings in the
Software.
