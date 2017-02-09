# README #


### What is this repository for? ###

This is python code that allows for animal behaviour analysis and animal electrophysiology (ECoG and spikes) analysis.
This code is meant to run on python 3.5 although no python 3 specific code has been implemented..

### How do I get set up? ###
You need to have all the required libraries that the different scripts import. Main libraries used are:  
numpy  
scipy  
matplotlib  
pandas  
mne  
opencv
pyqt5 (for GUIs)

### Overview ###
This repository also has python code that allows matplotlib graphs to be added as a plugin to the QT Designer of the pyqt5 package.
For installing the Qt Designer matplotlib plugin see the readme in the QtDesignerPlugin/designer directory.  

The main electroophysiology analysis functions are in the BrainDataAnalysis folder. The non-GUI basked visualization functions are in the ploting_functions.py file.

In order to use the mne package (which has very nice LFP analysis functions) do the following so that you will be able to wrap raw data into mne appropriate classes:
The code in MNE_extraCode/mne directory of the repository should be added to the mne package (follow the same directory structure). 
If you do not want to fully overide existing scripts in the mne package check out the extra lines added in the equivalent scripts in this repository (marked in the script as #added).
This adds an extra input class allowing for arbitrary arrays to be memapped from the hard disk onto an mne Raw data structure (for use of mne with very large arbitrary arrays that cannot be put onto memory).  

To run the code in the ShuttlingAnalysis/paper directory a specific (old) version of [Bonsai](https://bitbucket.org/horizongir/bonsai) is required. Contact [horizongir](https://bitbucket.org/horizongir)
for more information.

Any OpenCV code assume opencv3.

Any experiment sepcific code is stored in its own directory under the ExperimentsSpecifiCode folder. There you can also find examples of use of the different functions in the repo.

The GUIs folder has code that allows visualization of data through guis. If you are using pycharm the DataFrameViewer is irrelevant because pycharm can do that natively. For any other IDE you can use that code to view a pandas dataframe.
There are also guis to view time frequency spectra over trials and to clean up kilosort results. The ploting_functions.py in BrainDataAnalysis also has some functions that allow a gui like previewing of 2d and 3d data (time or frequency series over trials).

The IO folder holds some function to turn into numpy arrays different data formats.

The Layouts folder holds the functions that generate specific probe maps (electrode positions, wirings and sometimes connectivities).

### How to use ###
If you want to use the code on a specific data set make another folder in the ExperimentSpecificCode.

If you develop some functions that are generic enough to be used for analysis other than your own then either add them to an existic folder (preferable) or if not possible make one yourself.

