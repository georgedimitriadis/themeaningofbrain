# README #


### What is this repository for? ###

This is python code that allows for animal behaviour analysis and animal electrophysiology (ECoG and spikes) analysis.
This code is meant to run on python 3 although no python 3 specific code has been implemented (but see bellow for python 3 specific opencv).

### How do I get set up? ###
You need to have all the required libraries that the different scripts import. Main libraries used are:  
numpy  
scipy  
matplotlib  
pandas  
mne  
opencv
pyqt4 (for GUIs)

This repository also has python code that allows matplotlib graphs to be added as a plugin to the QT Designer of the pyqt4 package.
For installing the Qt Designer matplotlib plugin see the readme in the QtDesignerPlugin/designer directory.  

Electrophysiology analysis is done through a combination of islab developed code and the use of the [mne_python](http://martinos.org/mne/stable/mne-python.html) package.
To run the extra mne relevant analysis the code in MNE_extraCode/mne directory of this repository should be added to the mne package (follow the same directory structure). 
If you do not want to fully overide existing scripts in the mne package check out the extra lines added in the equivalent scripts in this repository (marked in the script as #added).
This adds an extra input class allowing for arbitrary arrays to be memapped from the hard disk onto an mne Raw data structure (for use of mne with very large arbitrary arrays that cannot be put onto memory).  

To run the code in the ShuttlingAnalysis/paper directory a specific (old) version of [Bonsai](https://bitbucket.org/horizongir/bonsai) is required. Contact [horizongir](https://bitbucket.org/horizongir)
for more information.

Opencv at the time of writting this (04/2015) only offers python 3 wrappings through the beta realease of opencv 3. To make matters worse, the pre compiled opencv 3 comes only with python 2 so for python 3 we need to build opencv 3 ourselves.
To compile (with python 3) and install opencv 3 do the following steps (for windows): 

1) Download [opencv 3](https://github.com/Itseez/opencv) from github. You can also make a local clone of the repository (see a large number of websites describing how to do that with git). 

2) The directory structure needs to have a special form for this to compile succesfully (hopefully future versions will not demand this). 
In the main opencv folder (put that anywhere in the system) there needs to be two subfolders.
One will have all the folders and files as they appear on the github repo and the other is where the build folders and files will be placed during the build. So for example do the following:
<Wherever on the system you put opencv>\opencv\sources\<stuff inside the opencv-master folder downloaded as zip file from the git> and
<Wherever on the system you put opencv>\opencv\build.
For the remaining folder I will assume that opencv is the name of the main folder with subfolders sources and build. 

3) Donwload and install [CMake](http://www.cmake.org/). 

4) Run cmake-gui and do the following (to generate the opencv solution): 

i) In the "Where is the source code:" put the opencv\sources directory (if you have generated the folder structure correctly it will have inside it a CMakeList.txt file). 

ii) In the "Where to build the binaries:" put the opencv\build directory. 

iii) Check Grouped and Advanced (helps a bit but not a must). 

iv) Press Configure. It will ask you for your compiler. It won't work with VS 2015 so the latest compiler you can set is VS 2013 (i.e. VS12). 

Also make sure you choose the compiler with the right platform (i.e. x86 or x64) according to how you want to build opencv later on. 

v) Go to the PYTHON2 and PYTHON3 entries and make sure the directories are set properly: Set the PYTHONX_INCLUDE_DIR. Also the PYTHONX_LIBRARY with the Pythonx/libs/pythonxx.lib file in your system (e.g. Python3.4/libs/python34.lib). 
You must also have a numpy package installed for any python version you want to build with opencv (and set its path in PYTHONX_NUMPY_INCLUDE_DIRS). 
If you do not have Python 2 delete everything from the PYTHON2 entry. 

5) Press Configure again. After a bit all the red entries should become white. You might (or might not) get a large number of Warnings. It shouldn't matter. 

6) Press Generate. That's it with cmake. 

7) Go to opencv/build and start the OpenCV.sln solution (I am assuming you have a Visual Studio installation that will open the solution automatically). 

8) Set the Configuration to Release and whatever platform you want (i.e. x86 or x64). 

9) Build the solution. Hopefully you will get no errors. 

10) Check into the following directory: opencv\build\lib\Release. There should be a file called cv2.pyd. 

11) Either copy that cv2.pyd file in the python3\lib\site-packages directory or add the opencv\build\lib/Release to the PYTHONPATH directory if you are using only one python version. 

12) Add a new environment variable with Name: OPENCV_DIR and Value the opencv\build directory (i.e. OPENCV_DIR = <Wherever opencv is>\<opencv dir name>\build). 

13) Add to the PATH environment variable the following: %OPENCV_DIR%\bin\Release. That will add the directory where all the opencv libraries have been built to the system's path. 

14) Now in a python console try import cv2. Hopefully you won't get an error.
