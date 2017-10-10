# Installation information for the matplotlib Qt Designer plugin #

This is a Qt Designer plugin that adds matplotlib graphics to the designed QT gui. It requires pyqt4 and works with the Qt Designer that comes with the pyqt4 installation for Qt4 and python 3. 
Python 2 might or might not work (there is no python 3 specific code but this has not been tested with the python 2 version of pyqt).
For some reason this plugin will not work in a virtual environment (at least not one created in a windows machine, other OSes have not been tested).  
  
  
To get Qt Designer to see the plugin the following steps must be followed.  
  
  
1) In the pyqt4 directory (python_root/Lib/site-packages/PyQt4) there is the directory plugins/designer. Add the python and the widgets directories to this directory and also the plugins.pyw file
(the plugins.pyw and other plugin examples can be found in python_root/Lib/site-packages/PyQt4/examples/designer/plugins and can be copied over to the plugins/designer directory for the Qt Designer to see them).  
  
2) Add to your System Variables the following two variables  
PYTHONPATH with the path to the plugins/designer/widgets directory  
PYQTDESIGNERPATH with the path to the plugins/designer/python directory  
  
3) Run the Qt designer executable found in python_root/Lib/site-packages/PyQt4 and the matplotlib plugin should be under the Display Widgets group.