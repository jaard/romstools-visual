# Romstools-visual

This is a module providing some tools for visualizing output of the ROMS regional ocean model in python. (e.g. extracting slices and sections). The functions were adapted from the original ROMSTOOLS Matlab code by Penven et al., 2007 (https://doi.org/10.1016/j.envsoft.2007.07.004).

## Installation (Mac OSX)

1. Save the romstools_visual.py file to your disk
2. Open a terminal and type "open .bash_profile"
3. Append the folder you saved the file to your PYTHONPATH environment variable by adding the line
   "export PYTHONPATH=/Users/Jaard/Documents/Python:$PYTHONPATH"
4. Save .bash_profile
5. Write "import romstools_visual as rv" at the top of your Python script
