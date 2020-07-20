"""
Example Script for testing tkinter gui
"""
import matplotlib
matplotlib.use('TkAgg')

import os
import json
import Dans_NexusLoader as dnex
from Dans_NexusLoader.tkgui.basic_widgets import SelectionBox
from Dans_NexusLoader.tkgui.config_gui import ConfigGui
from Dans_NexusLoader.tkgui.image_gui import ImageGui
from Dans_NexusLoader.tkgui.experiment_gui import ExperimentGui

#exp = dnex.Experiment([r'C:\Users\dgpor\Dropbox\Python\ExamplePeaks', r'C:\Users\grp66007\Dropbox\Python\ExamplePeaks'])
print(os.path.dirname(__file__))
exp = dnex.Experiment(working_directory='.')
exp.load_config()
#print(exp.config.info())
#print(exp.path)
#d = exp.loadscan(794940)
#print(d.nexus['/entry1/sample/transformations/phi'])
#print(d.tree())

print('----------------------------')

#exp.printscan(794940)

print('----------------------------')


#ConfigGui(r'Dans_NexusLoader/config_files/i16_config.json')
#ConfigGui(exp.config.filename)


#filenames = d.image_filenames()
#ImageGui(filenames, d.get_position_string(x_address), title=d.title())

ExperimentGui(exp)

#print(exp.allscannumbers())

from Dans_NexusLoader.tkgui.scaninfo_gui import ScanInfo

#ScanInfo(exp, [794935, 794940, 810002])

print('Finished')