"""
Classes to define python data structures and load data form files
"""

import sys, os, glob
from datetime import datetime
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
from nexusformat.nexus import nxload

try:
    from lmfit.models import GaussianModel, VoigtModel, LinearModel
    from . import peakfit_lmfit as peakfit
except ImportError:
    print('LMFit not available, using scipy.optimize.curve_fit instead')
    from scipy.optimize import curve_fit
    from . import peakfit_curvefit as peakfit

from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_nexus as fn
from . import nexus_config as config
from .scanplot import scanplot


class Experiment:
    def __init__(self, exp_directory='.', directories=[], title=''):
        self.path = [exp_directory] + directories
        # add current directory
        if '.' not in self.path:
            self.path += ['.']
        self.title = title

    def lastscan(self):
        """
        Get the latest scan number from the current experiment directory (self.path[0])
        """

        if os.path.isdir(self.path[0]) == False:
            print("I can't find the directory: {}".format(self.path[0]))
            return None

        # Get all data files in folder
        ls = glob.glob('%s/*.nxs' % (self.path[0]))
        ls = np.sort(ls)

        if len(ls) < 1:
            print("No files in directory: {}".format(self.path[0]))
            return None

        newest = ls[-1]  # file with highest number
        # newest = max(ls, key=os.path.getctime) # file created last
        num = fn.scanfile2number(newest)
        return num

    def allscanfiles(self):
        """
        Return list of all scan files in the data directories
        """
        filelist = []
        for directory in self.path:
            filelist += glob.glob('%s/*.nxs' % (directory))
        filelist = np.sort(filelist)
        return filelist

    def allscannumbers(self):
        """
        Return a list of all scan numbers in the data directories
        """
        filelist = self.allscanfiles()
        return [fn.scanfile2number(file) for file in filelist]

    def getfile(self, scan_number):
        if scan_number < 1:
            scan_number = self.lastscan() + scan_number

        filename = ''
        for directory in self.path:
            filename = os.path.join(directory, '%d.nxs' % scan_number)
            if os.path.isfile(filename): break
        return filename

    def loadnexus(self, scan_number=0, filename=None):
        if filename is None:
            filename = self.getfile(scan_number)
        if os.path.isfile(filename):
            return nxload(filename)
        else:
            print('Scan does not exist: %s' % filename)
            return None

    def loadscan(self, scan_number=0, filename=None):
        if filename is None:
            filename = self.getfile(scan_number)
        if os.path.isfile(filename):
            return Scan(filename, self)
        else:
            print('Scan does not exist: %s' % filename)
            return None

    def printscan(self, scan_number=0, filename=None):
        scan = self.loadscan(scan_number, filename)
        if scan is not None:
            print(scan.info())
            print('\n'.join([fmt(scan.nexus) for fmt in config._nxformat]))

    def loadscans(self, scan_numbers, variable_address=[]):
        """
        Return multi-scan object
        """
        scanlist = [self.loadscan(scan) for scan in scan_numbers]
        return MultiScans(scanlist, variable_address)


class Scan:
    """
    Scan Class, read data from Nexus files
        d = Scan('file/directory/12345.nxs')

    Sub-class:
        d.nexus     complete nexus tree, e.g. d.nexus['entry1/instrument']
        d.metadata  metadata values from nexus['entry1/before_scan']
        d.experiment link to parent experiment class

    Functions:
        d.tree()    prints the nexus tree
        d.title()   returns formated title
        d.autox()   returns name of default scan axes
        d.autoy()   returns name of default scan signal
        d.getx('eta')   returns array of scan axes
        d.gety('sum')   returns array of scan signal
        d.plot()    generates plot
        d.fit()     fit a peak

    Parameters:
        d.filename
        d.cmd
    """

    def __init__(self, scan_file, experiment=None):
        self.filename = scan_file
        self.experiment = experiment
        self.nexus = nxload(scan_file)
        self.cmd = str(self.nexus[config._NX_scan_command])

        keys = self.nexus[config._NX_measurement].keys()
        self.nx_address = {}
        for key in keys:
            setattr(self, key, np.array(self.nexus[config._NX_measurement][key]))
            self.nx_address[key] = self.nexus[config._NX_measurement][key].nxpath
        self.metadata = fn.MetaData(self.nexus, address=config._NX_metadata)

    def info(self, variable_address=[]):
        """
        return simple scan information
        """
        time = self.time().strftime('%Y-%m-%d %H:%M')
        number = self.scan_number()
        values = ' '.join([self.get_string(variable) for variable in variable_address])
        return '%s   #%s   %s  %s' % (time, number, values, self.cmd)

    def scan_number(self):
        return fn.scanfile2number(self.filename)

    def title(self):
        """
        Returns a formated title
        """
        if self.experiment is None:
            exp_title = ''
        else:
            exp_title = self.experiment.title

        return '%s\n%s\n%s' % (exp_title, self.filename, self.cmd)

    def time(self):
        """
        Returns the start time as datetime object
        """
        starttime = str(self.nexus[config._NX_starttime])
        #return datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%S.%f%z')
        return parser.parse(starttime)

    def duration(self):
        """
        Returns the scan duration in datatime.timedelta
        """
        #starttime = datetime.strptime(str(self.nexus[config._NX_starttime]), '%Y-%m-%dT%H:%M:%S.%f%z')
        #endtime = datetime.strptime(str(self.nexus[config._NX_endtime]), '%Y-%m-%dT%H:%M:%S.%f%z')
        starttime = parser.parse(str(self.nexus[config._NX_starttime]))
        endtime = parser.parse(str(self.nexus[config._NX_endtime]))
        return endtime - starttime

    def tree(self):
        print(self.title())
        print(self.nexus.tree)

    def search(self, name, whole_word=False, case_sensitive=False):
        """
        Search nexus tree for name, returns list of nexus addresses
        """
        return fn.nexus_search(self.nexus, name, whole_word, case_sensitive)

    def plot(self, xname=None, yname=None, yerrors=None):
        """
        Automatic plotting of scan
        """
        # self.nexus.entry1.measurement.plot()
        if xname is None:
            xname = self.autox()
        if yname is None:
            yname = self.autoy()
        xvals = self.getx(xname)
        yvals = self.gety(yname)
        scanplot(xvals, yvals, yerrors, self.title(), xname, yname, None)
        plt.show()

    def fit(self, xname=None, yname=None, yerrors=None, fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """
        if xname is None:
            xname = self.autox()
        if yname is None:
            yname = self.autoy()

        xvals = self.getx(xname)
        yvals = self.gety(yname)
        out = peakfit.peakfit(xvals, yvals)

        if print_result:
            print(self.title())
            print(out.fit_report())
        if plot_result:
            fig, grid = out.plot()
            plt.suptitle(self.title(), fontsize=12)
            plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax2.set_xlabel(xname)
            ax2.set_ylabel(yname)
            plt.show()
        return out

    def autox(self):
        return self.nexus[config._NX_measurement].axes

    def autoy(self):
        return self.nexus[config._NX_measurement].signal

    def getx(self, xname=None):
        if xname is None:
            xname = self.autox()
        return np.array(self.nexus[config._NX_measurement][xname])

    def gety(self, yname=None):
        if yname is None:
            yname = self.autoy()
        return np.array(self.nexus[config._NX_measurement][yname])

    def get_array(self, address):
        return np.array(self.nexus[address])

    def get_value(self, address):
        try:
            value = np.mean(self.nexus[address])
        except KeyError:
            value = np.nan
        return value

    def get_string(self, address):
        try:
            name = address.split('/')[-1]
            value = np.mean(self.nexus[address])
            output = '%s : %8.5g' % (name, value)
        except KeyError:
            output = ''
        return output

    def __call__(self, address):
        return self.nexus[address]

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return MultiScans([self, addee])


class MultiScans:
    """
    Add multiple scans together
    """

    def __init__(self, list_of_scans, variable_address=[]):
        self.scan_list = list_of_scans
        self.variable_address = variable_address
        self.update_variable()

    def update_variable(self):
        self.variable_names = [address.split('/')[-1] for address in self.variable_address]

        values = np.zeros([len(self.scan_list), len(self.variable_address)])
        for s, scan in enumerate(self.scan_list):
            for v, variable in enumerate(self.variable_address):
                values[s, v] = scan.get_value(variable)
        self.variable_values = values

    def add_variable(self, variable_address):
        self.variable_address += [variable_address]
        self.update_variable()

    def info(self):
        """
        return string of scan info
        """
        return '\n'.join([scan.info(self.variable_address) for scan in self.scan_list])

    def __add__(self, addee):
        return MultiScans(self.scan_list + [addee], self.variable_address)


