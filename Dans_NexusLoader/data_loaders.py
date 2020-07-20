"""
Classes to define python data structures and load data form files
"""

import sys, os, glob
from datetime import datetime
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt

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
from . import nexus_config
from .scanplot import plotline, scanplot, scansplot, imageplot


__version__ = "0.1.0"
__date__ = "15/07/20"


class Experiment:
    """
    Experiment Class
    Holder for experiment directories (location of data files) and working director (where to save files).
    Also contains a Config object that specifies beamline and experiment configuration options.
    The Config option also stores the experiment and working directory so previous work can be saved.

    exp = Experiment('E:\I16_Data\mm24570-1')
    exp = Experiment(directories=['E:\I16_Data\mm24570-1', 'E:\I16_Data\mm24570-1'])
    """

    _config_filename = 'NexusLoader_config.json'

    def __init__(self, experiment_directories=[], working_directory='.', title=None, config_file=None):
        self.working_directory = working_directory
        self.path = list(np.asarray(experiment_directories, dtype=str).reshape(-1))
        # add current directory
        if '.' not in self.path:
            self.path += ['.']
        if title is None:
            self.title = os.path.basename(self.path[0])
        else:
            self.title = title

        print(os.path.dirname(__file__))

        # Get config
        self.config = nexus_config.Config()
        if config_file is not None:
            self.load_config(config_file)
        self.update_config()

    def info(self):
        """
        Return information str
        :return:
        """
        out = "     Working Directory: %s\n          Config. File: %s\nExperiment Directories:\n                   %s"
        expdir = '\n                   '.join(self.path)
        return out % (self.working_directory, self.config.filename, expdir)

    def load_config(self, config_file=None):
        """Load beamline config file"""
        if config_file is None:
            config_file = os.path.join(self.working_directory, self._config_filename)
        self.config.load_json(config_file)
        # Beamline default files have config.experiment_config['working_directory'] = None
        if self.config.experiment_config['working_directory'] is not None:
            # Previous experiment config file
            self.working_directory = self.config.experiment_config['working_directory']
            self.path = self.config.experiment_config['experiment_directories']
            self.title = self.config.experiment_config['experiment_title']
            self._config_filename = os.path.basename(config_file)

    def update_config(self):
        """Update experiment parts of the config file"""
        self.config.experiment_config['working_directory'] = self.working_directory
        self.config.experiment_config['experiment_directories'] = self.path
        self.config.experiment_config['experiment_title'] = self.title

    def save_config(self, save_location=None):
        """Save current settings as new config file in working directory"""
        if save_location is None:
            save_location = os.path.join(self.working_directory, self._config_filename)
        # Update config
        self.update_config()
        # Save config file
        self.config.save_json(save_location)

    def lastscan(self):
        """
        Get the latest scan number from the current experiment directory (self.path[0])
        Return None if no scans found.
        """

        if len(self.path) == 0:
            print('Please set experiment directory!')
            return None
        elif os.path.isdir(self.path[0]) == False:
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
            filelist += glob.glob('%s/*.nxs' % directory)
        filelist = np.sort(filelist)
        return filelist

    def allscannumbers(self):
        """
        Return a list of all scan numbers in the data directories
        """
        filelist = self.allscanfiles()
        return [fn.scanfile2number(file) for file in filelist]

    def getfile(self, scan_number):
        """
        Convert int scan number to file
        :param scan_number: int : scan number, scans < 1 will look for the latest scan
        :return: filename or '' if scan doesn't appear in directory
        """
        scan_number = np.asarray(scan_number, dtype=int).reshape(-1)
        scan_number = scan_number[0]
        if scan_number < 1:
            scan_number = self.lastscan() + scan_number

        filename = ''
        for directory in self.path:
            filename = os.path.join(directory, self.config.nexus_name_format() % scan_number)
            if os.path.isfile(filename): break
        return filename

    def loadnexus(self, scan_number=0, filename=None, use_nexusformat=True):
        """
        Load Nexus file for scan number or filename
        :param scan_number: int
        :param filename: str : scan filename
        :param use_nexusformat: True : If False, return h5py file
        :return: Nexus object
        """
        if filename is None:
            filename = self.getfile(scan_number)
        if os.path.isfile(filename):
            return fn.loadnexus(filename, use_nexusformat)
        else:
            print('Scan does not exist: %s' % filename)
            return None

    def loadscan(self, scan_number=0, filename=None):
        """
        Generate Scan object for given scan using either scan number or filename.
        :param scan_number: int
        :param filename: str : scan filename
        :return:
        """
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
            print('\n'.join(self.config.get_parameter_strings(scan.nexus)))

    def loadscans(self, scan_numbers, variable_address=[]):
        """
        Return multi-scan object
        """
        scanlist = [self.loadscan(scan) for scan in scan_numbers]
        return MultiScans(scanlist, variable_address)


class Scan(fn.NexusWrapper):
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
        d.filename      .nxs filename
        d.cmd           scan command
        d.scan_number   scan number
        d.measurement_keys  list of available measurement keys
        d.n_points      length of each measurement array
    """

    def __init__(self, scan_file, experiment=None):

        super(Scan, self).__init__(scan_file)  # defines filename and nexus
        self.experiment = experiment
        if experiment is None:
            self.config = nexus_config.Config()
        else:
            self.config = experiment.config
        # Check nexus file
        check = self.config.check_nexus(self.nexus)

        self.cmd = str(self.nexus[self.config.scan_command()])
        self.scan_number = fn.scanfile2number(self.filename)

        measurement = self.nexus[self.config.measurement()]
        self.measurement_keys = list(measurement.keys())
        self._nx_address = {}
        for name, item in measurement.items():
            setattr(self, name, np.asarray(item))
            self._nx_address[name] = item.nxpath
            self.n_points = item.size
        self.metadata = fn.MetaData(self.nexus, address=self.config.metadata())

    def info(self):
        """
        return scan information as str, includes data requested by config file
        """
        text_list = ['  Filename: %s' % self.filename,
                     '   Command: %s' % self.cmd,
                     '    X-Axis: %s' % self.autox(),
                     '    Y-Axis: %s' % self.autoy(),
                     'Start time: %s' % self.time().strftime('%Y-%m-%d %H:%M'),
                     '  Duration: %s' % self.duration()]
        # config file scan details
        text_list += self.get_parameter_strings()
        return '\n'.join(text_list)

    def info_line(self, variable_address=None):
        """
        Returns single line information with option extra arguments
        :param variable_address: list or nexus addresses
        :return: str
        """
        time = self.time().strftime('%Y-%m-%d %H:%M')
        number = self.scan_number
        if variable_address is None:
            values = ''
        else:
            values = ' '.join([self.get_string(variable) for variable in variable_address])
        return '%s   #%s   %s  %s' % (time, number, values, self.cmd)

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
        starttime = str(self.nexus[self.config.starttime()])
        #return datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%S.%f%z')
        return parser.parse(str(starttime))

    def duration(self):
        """
        Returns the scan duration in datatime.timedelta
        """
        #starttime = datetime.strptime(str(self.nexus[config._NX_starttime]), '%Y-%m-%dT%H:%M:%S.%f%z')
        #endtime = datetime.strptime(str(self.nexus[config._NX_endtime]), '%Y-%m-%dT%H:%M:%S.%f%z')
        starttime = parser.parse(str(self.nexus[self.config.starttime()]))
        endtime = parser.parse(str(self.nexus[self.config.endtime()]))
        return endtime - starttime

    def plot(self, xname=None, yname=None, yerrors=None, axis=None, show=True, **kwargs):
        """
        Automatic plotting of scan
        if axis is given, plots line on the given axis and line2D object is returned
        """
        # self.nexus.entry1.measurement.plot()
        if xname is None:
            xname = self.autox()
        if yname is None:
            yname = self.autoy()
        xvals = self.getx(xname)
        yvals = self.gety(yname)
        if axis is None:
            scanplot(xvals, yvals, yerrors, self.title(), xname, yname, None, **kwargs)
        else:
            lineobj = plotline(axis, xvals, yvals, yerrors, **kwargs)
            return lineobj
        if show:
            plt.show()

    def image_plot(self, index=None):
        """
        Automatic plotting of detector image
        """
        xname = self.autox()
        xvals = self.getx(xname)
        if index is None:
            index = int(len(xvals)//2)

        image = self.get_image(index)
        ttl = '%s\n%s = %s [%d/%d]' % (self.title(), xname, xvals[index], index, len(xvals))
        imageplot(image, ttl)
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

    def get_measurement(self, name):
        """Return array from nexus measurement group"""
        #return np.asarray(self.nexus[self.config.measurement()][name])
        return getattr(self, name)

    def autox(self):
        return self.nexus[self.config.measurement()].axes

    def autoy(self):
        return self.nexus[self.config.measurement()].signal

    def getx(self, xname=None):
        if xname is None:
            xname = self.autox()
        return self.get_measurement(xname)

    def gety(self, yname=None):
        if yname is None:
            yname = self.autoy()
        return self.get_measurement(yname)

    def geterror(self, name=None):
        if name is None:
            name = self.autoy()
        values = self.get_measurement(name)
        # Not done yet - need a method in config!
        return np.sqrt(values)

    def get_parameter_strings(self):
        """Get parameter strings from config"""
        return self.config.get_parameter_strings(self)

    def __repr__(self):
        """
        Default print behaviour
        :return:
        """
        time = self.time().strftime('%Y-%m-%d %H:%M')
        number = self.scan_number
        return '%s   #%s   %s' % (time, number, self.cmd)

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
        self.experiment = list_of_scans[0].experiment
        self.scan_numbers = [scan.scan_number for scan in list_of_scans]
        self.variable_address = variable_address
        self.variable_names = None
        self.variable_values = None
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
        return '\n'.join([scan.info_line(self.variable_address) for scan in self.scan_list])

    def title(self):
        """Generate MultiScan title"""
        ttl = 'MultiScans(%s)' % fg.numbers2string(self.scan_numbers)
        return ttl

    def colors(self):
        return self.experiment.config.colors(len(self.scan_list))

    def plot(self, xname=None, yname=None, axis=None, show=True, **kwargs):
        """
        Automatic plotting of scan
        if axis is given, plots line on the given axis and line2D object is returned
        """
        # Use 1st scan in list to define defaults
        scan = self.scan_list[0]
        if xname is None:
            xname = scan.autox()
        if yname is None:
            yname = scan.autoy()
        colors = self.colors()
        xvals = []
        yvals = []
        lineobj = []
        for n, scan in enumerate(self.scan_list):
            xvals += [scan.getx(xname)]
            yvals += [scan.gety(yname)]
            if axis:
                kwargs['c'] = colors[n]
                lineobj += plotline(axis, xvals[-1], yvals[-1], **kwargs)
        if axis:
            return lineobj

        ax = scansplot(xvals, yvals, colours=colors)
        if show:
            plt.show()
        else:
            return ax

    def __repr__(self):
        """
        Default print behaviour
        :return:
        """
        return 'MultiScans(%s)' % fg.numbers2string(self.scan_numbers)

    def __add__(self, addee):
        return MultiScans(self.scan_list + [addee], self.variable_address)


