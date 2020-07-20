"""
Nexus Config file
Defines the locations of specific items in the nexus files
"""

import os
import glob
import numpy as np
import json
from matplotlib.pyplot import get_cmap
from matplotlib.colors import to_hex

__version__ = "0.1.0"
__date__ = "15/07/20"

# Config File directory
_config_directory = os.path.abspath(os.path.dirname(__file__))  # same directory as this file
_config_directory = os.path.join(_config_directory, 'config_files')
_default_config_file = os.path.join(_config_directory, 'default_config.json')
_default_nexus_file = os.path.join(_config_directory, 'example.nxs')
_default_nexus_format = '%d.nxs'


def get_config_files():
    """Return list of available config files"""
    return glob.glob('%s/*.json' % _config_directory)


class Config:
    """
    Config Class, Specify the config file
        c = Config()

    The Config class is used to store default configurations for each beamline or for individual experiments.
    Beamline default files have config.experiment_config['working_directory'] = None and when they are loaded in
    an Experiment class, the experiment parts are not changed.

    Sub-class:
        c.beamline_config       options that stay the same between experiments on the same beamline. dict with keys:
            'filename'
            'name'
            'description'
            'example_nexus'
            'colormap_name'
            'nx_name_format'
            'nx_scan_command'
            'nx_measurement'
            'nx_metadata'
            'nx_starttime'
            'nx_endtime'
            'nx_normalisation_addresses
            'normalisation_format'
            'format_specifiers'

        c.experiment_config     options that change during an experiment. dict with keys:
            'working_directory'
            'experiment_directories'
            'experiment_title'

    Functions:


    Parameters:
    """
    beamline_config = {
        'filename': _default_config_file,
        'name': 'default',
        'description': '',
        'example_nexus': _default_nexus_file,
        'colormap_name': 'rainbow',
        'nx_name_format': _default_nexus_format,
        'nx_scan_command': 'entry1/scan_command',
        'nx_measurement': 'entry1/measurement',
        'nx_metadata': 'entry1/before_scan',
        'nx_starttime': 'entry1/start_time',
        'nx_endtime': 'entry1/end_time',
        'nx_normalisation_addresses': [],
        'normalisation_format': 'x',
        'format_specifiers': {
            'format': [],
            'address': [],
        }
    }
    experiment_config = {
        'working_directory': None,
        'experiment_directories': [],
        'experiment_title': '',
    }

    def __init__(self, filename=None):
        self.filename = filename
        if filename is not None:
            self.load_json(filename)

    def __repr__(self):
        name = self.beamline_config['name']
        desc = self.experiment_config['experiment_title']
        file = self.filename
        return 'Config( %s : %s : %s)' % (name, desc, file)

    def info(self):
        """Display config values"""
        out = '----------------------------------\n'
        out += '      Config. File: %s\n\n' % self.filename
        bc_names = ['filename', 'name', 'description', 'example_nexus', 'colormap_name', 'nx_name_format',
                    'nx_scan_command', 'nx_measurement', 'nx_metadata', 'nx_starttime', 'nx_endtime',
                    'nx_normalisation_addresses', 'normalisation_format']
        ec_names = ['working_directory', 'experiment_directories', 'experiment_title']

        out += '  Beamline Config.:\n'
        for name in bc_names:
            out += '%30s : %s\n' % (name, self.beamline_config[name])
        out += '%30s :\n' % 'Format specifiers'
        for n in range(len(self.beamline_config['format_specifiers']['format'])):
            fmt = self.beamline_config['format_specifiers']['format'][n]
            addrs = self.beamline_config['format_specifiers']['address'][n]
            out += '    %40s : %s\n' % (fmt, addrs)
        out += 'Experiment Config.:\n'
        for name in ec_names:
            out += '%30s : %s\n' % (name, self.experiment_config[name])
        out += '----------------------------------\n'
        return out

    # --------------------- Add Parameters ------------------------------

    def add_parameter(self, format, address_list):
        """Add new parameter to format specifiers list"""
        self.beamline_config['format_specifiers']['format'] += [format]
        self.beamline_config['format_specifiers']['address'] += [address_list]

    def add_metaformat(self, metaformat):
        """Add new parameter from MetaFormat"""
        self.beamline_config['format_specifiers']['format'] += [metaformat.format]
        self.beamline_config['format_specifiers']['address'] += [metaformat.addresses]

    def get_parameters(self):
        """Return MetaFormat objects of format specifiers"""
        out = []
        fs = self.beamline_config['format_specifiers']
        for format, address in zip(fs['format'], fs['address']):
            out += [MetaFormat(format, address)]
        return out

    def get_parameter_strings(self, nexus):
        """ Return strings generated by MetaFormat"""
        param_list = self.get_parameters()
        return [fmt(nexus) for fmt in param_list]

    def check_nexus(self, nexus):
        """Check nexus (.nxs) file conforms to items in config"""
        bconfig = self.beamline_config
        try:
            _ = nexus[bconfig['nx_scan_command']]
            _ = nexus[bconfig['nx_measurement']]
            _ = nexus[bconfig['nx_metadata']]
            _ = nexus[bconfig['nx_starttime']]
            _ = nexus[bconfig['nx_endtime']]
            for addr in bconfig['nx_normalisation_addresses']:
                _ = nexus[addr]
        except Exception:
            print('Nexus file does not contain required data')
            raise
        return True

    # --------------------- Main Fields ------------------------------

    def name(self, field=None):
        """Set/ return name field"""
        if field is None:
            return self.beamline_config['name']
        self.beamline_config['name'] = field

    def description(self, field=None):
        """Set/ return description field"""
        if field is None:
            return self.beamline_config['description']
        self.beamline_config['description'] = field

    def nexus_example(self, field=None):
        """Set/ return example_nexus field"""
        if field is None:
            return self.beamline_config['example_nexus']
        self.beamline_config['example_nexus'] = field

    def nexus_name_format(self, field=None):
        """Set/ return nexus_name_format field"""
        if field is None:
            return self.beamline_config['nx_name_format']
        self.beamline_config['nx_name_format'] = field

    def scan_command(self, field=None):
        """Set/ return scan command field"""
        if field is None:
            return self.beamline_config['nx_scan_command']
        self.beamline_config['nx_scan_command'] = field

    def measurement(self, field=None):
        """Set/ return measurement field"""
        if field is None:
            return self.beamline_config['nx_measurement']
        self.beamline_config['nx_measurement'] = field

    def metadata(self, field=None):
        """Set/ return metadata field"""
        if field is None:
            return self.beamline_config['nx_metadata']
        self.beamline_config['nx_metadata'] = field

    def starttime(self, field=None):
        """Set/ return start time field"""
        if field is None:
            return self.beamline_config['nx_starttime']
        self.beamline_config['nx_starttime'] = field

    def endtime(self, field=None):
        """Set/ return end time field"""
        if field is None:
            return self.beamline_config['nx_endtime']
        self.beamline_config['nx_endtime'] = field

    # --------------------- Colormaps ----------------------------

    def colors(self, n_colors, return_hex=False):
        """Return a list of colors using the beamline_config['colormap']"""
        n_vals = np.linspace(0, 1, n_colors)
        cmap = get_cmap(self.beamline_config['colormap_name'])
        if return_hex:
            return [to_hex(c) for c in cmap(n_vals)]
        return cmap(n_vals)

    # ------------------- Normalisation --------------------------

    def normalisation(self, format=None, address_list=None):
        """Set/ return  normalisation"""
        if format is None:
            return self.beamline_config['normalisation_format'], self.beamline_config['nx_normalisation_addresses']
        self.beamline_config['normalisation_format'] = format
        self.beamline_config['nx_normalisation_addresses'] = address_list

    def normalisation_function(self):
        """Return the normalisation function defined from format and addresses"""
        format = self.beamline_config['normalisation_format']
        address_list = self.beamline_config['nx_normalisation_addresses']

        return NormaliseFormat(format, address_list)

    def normalise_value(self, value, nexus):
        """Normalise a value using defined normalisation"""
        normfun = self.normalisation_function()
        return normfun(value, nexus)

    # --------------------- Save/ Load ---------------------------

    def save_json(self, filename=None):
        """Save config file as .json format"""
        if filename is None:
            filename = self.filename
        if 'json' not in filename:
            filename = filename + '.json'
        full_dict = {'beamline_config': self.beamline_config, 'experiment_config': self.experiment_config}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_dict, f, ensure_ascii=False, indent=4)
        self.filename = filename
        print('Config file written to %s' % filename)

    def load_json(self, filename=None):
        """Load config file from json file"""
        if filename is None:
            filename = self.filename
        with open(filename) as f:
            full_dict = json.load(f)
        self.filename = filename
        self.beamline_config.update(full_dict['beamline_config'])
        self.experiment_config.update(full_dict['experiment_config'])
        print('Loaded Config from %s' % filename)


class MetaFormat:
    """
    Specify standard ways of displaying items in nexus
    """
    def __init__(self, format_specifier, addresses=[]):
        self.format = format_specifier
        self.addresses = addresses

    def get_address_values(self, nexus):
        values = []
        for address in self.addresses:
            try:
                values += [nexus[address]]
            except KeyError:
                values += [np.nan]
        return values

    def get_address_string(self, nexus):
        values = self.get_address_values(nexus)
        return self.format % tuple(values)

    def __call__(self, nexus):
        return self.get_address_string(nexus)


class NormaliseFormat:
    """
    Normalisation class
    """

    def __init__(self, format_specifier, addresses=[]):
        # format = 'x/Transmission/time/ic1monitor'
        self.format = format_specifier
        self.addresses = addresses

        self.name_dict = {}
        for address in self.addresses:
            name = address.split('/')[-1]
            self.name_dict[name] = address

    def fill_name(self, nexus):
        values_dict = {}
        for address in self.addresses:
            name = address.split('/')[-1]
            values_dict[name] = nexus[self.name_dict[name]]
        return values_dict

    def normalise(self, value, nexus):
        values_dict = self.fill_name(nexus)
        values_dict['x'] = value
        print(self.format)
        print(values_dict)
        print(eval(self.format, globals(), values_dict))
        return eval(self.format, globals(), values_dict)

    def __call__(self, value, nexus):
        return self.normalise(value, nexus)



