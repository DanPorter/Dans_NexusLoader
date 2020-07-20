"""
Additional useful functions to use on Nexus files
"""

import sys, os
import numpy as np
import h5py  # backup for nxload
from imageio import imread  # read Tiff images
from nexusformat.nexus import nxload
#from nexusformat.nexus.tree import NexusError


class NexusWrapper(object):
    """
    Generic wrapper for nexus files
    """
    _max_image_stack_elements = 20e6  # Max image stack, will not make stack larger than this

    def __init__(self, nexus_filename):
        self.filename = nexus_filename
        self.nexus = loadnexus(nexus_filename)
        self.image_address = None

    def tree(self):
        """
        Return the nexus tree structure as a string
        """
        return nexus_tree(self.nexus)

    def search(self, name, whole_word=False, case_sensitive=False):
        """
        Search nexus tree for name, returns list of nexus addresses
        """
        return nexus_search(self.nexus, name, whole_word, case_sensitive)

    def all_addresses(self):
        """
        Returns all nexus addresses
        """
        return nexus_addresses(self.nexus)

    def has_image_data(self):
        """
        Returns true if nexus contains links to image data
        """
        if self.find_image_address():
            return True
        return False

    def find_image_address(self):
        """
        Returns nexus address with image data
        """
        if self.image_address is None:
            path, file = os.path.split(self.filename)
            self.image_address = nexus_image_address(self.nexus, path)
        return self.image_address

    def image_filenames(self, address=None):
        """
        Returns list of image filenames from nexus file, if available
            addresses = image_address('/entry1/instrument/pil3_100k/image_data')
                * uses the given nexus address
            addresses = image_address()
                * searches automatically for an image address
        """
        if address is None:
            address = self.find_image_address()
        nx_addresses = self.nexus[address]

        path, nexusfilename = os.path.split(self.filename)
        return [os.path.join(path, str(address)) for address in nx_addresses]

    def get_image(self, index=None, address=None):
        """
        Return detector image as numpy array
        :param index: int : return nth 2D image
        :param address: str or None : image address, None will search for address
        :return: numpy array [n,m]
        """

        filenames = self.image_filenames(address)
        if index is None:
            index = int(len(filenames)//2)
        return imread(filenames[index])

    def get_images(self, address=None):
        """
        Return detector image stack as 3D numpy array
        :param address: str or None : image address, None will search for address
        :return: numpy array [n,m,image]
        """
        # image filenames
        path, nexusfilename = self.filename.split()
        filenames = self.image_filenames(address)

        # Load 1st file to determine image size
        file = os.path.join(path, filenames[0])
        image = loadimage(file)

        # Check max stack size
        max_index = len(filenames)
        if np.prod(image.shape)*max_index > self._max_image_stack_elements:
            max_index = int(self._max_image_stack_elements // np.prod(image.shape))

        image_volume = np.zeros(image.shape + (max_index,))
        image_volume[:, :, 0] = image
        for index in range(1, max_index):
            file = os.path.join(path, filenames[index])
            image_volume[:, :, index] = loadimage(file)
        return image_volume

    def get_array(self, address):
        return np.array(self.nexus[address])

    def get_value(self, address):
        try:
            value = np.mean(self.nexus[address])
        except KeyError:
            value = np.nan
        return value

    def get_string(self, address, name=None):
        if name is None:
            name = address.split('/')[-1]
        try:
            value = np.mean(self.nexus[address])
            output = '%s : %8.5g' % (name, value)
        except TypeError:
            output = '%s : %s' % (name, self.nexus[address])
        except KeyError:
            output = '%s : Not Available' % name
        return output

    def get_position_string(self, address=None, name=None):
        """
        Return list of position strings for each point in the scan
        :param address: nexus address
        :param name: name
        :return: list of strings
        """
        if name is None:
            name = address.split('/')[-1]
        values = self.get_array(address)
        return ['%s = %s [%d/%d]' % (name, value, n, len(values)) for n, value in enumerate(values)]

    def __call__(self, address):
        return self.nexus[address]


class MetaData:
    """
    Metadata Class
    Stores metadata as float attributes of class
        Functions:
            self.address(name) - returns nexus address of parameter, or None
            self.value(name) - returns float value of parameter, or None
            self.names() - returns a list of parameter names
    """

    def __init__(self, nexus, address):
        meta = nexus[address]
        self._nexus = nexus
        self._nexus_metadata_address = address
        addresses = nexus_addresses(meta)
        self._nx_names = [os.path.basename(adr) for adr in addresses]
        self._nx_addresses = [meta[adr].nxpath for adr in addresses]
        self._nx_values = [float(meta[adr]) for adr in addresses]

        for name, value in zip(self._nx_names, self._nx_values):
            setattr(self, name, value)

    def address(self, item):
        """Return nexus address of item"""
        if item not in self._nx_names:
            return None
        idx = self._nx_names.index(item)
        return self._nx_addresses[idx]

    def value(self, item):
        """Return nexus address of item"""
        if item not in self._nx_names:
            return None
        idx = self._nx_names.index(item)
        return self._nx_values[idx]

    def names(self):
        return self._nx_names

    def addresses(self):
        return self._nx_addresses

    def values(self):
        return self._nx_values

    def __repr__(self):
        out = 'Metadata:\n'
        for name, value in zip(self._nx_names, self._nx_values):
            out += '%30s : %s\n' % (name, value)
        return out


def loadnexus(filename, use_nexusformat=True):
    """wrapper for nexusformat.nexus.nxload, but falls back on h5py.File"""
    if use_nexusformat:
        return nxload(filename)
    return h5py.File(filename, 'r')


def loadimage(filename):
    """wrapper for imageio.imread to read image files such as *.tiff"""
    return imread(filename)


def scanfile2number(filename):
    return np.abs(np.int(os.path.split(filename)[-1][-10:-4]))


def nexus_addresses(nexus):
    """Returns complete nexus tree as list of addresses"""
    # Recursive function
    output_addresses = []
    try:
        for collection in nexus.keys():
            output_addresses += nexus_addresses(nexus[collection])
    except Exception:
        output_addresses += [nexus.nxpath]
    return output_addresses


def nexus_tree(nexus, debug=False):
    """
    Returns string of the complete nexus tree
        n = loadnexus('12345.nxs')
        ss = nexus_tree(n)
        print(ss)
    """

    # Recursive function
    output_string = ''
    try:
        # NXCollection
        if debug: print(nexus.keys())
        for collection in nexus.keys():
            if debug: print('Tree: %s' % collection)
            treestr = nexus_tree(nexus[collection], debug)
            output_string += '%s' % treestr
    except Exception:
        # NXfield
        if debug: print('   Field: %s' % nexus)
        if nexus.size > 1:
            field_string = str(nexus.shape)
        else:
            field_string = str(nexus)
        output_string += '%s: %s\n' % (nexus.nxpath, field_string)
    return output_string


def nexus_search(nexus, find_string, whole_word=False, case_sensitive=False):
    """
    Search a nexus tree for a specific name, returns list of addresses
    """

    # Recursive function
    output_addresses = []
    try:
        for collection in nexus.keys():
            output_addresses += nexus_search(nexus[collection], find_string, whole_word, case_sensitive)
    except Exception:
        path = nexus.nxpath
        name = nexus.nxname
        if (
                whole_word and case_sensitive and find_string == name or
                whole_word and not case_sensitive and find_string.lower() == name.lower() or
                not whole_word and case_sensitive and find_string in name or
                not whole_word and not case_sensitive and find_string.lower() in name.lower()
        ):
            output_addresses += [path]
    return output_addresses


def nexus_image_address(nexus, filepath):
    """Finds an address in the nexus file with image file addresses"""
    addresses = nexus_addresses(nexus)
    for address in addresses:
        if nexus[address].size == 1: continue
        file = str(nexus[address][0])
        file = os.path.join(filepath, file)
        if os.path.isfile(file):
            return address
    return None

