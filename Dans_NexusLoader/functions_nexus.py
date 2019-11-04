"""
Additional useful functions to use on Nexus files
"""

import sys, os
import numpy as np


class MetaData:
    """
    Metadata Class
    Stores metadata as float attributes of class
    """

    def __init__(self, nexus, address):
        meta = nexus[address]
        self.nx_address = {}
        for collection in meta.keys():
            if 'NXfield' in str(type(meta[collection])):
                setattr(self, collection, float(meta[collection]))
                self.nx_address[collection] = meta[collection].nxpath
            else:
                for key in meta[collection].keys():
                    setattr(self, key, float(meta[collection][key]))
                    self.nx_address[key] = meta[collection][key].nxpath


def scanfile2number(filename):
    return np.abs(np.int(os.path.split(filename)[-1][-10:-4]))


def nexus_search(nexus, find_string, whole_word=False, case_sensitive=False):
    """
    Search a nexus tree for a specific name, returns list of addresses
    """

    output_addresses = []
    try:
        for collection in nexus.keys():
            output_addresses += nexus_search(nexus[collection], find_string, whole_word, case_sensitive)
    except AttributeError:
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

