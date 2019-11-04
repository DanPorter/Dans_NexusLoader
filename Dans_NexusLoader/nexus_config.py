"""
Nexus Config file
Defines the locations of specific items in the nexus files
"""

import numpy as np


class MetaFormat:
    """
    Specify standard ways of displaying items in nexus
    """
    def __init__(self, format_specifier, addresses=[]):
        self.format = format_specifier
        self.addresses = addresses

    def __call__(self, nexus):
        values = []
        for address in self.addresses:
            try:
                values += [nexus[address]]
            except:
                values += [np.nan]
        return self.format % tuple(values)


# Nexus Addresses
_NX_scan_command = 'entry1/scan_command'
_NX_measurement = 'entry1/measurement'
_NX_metadata = 'entry1/before_scan'
_NX_starttime = 'entry1/start_time'
_NX_endtime = 'entry1/end_time'

# Nexus format specifiers
_nxformat_hkl = MetaFormat('hkl = (%1.3g,%1.3g,%1.3g)', ['/entry1/before_scan/diffractometer_sample/h', '/entry1/before_scan/diffractometer_sample/k', '/entry1/before_scan/diffractometer_sample/l'])
_nxformat_atten=MetaFormat('Atten = %3d (%1.3g)', ['/entry1/before_scan/gains_atten/Atten', '/entry1/before_scan/gains_atten/Transmission'])
_nxformat_energy = MetaFormat('E = %7.4f keV', ['/entry1/sample/beam/incident_energy'])
_nxformat_temp = MetaFormat('T = %6.2f K', ['/entry1/before_scan/lakeshore/Ta'])
_nxformat_temp2 = MetaFormat('Ta = %6.2f K, Tb = %6.2f K', ['/entry1/before_scan/lakeshore/Ta', '/entry1/before_scan/lakeshore/Tb'])
_nxformat_ss = MetaFormat('ss = [%4.2f,%4.2f]', ['/entry1/before_scan/jjslits/s5xgap', '/entry1/before_scan/jjslits/s5ygap'])
_nxformat_ds = MetaFormat('ds = [%4.2f,%4.2f]', ['/entry1/before_scan/s7xgap/s7xgap', '/entry1/before_scan/s7ygap/s7ygap'])
_nxformat_psi = MetaFormat('psi = %1.3g', ['/entry1/before_scan/psi/psi'])
_nxformat_euler = MetaFormat('eta: %9.3f chi: %9.3f phi: %9.3f mu: %9.3f delta: %9.3f gamma: %9.3f', ['/entry1/before_scan/diffractometer_sample/eta', '/entry1/before_scan/diffractometer_sample/chi', '/entry1/before_scan/diffractometer_sample/phi', '/entry1/before_scan/diffractometer_sample/mu', '/entry1/before_scan/diffractometer_sample/delta', '/entry1/before_scan/diffractometer_sample/gam'])
_nxformat_vertical = MetaFormat('eta: %9.3f chi: %9.3f phi: %9.3f delta: %9.3f', ['/entry1/before_scan/diffractometer_sample/eta', '/entry1/before_scan/diffractometer_sample/chi', '/entry1/before_scan/diffractometer_sample/phi', '/entry1/before_scan/diffractometer_sample/delta'])
_nxformat_horizontal = MetaFormat('mu: %9.3f chi: %9.3f phi: %9.3f gamma: %9.3f', ['/entry1/before_scan/diffractometer_sample/mu', '/entry1/before_scan/diffractometer_sample/chi', '/entry1/before_scan/diffractometer_sample/phi', '/entry1/before_scan/diffractometer_sample/gam'])
_nxformat_pos = MetaFormat('sx: %5.3f sy: %5.3f sz: %5.3f', ['/entry1/before_scan/positions/sx', '/entry1/before_scan/positions/sy', '/entry1/before_scan/positions/sz'])

# list of format specifiers
_nxformat = [_nxformat_hkl, _nxformat_atten, _nxformat_energy, _nxformat_temp, _nxformat_temp2, _nxformat_ss, _nxformat_ds, _nxformat_psi, _nxformat_euler, _nxformat_vertical, _nxformat_horizontal, _nxformat_pos]

