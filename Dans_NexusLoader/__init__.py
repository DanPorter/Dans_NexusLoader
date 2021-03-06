"""
Dans_NexusLoader

Load data files from I16, combined with automated fitting and plotting

Usage:
    ***In Python***
    import Dans_NexusLoader as dnex
    exp = dnex.Experiment('data/directory')
    scan = exp.loadscan(12345)
    scan.plot()
    fit = scan.fit(plot_result=True)

Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m -matplotlib tk Dans_NexusLoader

By Dan Porter, PhD
Diamond
2019

Version 0.3
Last updated: 15/07/20

Version History:
21/10/19 0.1.0  Version History started.
15/07/20 0.3.0  Further work on config, data_loaders and gui
"""


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
from . import nexus_config
from .data_loaders import Experiment, Scan, MultiScans

__version__ = "0.3.0"
__date__ = "15/07/20"

