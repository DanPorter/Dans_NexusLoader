"""
Dans_NexusLoader tkinter GUI
"""

import matplotlib

matplotlib.use("TkAgg")

__version__ = "0.1.0"
__date__ = "15/07/20"

from .config_gui import ConfigGui
from .image_gui import ImageGui
from .experiment_gui import ExperimentGui
