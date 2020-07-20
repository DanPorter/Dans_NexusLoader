"""
GUI for viewing detector images

By Dan Porter
Nov 2019
"""

from ..functions_nexus import loadnexus, nexus_addresses, loadimage
from ..nexus_config import Config
from .basic_widgets import SelectionBox
from .basic_widgets import TF, BF, SF, MF, LF, HF
from .basic_widgets import bkg, ety, btn, opt, btn2
from .basic_widgets import btn_active, opt_active, txtcol, btn_txt, ety_txt, opt_txt, ttl_txt

import sys, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox

_figure_size = [6, 4]


class ImageGui:
    """
    A standalone GUI window that displays multiple images using a list of filenames
        ImageGui( file_list, name_list, title, initial_index)

    file_list: [] list of filenames
    name_list: [] list of strings for each filename (default=displays filename)
    title: str: title to display in GUI
    initial_index: int: first image to show
    """

    _increment = 1
    _increment_fast = 10

    def __init__(self, file_list, name_list=None, title='', initial_index=0):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Image Display')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.file_list = file_list
        self.index = tk.IntVar(initial_index)
        image = loadimage(self.file_list[initial_index])
        if name_list is None:
            name_list = file_list
        self.name_list = name_list
        self._increment_fast = int(len(file_list)//self._increment_fast)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # textvariables
        self.name_text = tk.StringVar(frame, name_list[initial_index])

        # ---Image title---
        frm = tk.Frame(frame)
        frm.pack(fill=tk.X, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(frm, text=title, font=SF, fg=ttl_txt)
        var.pack(pady=5)
        var = tk.Label(frm, textvariable=self.name_text, font=LF)
        var.pack(pady=3)

        # ---Figure window---
        frm = tk.Frame(frame)
        frm.pack(fill=tk.BOTH, expand=tk.YES)

        self.fig = plt.Figure(figsize=_figure_size, dpi=80)
        self.fig.patch.set_facecolor('w')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        self.ax.set_frame_on(False)
        self.current_image = self.ax.imshow(image)
        self.ax.set_position([0, 0, 1, 1])

        """
        ROIcen = initial_pilcen
        ROIsize = [75, 67]
        pil_centre = initial_pilcen
        pil_size = [195, 487]
        idxi = np.array([ROIcen[0] - ROIsize[0] // 2, ROIcen[0] + ROIsize[0] // 2 + 1])
        idxj = np.array([ROIcen[1] - ROIsize[1] // 2, ROIcen[1] + ROIsize[1] // 2 + 1])
        self.pilp1, = self.ax2.plot(idxj[[0, 1, 1, 0, 0]], idxi[[0, 0, 1, 1, 0]], 'k-', linewidth=2)  # ROI
        self.pilp2, = self.ax2.plot([pil_centre[1], pil_centre[1]], [0, pil_size[0]], 'k:',
                                    linewidth=2)  # vertical line
        self.pilp3, = self.ax2.plot([0, pil_size[1]], [pil_centre[0], pil_centre[0]], 'k:',
                                    linewidth=2)  # Horizontal line
        self.pilp4, = self.ax2.plot([], [], 'r-', linewidth=2)  # ROI background
        self.pilp5, = self.ax2.plot([], [], 'y-', linewidth=2)  # Peak region
        self.ax.set_aspect('equal')
        self.ax.autoscale(tight=True)
        """

        canvas = FigureCanvasTkAgg(self.fig, frm)
        canvas.get_tk_widget().configure(bg='black')
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

        # ---Toolbar---
        frm = tk.Frame(frame)
        frm.pack(expand=tk.YES)

        # Add matplotlib toolbar under plot
        self.toolbar = NavigationToolbar2TkAgg(canvas, frm)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, expand=tk.YES)

        # ---Slider---
        frm = tk.Frame(frame)
        frm.pack(expand=tk.YES, padx=3, pady=3)

        var = tk.Scale(frm, from_=0, to=len(self.file_list)-1, variable=self.index, orient=tk.HORIZONTAL,
                       command=self.fun_scale)
        #var.bind("<ButtonRelease-1>", self.fun_scale)
        var.pack(side=tk.LEFT)

        # ---Image move buttons---
        frm = tk.Frame(frame)
        frm.pack(expand=tk.YES, padx=3, pady=3)

        var = tk.Button(frm, text='<<', font=BF, command=self.fun_left_fast,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='<', font=BF, command=self.fun_left,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='>', font=BF, command=self.fun_right,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='>>', font=BF, command=self.fun_right_fast,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def load_image(self, index):
        self.name_text.set(self.name_list[index])
        image = loadimage(self.file_list[index])
        self.current_image.set_data(image)
        self.toolbar.update()
        self.fig.canvas.draw()

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def fun_scale(self, event=None):
        """Move scroll bar"""
        index = self.index.get()
        self.load_image(index)

    def fun_left_fast(self):
        """Decrease image index 10%"""
        index = self.index.get() - self._increment_fast
        if index < 0:
            index = 0
        self.load_image(index)
        self.index.set(index)

    def fun_left(self):
        """Decrease image index by 1"""
        index = self.index.get() - self._increment
        if index < 0:
            index = 0
        self.load_image(index)
        self.index.set(index)

    def fun_right(self):
        """Increase image index by 1"""
        index = self.index.get() + self._increment
        if index >= len(self.file_list):
            index = len(self.file_list) - 1
        self.load_image(index)
        self.index.set(index)

    def fun_right_fast(self):
        """Increase image index 10%"""
        index = self.index.get() + self._increment_fast
        if index >= len(self.file_list):
            index = len(self.file_list) - 1
        self.load_image(index)
        self.index.set(index)

    def f_exit(self):
        """Closes the current data window"""
        self.root.destroy()
