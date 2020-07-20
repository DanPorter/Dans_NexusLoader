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
import numpy as np

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox

_figure_size = [6, 4]


class ScanInfo:
    """
    A standalone GUI window that displays information from a range of scans
    Usage:
        ScanInfo(exp, scan_numbers)

    exp is an Experiment class
    scan_numbers is a list of scans to initialise
    """

    def __init__(self, experiment, scan_numbers=[]):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Scan Info')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.experiment = experiment
        self.addresses = []
        self.field_names = tk.StringVar(self.root, '')

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.NO)

        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)

        "-------------------------Buttons------------------------------"
        frm = tk.Frame(line)
        frm.pack(side=tk.LEFT, fill=tk.Y, expand=tk.NO)

        but1 = tk.Button(frm, text='Display', font=BF, command=self.fun_display, width=20, bg=btn,
                        activebackground=btn_active)
        but2 = tk.Button(frm, text='Select Fields', font=BF, command=self.fun_fields, width=20, bg=btn,
                        activebackground=btn_active)
        but3 = tk.Button(frm, text='Exit', font=BF, command=self.f_exit, width=20, bg=btn,
                        activebackground=btn_active)

        but3.pack(side=tk.BOTTOM)
        but2.pack(side=tk.BOTTOM)
        but1.pack(side=tk.BOTTOM, fill=tk.Y, expand=tk.YES)

        frm = tk.Frame(line)
        frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        "-------------------------Scannos------------------------------"
        frm_txt = tk.LabelFrame(frm, text='Scan numbers', relief=tk.RIDGE)
        frm_txt.pack(fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)

        scl_texty = tk.Scrollbar(frm_txt)
        scl_texty.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.scannos = tk.Text(
            frm_txt,
            font=HF,
            width=40,
            height=3,
            wrap=tk.WORD,
            background='white',
            yscrollcommand=scl_texty.set
        )
        self.scannos.configure(exportselection=True)

        # Populate text box
        self.scannos.insert(tk.END, str(scan_numbers))

        self.scannos.pack(fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)

        scl_texty.config(command=self.scannos.yview)

        "-------------------------Fields------------------------------"
        frm_txt = tk.LabelFrame(frm, text='Parameters', relief=tk.RIDGE)
        frm_txt.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        var = tk.Label(frm_txt, textvariable=self.field_names)
        var.pack(fill=tk.X, expand=tk.YES, padx=2)

        "-------------------------Infobox------------------------------"
        # Eval box with scroll bar
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        scl_textx = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        scl_textx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scl_texty = tk.Scrollbar(frame)
        scl_texty.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.text = tk.Listbox(
            frame,
            font=HF,
            width=60,
            height=5,
            background='white',
            xscrollcommand=scl_textx.set,
            yscrollcommand=scl_texty.set
        )
        self.text.configure(exportselection=True)

        # Populate text box
        self.text.insert(tk.END, '')

        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        scl_textx.config(command=self.text.xview)
        scl_texty.config(command=self.text.yview)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def get_scannos(self):
        scanstr = self.scannos.get('1.0', tk.END)
        if not scanstr: return None
        return list(np.asarray(eval(scanstr), dtype=int).reshape(-1))

    def set_fields(self):
        names = [os.path.basename(address) for address in self.addresses]
        self.field_names.set(', '.join(names))

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def fun_display(self):
        """Display scan info"""
        scannos = self.get_scannos()
        variable_addresses = self.addresses
        if scannos is None: return
        self.text.delete(0, tk.END)
        for scanno in scannos:
            scan = self.experiment.loadscan(scanno)
            self.text.insert(tk.END, scan.info_line(variable_addresses))

    def fun_fields(self):
        """Select fields"""
        # Load first scan
        scannos = self.get_scannos()
        if scannos is None:
            messagebox.showinfo(
                title='Scan Info',
                message='Enter a scan number to select metadata fields',
            )
            return
        scan = self.experiment.loadscan(scannos[0])
        meta_addresses = scan.metadata.addresses()
        out = SelectionBox(
            parent=self.root,
            data_fields=meta_addresses,
            current_selection=self.addresses,
            title='Select metadata to display',
            multiselect=True
        ).show()
        self.addresses = out
        self.set_fields()

    def f_exit(self):
        """Closes the current data window"""
        self.root.destroy()
