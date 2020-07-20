"""
GUI for showing scan plots and details

By Dan Porter
Nov 2019
"""

from ..data_loaders import Experiment, Scan, MultiScans
from ..functions_nexus import loadnexus, nexus_addresses
from ..nexus_config import Config
from .basic_widgets import SelectionBox
from .basic_widgets import TF, BF, SF, MF, LF, HF
from .basic_widgets import bkg, ety, btn, opt, btn2
from .basic_widgets import btn_active, opt_active, txtcol, btn_txt, ety_txt, opt_txt, ttl_txt
from .config_gui import ConfigGui
from .scaninfo_gui import ScanInfo

import sys, os
import numpy as np
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
_figure_dpi = 90


class ExperimentGui:
    """
    Configuration selecter
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initialisation----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, experiment=None, config_filename=None):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Dans Nexus Loader')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        if type(experiment) is Experiment:
            self.experiment = experiment
        else:
            self.experiment = Experiment(config_file=config_filename)

        print(self.experiment.info())
        print(self.experiment.config.info())

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # textvariables
        self.working_directory = tk.StringVar(frame, os.path.abspath(self.experiment.working_directory))
        self.exp_title = tk.StringVar(frame, self.experiment.title)
        self.scan_number = tk.StringVar(frame, "0")

        "----------------menubar----------------------"
        # ---Menu---
        menubar = tk.Menu(self.root)

        men = tk.Menu(menubar, tearoff=0)
        men.add_command(label='Load Beamline', command=self.menfun_loadbeamline)
        men.add_command(label='Load Experiment', command=self.menfun_loadexperiment)
        men.add_command(label='Edit config.', command=self.menfun_editconfig)
        men.add_command(label='Save config.', command=self.menfun_saveconfig)
        men.add_command(label='Exit', command=self.menfun_exit)
        menubar.add_cascade(label='Config.', menu=men)

        men = tk.Menu(menubar, tearoff=0)
        men.add_command(label='Log inspector', command=self.menfun_logviewer)
        men.add_command(label='Last log line', command=self.menfun_logline)
        menubar.add_cascade(label='Log', menu=men)

        men = tk.Menu(menubar, tearoff=0)
        men.add_command(label='Update Jupyter logbook', command=self.menfun_jupyter)
        menubar.add_cascade(label='Jupyter', menu=men)

        men = tk.Menu(menubar, tearoff=0)
        men.add_command(label='Manual', command=self.menfun_helpmanual)
        men.add_command(label='About', command=self.menfun_about)
        menubar.add_cascade(label='Help', menu=men)

        self.root.config(menu=menubar)

        "----------------LINE 1----------------------"
        line = tk.Frame(frame)
        line.pack(fill=tk.BOTH, expand=tk.YES)

        # ---LEFT SIDE---
        side = tk.Frame(line)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # ---Working Directory---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        var = tk.Button(frm, text='Working Directory', font=BF, command=self.fun_browse_working_dir, width=20, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frm, textvariable=self.working_directory, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        # ---Experiment Directories---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        frm2 = tk.Frame(frm)
        frm2.pack(side=tk.LEFT)

        var = tk.Button(frm2, text='Add Experiment', font=BF, command=self.fun_browse_experiment, width=20, bg=btn,
                        activebackground=btn_active)
        var.pack()
        var = tk.Button(frm2, text='Remove', font=BF, command=self.fun_remove_experiment, width=20, bg=btn,
                        activebackground=btn_active)
        var.pack()

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.lst_exp = tk.Listbox(frm, font=HF, selectmode=tk.SINGLE, width=40, height=3, bg=ety,
                                   yscrollcommand=scly.set)
        self.lst_exp.configure(exportselection=False)
        self.lst_exp.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        self.lst_exp.select_set(0)
        scly.config(command=self.lst_exp.yview)
        # Add directories
        for exp_dir in self.experiment.path:
            self.lst_exp.insert(tk.END, os.path.abspath(exp_dir))

        # ---Experiment Title---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        var = tk.Label(frm, text='Title:', font=BF, width=20)
        var.pack(side=tk.LEFT)

        var = tk.Entry(frm, textvariable=self.exp_title, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        # ---RIGHT SIDE---
        side = tk.Frame(line)
        side.pack(side=tk.LEFT)

        # ---Config Files---
        frm = tk.Frame(side)
        frm.pack()

        # ---GUI Buttons---
        frm = tk.Frame(side)
        frm.pack()

        "----------------LINE 2----------------------"
        #line = tk.Frame(frame)
        #line.pack(fill=tk.BOTH, expand=tk.YES)

        # ---LEFT SIDE---
        #side = tk.Frame(line)
        #side.pack(side=tk.LEFT)

        "----------------LINE 3----------------------"
        line = tk.Frame(frame)
        line.pack(fill=tk.BOTH, expand=tk.YES)

        # ---LEFT SIDE---
        side = tk.Frame(line)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.NO)

        var = tk.Button(side, text='Check Scans', font=BF, command=self.fun_checkscans, width=20, bg=btn,
                        activebackground=btn_active)
        var.pack(fill=tk.X)

        # Scan list
        frm = tk.Frame(side)
        frm.pack(fill=tk.BOTH, expand=tk.YES)

        sclx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.scan_list = tk.Listbox(frm, font=MF, selectmode=tk.EXTENDED, width=20, bg=ety,
                                    xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.scan_list.configure(exportselection=False)
        self.scan_list.bind('<<ListboxSelect>>', self.fun_choose_scan)
        self.scan_list.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.current_scans = []

        sclx.config(command=self.scan_list.xview)
        scly.config(command=self.scan_list.yview)

        # ---MIDDLE---
        side = tk.Frame(line)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # ---Scan number---
        frm = tk.Frame(side)
        frm.pack()

        var = tk.Label(frm, text='Scan number:', font=LF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(frm, textvariable=self.scan_number, font=TF, bg=ety, fg=ety_txt)
        var.bind('<Return>', self.load_scan)
        var.bind('<KP_Enter>', self.load_scan)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)
        var = tk.Button(frm, text='<', font=BF, command=self.fun_scan_left, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='>', font=BF, command=self.fun_scan_right, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Last', font=BF, command=self.fun_scan_last, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # ---Scan Details---
        frm = tk.Frame(side)
        frm.pack(fill=tk.BOTH, expand=tk.YES)

        scanx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)
        scany = tk.Scrollbar(frm)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        self.detail_text = tk.Listbox(frm, width=40, height=10, font=HF, bg=bkg)
        self.detail_text.configure(selectmode=tk.EXTENDED, exportselection=False)
        self.detail_text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.detail_text.insert(tk.END, '')

        self.detail_text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.detail_text.xview)
        scany.config(command=self.detail_text.yview)

        # ---RIGHT SIDE---
        side = tk.Frame(line)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # ---x-axis variable---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.X)

        labfrm = tk.LabelFrame(frm, text='x-axis ', relief=tk.RIDGE)
        labfrm.pack(side=tk.LEFT, padx=3)

        self.x_axis = tk.StringVar(self.root, 'x-axis')
        self.x_axis_menu = tk.OptionMenu(labfrm, self.x_axis, 'x-axis')
        self.x_axis_menu.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        self.x_axis_menu["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        self.x_axis_menu.pack(side=tk.LEFT)

        # ---Plot Command box---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.X)

        var = tk.Button(frm, text='Command', font=BF, command=self.fun_plotcommand, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='?', font=BF, command=self.fun_plotcommand_help, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        self.plot_command = tk.StringVar(self.root, '')
        var = tk.Entry(frm, textvariable=self.plot_command, font=TF, bg=ety, fg=ety_txt)
        var.bind('<Return>', self.fun_plotcommand)
        var.bind('<KP_Enter>', self.fun_plotcommand)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        # ---Scan Plot---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)

        self.fig1 = plt.Figure(figsize=_figure_size, dpi=_figure_dpi)
        self.fig1.patch.set_facecolor('w')
        self.ax1 = self.fig1.add_subplot(111)
        #self.ax1.set_xticklabels([])
        #self.ax1.set_yticklabels([])
        #self.ax1.set_xticks([])
        #self.ax1.set_yticks([])
        self.ax1.set_autoscaley_on(True)
        self.ax1.set_autoscalex_on(True)
        #self.ax1.set_frame_on(False)
        #self.ax1.set_position([0, 0, 1, 1])

        self.plot_list = []

        canvas1 = FigureCanvasTkAgg(self.fig1, frm)
        canvas1.get_tk_widget().configure(bg='black')
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

        # Add matplotlib toolbar under plot
        self.toolbar = NavigationToolbar2TkAgg(canvas1, frm)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, expand=tk.YES)

        """
        # ---Image Plot---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)

        self.fig2 = plt.Figure(figsize=_figure_size, dpi=_figure_dpi)
        self.fig2.patch.set_facecolor('w')
        self.ax2 = self.fig2.add_subplot(111)
        #self.ax2.set_xticklabels([])
        #self.ax2.set_yticklabels([])
        #self.ax2.set_xticks([])
        #self.ax2.set_yticks([])
        self.ax2.set_autoscaley_on(True)
        self.ax2.set_autoscalex_on(True)
        self.ax2.set_frame_on(False)
        #self.current_image = self.ax2.imshow(image)
        self.ax2.set_position([0, 0, 1, 1])

        canvas2 = FigureCanvasTkAgg(self.fig2, frm)
        canvas2.get_tk_widget().configure(bg='black')
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        """

        # ---FAR RIGHT SIDE---
        side = tk.LabelFrame(line, text='y-axis')
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # ---y-axis variable---
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.X)

        #var = tk.Label(frm, text='y-axis: ', font=LF)
        #var.pack(side=tk.LEFT, padx=3)

        frm = tk.Frame(side)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        sclx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.y_axis_menu = tk.Listbox(frm, font=MF, selectmode=tk.EXTENDED, width=20, height=20, bg=ety,
                                      xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.y_axis_menu.configure(exportselection=False)
        self.y_axis_menu.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        sclx.config(command=self.y_axis_menu.xview)
        scly.config(command=self.y_axis_menu.yview)

        # ---Run mainloops---
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def update_experiment(self):
        """Update experiment directories"""
        working_dir = self.working_directory.get()
        exp_list = self.lst_exp.get(0, tk.END)
        self.experiment.working_directory = working_dir
        self.experiment.path = exp_list
        self.update_experiment_scans()

    def update_experiment_scans(self):
        """Get list of scan numbers from available experiment directories"""
        new_scans = self.experiment.allscannumbers()
        if len(new_scans) != len(self.current_scans):
            self.scan_list.delete(0, tk.END)
            for scan_no in new_scans[::-1]:
                self.scan_list.insert(tk.END, str(scan_no))
            self.current_scans = new_scans[::-1]

    def get_scan_numbers(self):
        """Get scan numbers from self.scan_number"""
        self.update_experiment()
        scanstr = self.scan_number.get()
        # evaluate string and cast as numpy array
        return list(np.asarray(eval(scanstr), dtype=int).reshape(-1))

    def load_scan(self, event=None):
        """Read scan number, update gui"""

        scannos = self.get_scan_numbers()
        if len(scannos) > 1:
            print('Multiscan - to be implemented')
            scan = self.experiment.loadscans(scannos)  # return MultiScans
        else:
            scan = self.experiment.loadscan(scannos[0])
        self.update_details(scan)
        self.update_plot(scan)

    def update_details(self, scan):
        """Update scan details text"""

        if type(scan) is Scan:
            text_list = self.update_scan(scan)
        elif type(scan) is MultiScans:
            text_list = ['Multi-scan not implemented yet']
        else:
            text_list = ['Scan does not exist.']

        self.detail_text.delete(0, tk.END)
        for line in text_list:
            self.detail_text.insert(tk.END, line)

    def update_scan(self, scan):
        """Update items for indiviual scan, return list of text lines"""
        # Basic parameters
        text_list = ['  Filename: %s' % scan.filename,
                     '   Command: %s' % scan.cmd,
                     '    X-Axis: %s' % scan.autox(),
                     '    Y-Axis: %s' % scan.autoy(),
                     'Start time: %s' % scan.time().strftime('%Y-%m-%d %H:%M'),
                     '  Duration: %s' % scan.duration()]
        # config file scan details
        text_list += self.experiment.config.get_parameter_strings(scan.nexus)

        # x-axis values
        self.x_axis_menu['menu'].delete(0, 'end')
        for key in scan.measurement_keys:
            self.x_axis_menu['menu'].add_command(
                label=key,
                command=lambda k=key, s=scan: self.fun_choose_xaxis(k, s)
            )
        self.x_axis.set(scan.autox())
        # Change color of default
        #autox_idx = scan.measurement_keys.index(scan.autoy())

        # y-axis values
        ykeys = scan.measurement_keys[::-1]
        self.y_axis_menu.delete(0, tk.END)
        for key in ykeys:
            self.y_axis_menu.insert(tk.END, key)
        autoy_idx = ykeys.index(scan.autoy())
        self.y_axis_menu.select_set(autoy_idx)
        self.y_axis_menu.itemconfig(autoy_idx, fg='red')
        self.y_axis_menu.event_generate("<<ListboxSelect>>")
        self.y_axis_menu.bind("<<ListboxSelect>>", lambda event, s=scan: self.update_plot(s))
        return text_list

    def update_plot(self, scan):
        """Update scan plot"""

        # Remove old plots
        for ln in self.plot_list:
            ln.remove()
        self.plot_list = []

        if type(scan) is Scan:
            x_name = self.x_axis.get()
            y_selection = self.y_axis_menu.curselection()
            y_name = 'None'
            for y_idx in y_selection:
                y_name = self.y_axis_menu.get(y_idx)
                self.plot_list += scan.plot(x_name, y_name, axis=self.ax1, show=False)

            self.ax1.relim()
            self.ax1.autoscale_view()

            #self.toolbar.update()  # update toolbar home position

            self.ax1.set_xlabel(x_name)
            self.ax1.set_ylabel(y_name)
            self.ax1.set_title('#{}'.format(scan.scan_number), fontsize=16)
            self.fig1.canvas.draw()
        elif type(scan) is MultiScans:
            x_name = self.x_axis.get()
            y_selection = self.y_axis_menu.curselection()
            y_name = 'None'
            for y_idx in y_selection:
                y_name = self.y_axis_menu.get(y_idx)
                self.plot_list += scan.plot(x_name, y_name, axis=self.ax1, show=False)

            self.ax1.relim()
            self.ax1.autoscale_view()

            # self.toolbar.update()  # update toolbar home position

            self.ax1.set_xlabel(x_name)
            self.ax1.set_ylabel(y_name)
            self.ax1.set_title('{}'.format(scan), fontsize=16)
            self.fig1.canvas.draw()
        else:
            pass

    "------------------------------------------------------------------------"
    "----------------------------Menu Functions------------------------------"
    "------------------------------------------------------------------------"

    def menfun_loadbeamline(self):
        """Load beamline config file"""
        pass

    def menfun_loadexperiment(self):
        """Load experiment config file"""
        pass

    def menfun_editconfig(self):
        """Start edit config gui"""
        ConfigGui(config=self.experiment.config)

    def menfun_saveconfig(self):
        """Save current config file"""
        pass

    def menfun_exit(self):
        """Exit"""
        self.fun_close()

    def menfun_logviewer(self):
        """Launch logviewer GUI"""
        print('Not implemented yet')
        pass

    def menfun_logline(self):
        """Launch logviewer GUI"""
        print('Not implemented yet')
        pass

    def menfun_jupyter(self):
        """Launch jupyter logbook GUI"""
        print('Not implemented yet')
        pass

    def menfun_helpmanual(self):
        """Open browser to help manual"""
        message = "Manual will go here, or maybe a link to a website, \nexcept links aren't possible in tkinter."
        messagebox.showinfo(
            title='Dans_NexusLoader',
            message=message,
            parent=self.root,
        )

    def menfun_about(self):
        """Show about info"""
        from .. import __version__, __date__
        message = "Dans_NexusLoader\nVersion: %s\n  Date: %s\nBy Dan Porter\nDiamond Light Source Ltd."
        message = message % (__version__, __date__)
        messagebox.showinfo(
            title='Dans_NexusLoader',
            message=message,
            parent=self.root,
        )

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def fun_browse_working_dir(self):
        """Select working directory"""
        dir = filedialog.askdirectory()#(initialdir=inidir)
        self.working_directory.set(dir)

    def fun_browse_experiment(self):
        """Add experiment directory to list"""
        dir = filedialog.askdirectory()  # (initialdir=inidir)
        if dir:
            self.lst_exp.insert(tk.END, dir)

    def fun_remove_experiment(self):
        """Remove experiment direcotory from list"""
        if len(self.lst_exp.curselection()) == 0: return
        index = self.lst_exp.curselection()[0]
        self.lst_exp.delete(index)

    def fun_checkscans(self):
        """Open Checkscans GUI"""
        current_scans = self.get_scan_numbers()
        ScanInfo(self.experiment, current_scans)

    def fun_plotcommand(self, event=None):
        """Runs plot command"""
        command_str = self.plot_command.get()
        print('Running: %s' % command_str)

    def fun_plotcommand_help(self):
        """Show help box for plot command"""
        messagebox.showinfo(
            title='Dans_NexusLoader',
            message='What does the Command button do?',
            parent=self.root,
        )

    def fun_close(self):
        """close window"""
        self.root.destroy()

    def fun_choose_scan(self, event=None):
        """Get scan list values from scan listbox"""
        scan_idx = self.scan_list.curselection()
        # Set listscan colours
        colors = self.experiment.config.colors(len(scan_idx), return_hex=True)
        for idx, col in zip(scan_idx, colors):
            self.scan_list.itemconfig(idx, selectbackground=col)

        scan_numbers = [str(self.current_scans[idx]) for idx in scan_idx]
        txt = ', '.join(scan_numbers)
        if ',' in txt: txt = '[%s]' % txt
        self.scan_number.set(txt)
        self.load_scan()

    def fun_scan_left(self):
        """Decrease scan number"""
        scannos = self.get_scan_numbers()
        scan = scannos[0]
        self.scan_number.set(str(scan - 1))
        self.load_scan()

    def fun_scan_right(self):
        """Decrease scan number"""
        scannos = self.get_scan_numbers()
        scan = scannos[0]
        self.scan_number.set(str(scan + 1))
        self.load_scan()

    def fun_scan_last(self):
        """Get last scan number for experiment"""
        self.update_experiment()
        scanno = self.experiment.lastscan()
        self.scan_number.set(str(scanno))
        self.load_scan()

    def fun_choose_xaxis(self, key, scan):
        """Make a choice from x-axis drop menu"""
        self.x_axis.set(key)
        self.update_plot(scan)
