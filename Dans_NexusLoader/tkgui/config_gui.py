"""
General tkinter widgets

By Dan Porter
Nov 2019
"""

from ..functions_nexus import loadnexus, nexus_addresses
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


class ConfigGui:
    """
    Configuration selecter
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initialisation----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, config_filename=None, config=None):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Config')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Config file
        if config is None:
            self.config = Config()
            if config_filename is None:
                config_filename = os.path.expanduser('~')
            else:
                self.config.load_json(config_filename)
        else:
            self.config = config
            config_filename = config.filename

        # nexus file
        nexus_filename = self.config.beamline_config['example_nexus']
        self.nexus = loadnexus(nexus_filename)
        self.nexus_addresses = nexus_addresses(self.nexus)

        "----------------menubar----------------------"
        # ---Menu---
        menubar = tk.Menu(self.root)

        men = tk.Menu(menubar, tearoff=0)
        men.add_command(label='Load Config. File', command=self.menfun_load)
        men.add_command(label='Upate from file', command=self.menfun_update)
        men.add_command(label='Save As', command=self.menfun_saveas)
        men.add_command(label='Save', command=self.menfun_save)
        men.add_command(label='Exit', command=self.menfun_exit)
        menubar.add_cascade(label='File', menu=men)

        men = tk.Menu(menubar, tearoff=0)
        men.add_command(label='Manual', command=self.menfun_helpmanual)
        men.add_command(label='About', command=self.menfun_about)
        menubar.add_cascade(label='Help', menu=men)

        self.root.config(menu=menubar)

        "----------------WIDGETS----------------------"

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # textvariables
        self.config_filename = tk.StringVar(frame, config_filename)
        self.config_beamline_filename = tk.StringVar(frame, self.config.beamline_config['filename'])
        self.config_name = tk.StringVar(frame, self.config.beamline_config['name'])
        self.config_description = tk.StringVar(frame, self.config.beamline_config['description'])
        self.exp_working_directory = tk.StringVar(frame, self.config.experiment_config['working_directory'])
        self.exp_title = tk.StringVar(frame, self.config.experiment_config['experiment_title'])
        self.nexus_file_name = tk.StringVar(frame, nexus_filename)
        self.main_parameters_edit_nameformat = tk.StringVar(frame, self.config.beamline_config['nx_name_format'])
        self.main_parameters_edit_scancommand = tk.StringVar(frame, self.config.beamline_config['nx_scan_command'])
        self.main_parameters_edit_measurement = tk.StringVar(frame, self.config.beamline_config['nx_measurement'])
        self.main_parameters_edit_metadata = tk.StringVar(frame, self.config.beamline_config['nx_metadata'])
        self.main_parameters_edit_starttime = tk.StringVar(frame, self.config.beamline_config['nx_starttime'])
        self.main_parameters_edit_endtime = tk.StringVar(frame, self.config.beamline_config['nx_endtime'])
        self.normalisation_format = tk.StringVar(frame, self.config.beamline_config['normalisation_format'])
        self.normalisation_result = tk.DoubleVar(frame, 1.0)
        self.list_parameters_edit_index = tk.IntVar(frame, 0)
        self.list_parameters_edit_format = tk.StringVar(frame, '%5.2f')
        self.list_parameters_edit_output = tk.StringVar(frame, '')
        self.list_parameters_format_list = []
        self.list_parameters_address_list = []
        self.list_parameters_current_fields = []

        # ---Config File---
        frm = tk.LabelFrame(frame, text='Config. File', relief=tk.GROOVE)
        frm.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        line = tk.Frame(frm)
        line.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(line, text='Filename:', font=SF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Label(line, textvariable=self.config_filename)
        var.pack(side=tk.LEFT, padx=3)

        line = tk.LabelFrame(frm, text='Beamline', relief=tk.GROOVE)
        line.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(line, text='Beamline config. file:', font=LF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.config_beamline_filename, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Label(line, text='Name:', font=LF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.config_name, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Label(line, text='Description:', font=LF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.config_description, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        if self.config.experiment_config['working_directory'] is not None:
            line = tk.LabelFrame(frm, text='Experiment', relief=tk.GROOVE)
            line.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

            # ---Working Directory---
            frm = tk.Frame(line)
            frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

            var = tk.Button(frm, text='Working Directory', font=BF, command=self.fun_browse_working_dir, width=20,
                            bg=btn,
                            activebackground=btn_active)
            var.pack(side=tk.LEFT)
            var = tk.Entry(frm, textvariable=self.exp_working_directory, font=TF, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

            # ---Experiment Directories---
            frm = tk.Frame(line)
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
            for exp_dir in self.config.experiment_config['experiment_directories']:
                self.lst_exp.insert(tk.END, exp_dir)

            # ---Experiment Title---
            frm = tk.Frame(line)
            frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

            var = tk.Label(frm, text='Title:', font=BF, width=20)
            var.pack(side=tk.LEFT)

            var = tk.Entry(frm, textvariable=self.exp_title, font=TF, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        # ---Nexus File---
        frm = tk.LabelFrame(frame, text='Example Nexus File', relief=tk.GROOVE)
        frm.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        var = tk.Button(frm, text='Select File', font=BF, command=self.fun_select_nexus_file, width=20, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frm, textvariable=self.nexus_file_name, font=TF, width=40, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=5)

        # ---Main Parameters---
        frm = tk.LabelFrame(frame, text='Basic Parameters', relief=tk.GROOVE)
        frm.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        line = tk.Frame(frm)
        line.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(line, text='Scan Command:', font=LF, width=20)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.main_parameters_edit_scancommand, font=TF, bg=ety, fg=ety_txt, width=20)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        var = tk.Label(line, text='Measurement:', font=LF, width=20)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.main_parameters_edit_measurement, font=TF, bg=ety, fg=ety_txt, width=20)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        var = tk.Label(line, text='Metadata:', font=LF, width=20)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.main_parameters_edit_metadata, font=TF, bg=ety, fg=ety_txt, width=20)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        line = tk.Frame(frm)
        line.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(line, text='Scan file format:', font=LF, width=20)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.main_parameters_edit_nameformat, font=TF, bg=ety, fg=ety_txt, width=20)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        var = tk.Label(line, text='Start time:', font=LF, width=20)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.main_parameters_edit_starttime, font=TF, bg=ety, fg=ety_txt, width=20)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        var = tk.Label(line, text='End time:', font=LF, width=20)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.main_parameters_edit_endtime, font=TF, bg=ety, fg=ety_txt, width=20)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=3)

        # ---Operations---
        frm = tk.LabelFrame(frame, text='Operations', relief=tk.GROOVE)
        frm.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(frm, text='Normalisation:', font=LF)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Select Fields', font=BF, command=self.fun_select_normalisation, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Entry(frm, textvariable=self.normalisation_format, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=5)
        var.bind('<Return>', self.update_normalisation)
        var.bind('<KP_Enter>', self.update_normalisation)
        var = tk.Label(frm, textvariable=self.normalisation_result, font=LF, width=12)
        var.pack(side=tk.RIGHT)

        # ---List Parameters---
        frm = tk.LabelFrame(frame, text='Information List', relief=tk.RIDGE)
        frm.pack(fill=tk.BOTH, expand=tk.YES, padx=3, pady=3)

        # ---List Parameters Adder---
        frm2 = tk.Frame(frm)
        frm2.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=1, pady=1)

        var = tk.Label(frm2, textvariable=self.list_parameters_edit_index, font=LF, width=2)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Button(frm2, text='Select Fields', font=BF, command=self.fun_select_field, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)

        frm3 = tk.Frame(frm2)
        frm3.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=1, pady=1)

        frm4 = tk.Frame(frm3)
        frm4.pack(fill=tk.X, expand=tk.YES, padx=1, pady=1)
        var = tk.Label(frm4, text='Format:', font=LF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frm4, textvariable=self.list_parameters_edit_format, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=5)
        var.bind('<Return>', self.update_format)
        var.bind('<KP_Enter>', self.update_format)

        frm4 = tk.Frame(frm3)
        frm4.pack(fill=tk.BOTH, expand=tk.YES, padx=1, pady=1)
        var = tk.Label(frm4, text='Output:', font=LF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Label(frm4, textvariable=self.list_parameters_edit_output, relief=tk.GROOVE, font=LF)
        var.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=5)

        var = tk.Button(frm2, text='Change', font=BF, command=self.fun_change_default_param, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Button(frm2, text='Add', font=BF, command=self.fun_add_default_param, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Button(frm2, text='Remove', font=BF, command=self.fun_remove_default_param, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)

        # ---List Parameters TEXT---
        frm2 = tk.Frame(frm)
        frm2.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=1, pady=1)
        sclx = tk.Scrollbar(frm2, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)
        scly = tk.Scrollbar(frm2)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.lst_data = tk.Listbox(frm2, font=HF, selectmode=tk.SINGLE, width=60, height=10, bg=ety,
                                   xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.lst_data.configure(exportselection=True)
        self.lst_data.pack(fill=tk.BOTH, expand=tk.YES)
        self.generate_list_parameters()
        # self.lst_data.select_set(0)
        self.lst_data.bind("<<ListboxSelect>>", self.fun_select_list_parameter)
        # print( self.lst_data.curselection()[0] )

        sclx.config(command=self.lst_data.xview)
        scly.config(command=self.lst_data.yview)

        self.load_config()
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def load_config(self):
        """Load config file"""
        config_filename = self.config_filename.get()
        self.config.load_json(config_filename)
        cdict = self.config.beamline_config

        self.config_name.set(cdict['name'])
        self.config_description.set(cdict['description'])

        self.main_parameters_edit_scancommand.set(cdict['nx_scan_command'])
        self.main_parameters_edit_measurement.set(cdict['nx_measurement'])
        self.main_parameters_edit_metadata.set(cdict['nx_metadata'])
        self.main_parameters_edit_nameformat.set(cdict['nx_name_format'])
        self.main_parameters_edit_starttime.set(cdict['nx_starttime'])
        self.main_parameters_edit_endtime.set(cdict['nx_endtime'])

        self.list_parameters_format_list = cdict['format_specifiers']['format']
        self.list_parameters_address_list = cdict['format_specifiers']['address']
        self.generate_list_parameters()

    def save_config(self):
        """Save config file"""
        config_filename = self.config_filename.get()

        # Beamline part
        cdict = self.config.beamline_config

        cdict['name'] = self.config_name.get()
        cdict['description'] = self.config_description.get()

        cdict['nx_scan_command'] = self.main_parameters_edit_scancommand.get()
        cdict['nx_measurement'] = self.main_parameters_edit_measurement.get()
        cdict['nx_metadata'] = self.main_parameters_edit_metadata.get()
        cdict['nx_name_format'] = self.main_parameters_edit_nameformat.get()
        cdict['nx_starttime'] = self.main_parameters_edit_starttime.get()
        cdict['nx_endtime'] = self.main_parameters_edit_endtime.get()

        cdict['format_specifiers']['format'] = self.list_parameters_format_list
        cdict['format_specifiers']['address'] = self.list_parameters_address_list

        # Experiment part
        edict = self.config.experiment_config
        if edict['working_directory'] is not None:
            edict['working_directory'] = self.exp_working_directory.get()
            edict['experiment_directories'] = self.lst_exp.get(0, tk.END)
            edict['experiment_title'] = self.exp_title.get()

        self.config.save_json(config_filename)

    def load_nexus(self, event=None):
        """Load nexus file and get list of nexus addresses"""
        filename = self.nexus_file_name.get()
        self.nexus = loadnexus(filename)
        self.nexus_addresses = nexus_addresses(self.nexus)

    def generate_list_parameters(self):
        """Populate list parameters list box"""

        self.lst_data.delete(0, tk.END)
        for format, address in zip(self.list_parameters_format_list, self.list_parameters_address_list):
            str_address = ', '.join([field.split('/')[-1] for field in address])
            strval = '%40s | %20s' % (format, str_address)
            self.lst_data.insert(tk.END, strval)

    def update_normalisation(self, event=None):
        """Test normalisation"""
        fmt = self.normalisation_format.get()
        self.config.beamline_config['normalisation_format'] = fmt
        try:
            norm_value = self.config.normalise_value(1.0, self.nexus)
        except NameError:
            norm_value = np.nan
        self.normalisation_result.set(norm_value)

    def update_format(self, event=None):
        """Test format"""

        fmt = self.list_parameters_edit_format.get()
        address = self.list_parameters_current_fields
        values = [self.nexus[a] for a in address]
        try:
            newstring = fmt % tuple(values)
        except TypeError:
            newstring = 'Error'
        self.list_parameters_edit_output.set(newstring)

    "------------------------------------------------------------------------"
    "----------------------------Menu Functions------------------------------"
    "------------------------------------------------------------------------"

    def menfun_load(self):
        """Load beamline config file"""
        defdir = self.config_filename.get()
        filename = filedialog.askopenfilename(
            parent=self.root,
            title='Select Config. file',
            initialfile=defdir,
            filetypes=[('JSON File', '.json'),
                       ('All files', '.*')]
        )
        if filename:
            self.config_filename.set(filename)
        self.load_config()

    def menfun_update(self):
        """Load experiment config file"""
        self.load_config()

    def menfun_saveas(self):
        """Start edit config gui"""
        defdir = self.config_filename.get()
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title='Save Config. file',
            initialfile=defdir,
            filetypes=[('JSON File', '.json'),
                       ('All files', '.*')]
        )
        if filename:
            self.config_filename.set(filename)
            self.save_config()

    def menfun_save(self):
        """Save current config file"""
        self.save_config()

    def menfun_exit(self):
        """Exit"""
        self.fun_close()

    def menfun_helpmanual(self):
        """Open browser to help manual"""
        print('Not implemented yet')

    def menfun_about(self):
        """Show about info"""
        print('By Dan Porter')

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def fun_select_nexus_file(self):
        """Select Nexus file"""

        defdir = self.nexus_file_name.get()
        filename = filedialog.askopenfilename(initialdir=defdir,
                                              filetypes=[('Nexus File', '.nxs'),
                                                         ('All files', '.*')])
        if filename:
            self.nexus_file_name.set(filename)

    def fun_browse_working_dir(self):
        """Select working directory"""
        dir = filedialog.askdirectory()#(initialdir=inidir)
        self.exp_working_directory.set(dir)

    def fun_browse_experiment(self):
        """Add experiment directory to list"""
        dir = filedialog.askdirectory()  # (initialdir=inidir)
        self.lst_exp.insert(tk.END, dir)

    def fun_remove_experiment(self):
        """Remove experiment direcotory from list"""
        if len(self.lst_exp.curselection()) == 0: return
        index = self.lst_exp.curselection()[0]
        self.lst_exp.delete(index)

    def fun_select_normalisation(self, event=None):
        """Select NEXUS fields for normalisation"""

        self.load_nexus()
        out = SelectionBox(self.root, self.nexus_addresses,
                           self.config.beamline_config['nx_normalisation_addresses'],
                           'Select fields', True).show()
        self.config.beamline_config['nx_normalisation_addresses'] = out
        str_address = 'x/'+'/'.join([field.split('/')[-1] for field in out])
        self.normalisation_format.set(str_address)
        self.update_normalisation()

    def fun_select_list_parameter(self, event=None):
        """On selection of list parameter list box, load to edit box"""
        if len(self.lst_data.curselection()) == 0: return
        index = self.lst_data.curselection()[0]
        address = self.list_parameters_address_list[index]
        self.list_parameters_edit_index.set(index)
        self.list_parameters_edit_format.set(self.list_parameters_format_list[index])
        self.list_parameters_current_fields = address
        self.update_format()

    def fun_select_field(self):
        """Select NEXUS field(s)"""

        self.load_nexus()
        out = SelectionBox(self.root, self.nexus_addresses,
                           self.list_parameters_current_fields,
                           'Select fields', True).show()
        self.list_parameters_current_fields = out
        str_address = ', '.join(['%s = %%s' % field.split('/')[-1] for field in out])
        self.list_parameters_edit_format.set(str_address)
        self.update_format()

    def fun_change_default_param(self):
        """Change list parameter list item"""

        index = self.list_parameters_edit_index.get()
        format = self.list_parameters_edit_format.get()
        address = self.list_parameters_current_fields
        self.list_parameters_format_list[index] = format
        self.list_parameters_address_list[index] = address
        self.update_format()
        self.generate_list_parameters()

    def fun_add_default_param(self):
        """Add list parameter list item"""

        format = self.list_parameters_edit_format.get()
        address = self.list_parameters_current_fields
        self.list_parameters_format_list += [format]
        self.list_parameters_address_list += [address]
        self.generate_list_parameters()

    def fun_remove_default_param(self):
        """Remove list parameter list item"""

        index = self.list_parameters_edit_index.get()
        del(self.list_parameters_format_list[index])
        del(self.list_parameters_address_list[index])
        self.generate_list_parameters()

    def fun_save_config(self):
        """Save config button"""
        self.save_config()

    def fun_close(self):
        """close window"""
        self.root.destroy()
