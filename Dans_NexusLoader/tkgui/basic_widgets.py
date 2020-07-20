"""
General tkinter widgets

By Dan Porter
Nov 2019
"""

import sys, os, re

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize, LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg


__version__ = '0.1'

# Fonts
TF = ["Times", 12]  # entry
BF = ["Times", 14]  # Buttons
SF = ["Times New Roman", 14]  # Title labels
MF = ["Courier", 8]  # fixed distance format
LF = ["Times", 14]  # Labels
HF = ["Courier", 12]  # Text widgets (big)
# Colours - background
bkg = 'snow'
ety = 'white'
btn = 'azure' #'light slate blue'
opt = 'azure' #'light slate blue'
btn2 = 'gold'
# Colours - active
btn_active = 'grey'
opt_active = 'grey'
# Colours - Fonts
txtcol = 'black'
btn_txt = 'black'
ety_txt = 'black'
opt_txt = 'black'
ttl_txt = 'red'


class StringViewer:
    """
    Simple GUI that displays strings
    """

    def __init__(self, string, title=''):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(title)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Textbox height
        height = string.count('\n')
        if height > 40: height = 40

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # --- label ---
        # labframe = tk.Frame(frame,relief='groove')
        # labframe.pack(side=tk.TOP, fill=tk.X)
        # var = tk.Label(labframe, text=label_text,font=SF,justify='left')
        # var.pack(side=tk.LEFT)

        # --- Button ---
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Button(frm1, text='Close', font=BF, command=self.fun_close, bg=btn, activebackground=btn_active)
        var.pack(fill=tk.X)

        # --- Text box ---
        frame_box = tk.Frame(frame)
        frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        scanx = tk.Scrollbar(frame_box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)

        scany = tk.Scrollbar(frame_box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Editable string box
        self.text = tk.Text(frame_box, width=40, height=height, font=HF, wrap=tk.NONE)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.text.insert(tk.END, string)

        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def fun_close(self):
        """close window"""
        self.root.destroy()


"------------------------------------------------------------------------"
"----------------------------Selection Box-------------------------------"
"------------------------------------------------------------------------"


class SelectionBox:
    """
    Displays all data fields and returns a selection
    Making a selection returns a list of field strings

    out = SelectionBox(['field1','field2','field3'], current_selection=['field2'], title='', multiselect=False)
    # Make selection and press "Select" > box disappears
    out.output = ['list','of','strings']

    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initilisation-----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, parent, data_fields, current_selection=[], title='Make a selection', multiselect=True):
        self.data_fields = data_fields
        self.initial_selection = current_selection

        # Create Tk inter instance
        self.root = tk.Toplevel(parent)
        self.root.wm_title(title)
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        self.output = []

        # Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, anchor=tk.N)

        "---------------------------ListBox---------------------------"
        # Eval box with scroll bar
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        sclx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.lst_data = tk.Listbox(frm, font=MF, selectmode=tk.SINGLE, width=60, height=20, bg=ety,
                                   xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.lst_data.configure(exportselection=True)
        if multiselect:
            self.lst_data.configure(selectmode=tk.EXTENDED)
        self.lst_data.bind('<<ListboxSelect>>', self.fun_listboxselect)

        # Populate list box
        for k in self.data_fields:
            # if k[0] == '_': continue # Omit _OrderedDict__root/map
            strval = '{}'.format(k)
            self.lst_data.insert(tk.END, strval)

        self.lst_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        for select in current_selection:
            if select in data_fields:
                idx = data_fields.index(select)
                self.lst_data.select_set(idx)

        sclx.config(command=self.lst_data.xview)
        scly.config(command=self.lst_data.yview)

        # self.txt_data.config(xscrollcommand=scl_datax.set,yscrollcommand=scl_datay.set)

        "----------------------------Search Field-----------------------------"
        frm = tk.LabelFrame(frame, text='Search', relief=tk.RIDGE)
        frm.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        self.searchbox = tk.StringVar(self.root, '')
        var = tk.Entry(frm, textvariable=self.searchbox, font=TF, bg=ety, fg=ety_txt)
        var.bind('<Key>', self.fun_search)
        var.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        "----------------------------Exit Button------------------------------"
        frm_btn = tk.Frame(frame)
        frm_btn.pack(fill=tk.X, expand=tk.YES)

        self.numberoffields = tk.StringVar(self.root, '%3d Selected Fields' % len(self.initial_selection))
        var = tk.Label(frm_btn, textvariable=self.numberoffields, width=20)
        var.pack(side=tk.LEFT)
        btn_exit = tk.Button(frm_btn, text='Select', font=BF, command=self.fun_exitbutton, bg=btn,
                             activebackground=btn_active)
        btn_exit.pack(side=tk.RIGHT)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        #self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def show(self):
        """Run the selection box, wait for response"""

        #self.root.deiconify()  # show window
        self.root.wait_window()  # wait for window
        return self.output

    def fun_search(self, event=None):
        """Search the selection for string"""
        search_str = self.searchbox.get()
        search_str = search_str + event.char
        search_str = search_str.strip().lower()
        if not search_str: return

        # Clear current selection
        self.lst_data.select_clear(0, tk.END)
        view_idx = None
        # Search for whole words first
        for n, item in enumerate(self.data_fields):
            if re.search(r'\b%s\b' % search_str, item.lower()):  # whole word search
                self.lst_data.select_set(n)
                view_idx = n
        # if nothing found, search anywhere
        if view_idx is None:
            for n, item in enumerate(self.data_fields):
                if search_str in item.lower():
                    self.lst_data.select_set(n)
                    view_idx = n
        if view_idx is not None:
            self.lst_data.see(view_idx)
        self.fun_listboxselect()

    def fun_listboxselect(self, event=None):
        """Update label on listbox selection"""
        self.numberoffields.set('%3d Selected Fields' % len(self.lst_data.curselection()))

    def fun_exitbutton(self):
        """Closes the current data window and generates output"""
        selection = self.lst_data.curselection()
        self.output = [self.data_fields[n] for n in selection]
        self.root.destroy()

    def f_exit(self):
        """Closes the current data window"""
        self.output = self.initial_selection
        self.root.destroy()


"------------------------------------------------------------------------"
"------------------------------Python Editor-----------------------------"
"------------------------------------------------------------------------"


class PythonEditor:
    """
    A very simple python editor, load and edit python files, execute them in
    current python shell.
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initialisation----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, disp_str='', filename=''):
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('I16 Python Editor by D G Porter [dan.porter@diamond.ac.uk]')
        self.root.minsize(width=200, height=100)
        self.root.maxsize(width=1800, height=1000)

        box_height = 30
        box_width = 100

        self.savelocation = ''
        self.text_changed = False

        if os.path.isfile(filename):
            self.root.wm_title(filename)
            self.savelocation = filename

        # Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, anchor=tk.N)

        "---------------------------Metadata ListBox---------------------------"
        # Eval box with scroll bar
        frm_text = tk.Frame(frame)
        frm_text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        scl_textx = tk.Scrollbar(frm_text, orient=tk.HORIZONTAL)
        scl_textx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scl_texty = tk.Scrollbar(frm_text)
        scl_texty.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.text = tk.Text(frm_text,
                            font=HF,
                            width=box_width,
                            height=box_height,
                            wrap=tk.NONE,
                            background='white',
                            xscrollcommand=scl_textx.set,
                            yscrollcommand=scl_texty.set)
        self.text.configure(exportselection=True)
        self.text.bind('<Control-s>', self.f_save)
        self.text.bind('<Control-b>', self.f_run)
        self.text.bind('<Control-r>', self.f_run)
        # self.text.bind('<<Modified>>', self.f_change)

        # Populate text box
        self.text.insert(tk.END, disp_str)

        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        scl_textx.config(command=self.text.xview)
        scl_texty.config(command=self.text.yview)

        # self.txt_text.config(xscrollcommand=scl_textx.set,yscrollcommand=scl_texty.set)

        "----------------------------Exit Button------------------------------"
        frm_btn = tk.Frame(frame)
        frm_btn.pack(fill=tk.X)

        bt = tk.Button(frm_btn, text='RUN', font=BF, command=self.f_run)
        bt.pack(side=tk.LEFT)
        bt = tk.Button(frm_btn, text='Exit', font=BF, command=self.f_exit)
        bt.pack(side=tk.RIGHT)
        bt = tk.Button(frm_btn, text='Save As', font=BF, command=self.f_saveas)
        bt.pack(side=tk.RIGHT)
        bt = tk.Button(frm_btn, text='Save', font=BF, command=self.f_save)
        bt.pack(side=tk.RIGHT)
        bt = tk.Button(frm_btn, text='Open', font=BF, command=self.f_open)
        bt.pack(side=tk.RIGHT)

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def f_run(self, event=None):
        """Run the code"""
        global pp, __file__
        __file__ = cf
        code = self.text.get(1.0, tk.END)
        exec (code)
        print('Finished')

    def f_change(self, event=None):
        """Change the saved state"""
        self.text_changed = True
        self.root.wm_title('* ' + self.savelocation)

    def f_open(self):
        """Open a new file"""
        newsavelocation = filedialog.askopenfilename(
            title='Open your python script',
            initialdir=pp.savedir,
            initialfile='script.py',
            defaultextension='.py',
            filetypes=(("python file", "*.py"), ("all files", "*.*")))

        if newsavelocation == '':
            return
        with open(newsavelocation) as file:
            disp_str = file.read()
        PythonEditor(disp_str, newsavelocation)

    def f_save(self, event=None):
        """"Save the file"""
        if self.savelocation == '':
            self.f_saveas()
            return

        code = self.text.get(1.0, tk.END)
        with open(self.savelocation, 'wt') as outfile:
            outfile.write(code)
        self.root.wm_title(self.savelocation)
        self.text_changed = False
        print('Saved as {}'.format(self.savelocation))

    def f_saveas(self):
        """Save the file"""
        code = self.text.get(1.0, tk.END)
        self.savelocation = filedialog.asksaveasfilename(
            title='Save your python script',
            initialdir=pp.savedir,
            initialfile='script.py',
            defaultextension='.py',
            filetypes=(("python file", "*.py"), ("all files", "*.*")))
        if self.savelocation != '':
            self.f_save()

    def f_exit(self):
        """Closes the current text window"""
        if self.text_changed:
            if messagebox.askyesno(self.savelocation, "Would you like to save the script?"):
                self.f_save()
        self.root.destroy()

    def on_closing(self):
        """End mainloop on close window"""
        if self.text_changed:
            if messagebox.askyesno(self.savelocation, "Would you like to save the script?"):
                self.f_save()
        self.root.destroy()


"------------------------------------------------------------------------"
"-------------------------------ColourCutoffs----------------------------"
"------------------------------------------------------------------------"


class ColourCutoffs:
    """
    Change the vmin/vmax colormap limits of the current figure
    Activate form the console by typing:
        ColourCutoffs()
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initilisation-----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self):
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Change clim')
        # self.root.minsize(width=300, height=200)
        # self.root.maxsize(width=400, height=500)

        ini_vmin, ini_vmax = plt.gci().get_clim()

        # Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # GCF button
        frm_btn = tk.Button(frame, text='Get Current Figure', font=BF, command=self.f_gcf)
        frm_btn.pack(fill=tk.X)

        # increment setting
        f_inc = tk.Frame(frame)
        f_inc.pack(fill=tk.X)

        inc_btn1 = tk.Button(f_inc, text='1', font=BF, command=self.f_but1)
        inc_btn1.pack(side=tk.LEFT)

        inc_btn2 = tk.Button(f_inc, text='100', font=BF, command=self.f_but2)
        inc_btn2.pack(side=tk.LEFT)

        inc_btn3 = tk.Button(f_inc, text='1000', font=BF, command=self.f_but3)
        inc_btn3.pack(side=tk.LEFT)

        self.increment = tk.DoubleVar(f_inc, 1.0)
        inc_ety = tk.Entry(f_inc, textvariable=self.increment, width=6)
        inc_ety.pack(side=tk.LEFT)

        # Upper clim
        f_upper = tk.Frame(frame)
        f_upper.pack(fill=tk.X)

        up_left = tk.Button(f_upper, text='<', font=BF, command=self.f_upper_left)
        up_left.pack(side=tk.LEFT)

        self.vmin = tk.DoubleVar(f_upper, ini_vmin)
        up_edit = tk.Entry(f_upper, textvariable=self.vmin, width=12)
        up_edit.bind('<Return>', self.update)
        up_edit.bind('<KP_Enter>', self.update)
        up_edit.pack(side=tk.LEFT, expand=tk.YES)

        up_right = tk.Button(f_upper, text='>', font=BF, command=self.f_upper_right)
        up_right.pack(side=tk.LEFT)

        # Lower clim
        f_lower = tk.Frame(frame)
        f_lower.pack(fill=tk.X)

        lw_left = tk.Button(f_lower, text='<', font=BF, command=self.f_lower_left)
        lw_left.pack(side=tk.LEFT)

        self.vmax = tk.DoubleVar(f_lower, ini_vmax)
        lw_edit = tk.Entry(f_lower, textvariable=self.vmax, width=12)
        lw_edit.bind('<Return>', self.update)
        lw_edit.bind('<KP_Enter>', self.update)
        lw_edit.pack(side=tk.LEFT, expand=tk.YES)

        lw_right = tk.Button(f_lower, text='>', font=BF, command=self.f_lower_right)
        lw_right.pack(side=tk.LEFT)

        # Update button
        frm_btn = tk.Button(frame, text='Update', font=BF, command=self.update)
        frm_btn.pack(fill=tk.X)

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def f_gcf(self):
        # fig = plt.gcf()
        # fig.canvas.manager.window.raise_()

        new_vmin, new_vmax = plt.gci().get_clim()
        self.vmin.set(new_vmin)
        self.vmax.set(new_vmax)

    def f_but1(self):
        self.increment.set(1.0)

    def f_but2(self):
        self.increment.set(100)

    def f_but3(self):
        self.increment.set(1e3)

    def f_upper_left(self):
        inc = self.increment.get()
        cur_vmin = self.vmin.get()
        self.vmin.set(cur_vmin - inc)
        self.update()

    def f_upper_right(self):
        inc = self.increment.get()
        cur_vmin = self.vmin.get()
        self.vmin.set(cur_vmin + inc)
        self.update()

    def f_lower_left(self):
        inc = self.increment.get()
        cur_vmax = self.vmax.get()
        self.vmax.set(cur_vmax - inc)
        self.update()

    def f_lower_right(self):
        inc = self.increment.get()
        cur_vmax = self.vmax.get()
        self.vmax.set(cur_vmax + inc)
        self.update()

    def update(self, event=None):
        cur_vmin = self.vmin.get()
        cur_vmax = self.vmax.get()

        # fig = plt.gcf()
        # fig.canvas.manager.window.raise_()
        plt.clim(cur_vmin, cur_vmax)
        plt.show()
