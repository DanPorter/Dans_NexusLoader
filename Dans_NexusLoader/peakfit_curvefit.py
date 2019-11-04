"""
Peak fitting funcitons from Py16
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from scipy.optimize import curve_fit # Peak fitting
from itertools import product

from . import functions_general as fg

"-----------------------Error Estimation Parameters-----------------------"
error_func = lambda x: np.sqrt(np.abs(x)+0.1) # Define how the error on each intensity is estimated
#error_func = rolling_fun
# error_func = lambda x: 0*x + 1 # Switch errors off


def ispeak(Y, dY=None, test=1, disp=False, return_rat=False):
    "Determines whether a peak exists in the given dataset"

    if dY is None:
        dY = error_func(Y)

    "From Blessing, J. Appl. Cryst. (1997). 30, 421-426"
    "EQU: (1) + (6)"
    " Background estimation added by me"
    s = np.mean(Y)
    bkg = np.min(Y)
    wi = 1 / dY ** 2
    signal = np.sum(wi * (Y - bkg)) / np.sum(wi)
    err = np.sqrt(len(Y) / np.sum(wi))

    # s = np.sum(Y)/len(Y)
    # h,bin = np.histogram(Y,10)
    # bkg = bin[np.argmax(h)]
    # signal = np.sum(Y-bkg)/len(Y)
    # srt = np.sort(Y)
    # err = 3*np.mean(np.abs(np.diff(srt[:len(Y)//2])))

    # sig=np.average(Y,weights=np.sqrt(np.abs(Y)))
    # err=np.average(np.sqrt(np.abs(Y)),weights=np.sqrt(np.abs(Y)))

    # sig = np.mean(Y)
    # err = np.mean(np.sqrt(np.abs(Y)))

    rat = signal / err
    # A peak exists if the signal/background ratio is greater than about 15
    if disp:
        print('avg: ', s)
        print('bkg: ', bkg)
        print('signal: ', signal)
        print('error: ', err)
        print('rat: ', rat)
    if return_rat:
        return rat
    return rat > test


def FWHM(x, y, interpolate=False):
    "Calculate a simple FWHM from a peak"

    if interpolate:
        interx = np.linspace(x[0], x[-1], len(x) * 100)
        intery = np.interp(interx, x, y)
        x, y = interx, intery

    mx = max(y)
    ln = len(y)

    # Peak position
    pkpos = y.argmax()

    # Split into two parts - before and after the peak
    hfxx1 = x[:pkpos + 1]
    hfxx2 = x[pkpos:]

    # Find the half-max positions
    hfmx1 = abs(y[:pkpos + 1] - mx // 2)
    hfmx2 = abs(y[pkpos:] - mx // 2)

    hfpos1 = hfxx1[hfmx1.argmin()]
    hfpos2 = hfxx2[hfmx2.argmin()]

    # Return FWHM
    return abs(hfpos2 - hfpos1)


def centre(x, y):
    "Calcualte centre of a peak"
    srt = np.argsort(y)
    cen = np.average(x[srt[-len(x) // 5:]], weights=y[srt[-len(x) // 5:]] ** 2)
    return cen


def straightline(x, grad=1.0, inter=0.0):
    "Staigh line"
    return grad * x + inter


def linefit(x, y, dy=None, disp=False):
    """
    Fit a line to data, y = mx + c

    fit,err = linefit(x,y,dy,disp=False)
    x,y = arrays of data to fit
    dy = error bars on each y value (or leave as None)
    disp = True/False display results

    fit/err = dicts of results with entries:
        'Gradient'    - the gradient of the line (m)
        'Intercept'   - the intercept (c)
        'x'           - x values, same as x
        'y'           - the fitted y values for each x

    Note: a matching line can be generated with:
        y = straightline(x,grad=fit['Gradient'],inter=fit['Intercept'])
    """

    # Set dy to 1 if not given
    if dy is None: dy = np.ones(len(y))

    # Remove zeros from x - causes errors in covariance matrix
    xold = x
    offset = 0.
    if any(np.abs(x) < 0.001):
        print('Zero detected - adding 0.001 to x values')
        offset = 0.001
        x = x + offset
    if any(np.isnan(dy)):
        print('Ignoring errors due to NaNs')
        dy = np.ones(len(y))

    # Handle zero intensities
    y[y < 0.01] = 0.01
    dy[dy < 0.01] = 0.01

    # Starting parameters
    grad = 0.0
    inter = np.mean(y)

    try:
        vals, covmat = curve_fit(straightline, x, y, [grad, inter], sigma=dy)
    except RuntimeError:
        vals = [0, 0]
        covmat = np.diag([np.nan, np.nan])

    # Values
    grad = vals[0]
    inter = vals[1]
    # Errors
    perr = np.sqrt(np.diag(covmat))
    dgrad = perr[0]
    dinter = perr[1]

    # Calculate fit
    yfit = straightline(xold, grad, inter)

    # Calculate CHI^2
    chi = np.sum((y - yfit) ** 2 / dy)
    dof = len(y) - 2  # Number of degrees of freedom (Nobs-Npar)
    chinfp = chi / dof

    fit, err = {}, {}
    fit['Gradient'] = grad
    fit['Intercept'] = inter
    fit['x'] = xold
    fit['y'] = yfit
    err['Gradient'] = dgrad
    err['Intercept'] = dinter

    # Print Results
    if disp:
        print(' ------Line Fit:----- ')
        print('  Gradient = {0:10.3G} +/- {1:10.3G}'.format(grad, dgrad))
        print(' Intercept = {0:10.3G} +/- {1:10.3G}'.format(inter, dinter))
        print('     CHI^2 = {0:10.3G}'.format(chi))
        print('  CHI^2 per free par = {0:10.3G}'.format(chinfp))
    return fit, err


def simpfit(x, y, disp=None):
    "Simple peak parameters"

    # Starting parameters
    wid = FWHM(x, y, interpolate=True)

    # bkgrgn = np.concatenate( (y[:len(x)//5],y[-len(x)//5:]) ) # background method 1 - wrong if peak is off centre
    # bkgrgn = np.percentile(y,range(0,20)) # background method 2 - average lowest 5th of data
    # bkg = np.mean(bkgrgn)
    h, bin = np.histogram(y, 10)
    bincen = (bin[1:] + bin[:-1]) / 2.0
    bkg = bincen[np.argmax(h)]
    amp = max(y) - bkg
    # if amp > 5*bkg:
    #    cen = x[y.argmax()]
    # else:
    #    cen = x[len(x)//2]
    # Alternative centre method 9/2/16
    # srt = np.argsort(y)
    # cen = np.average( x[ srt[ -len(x)//5: ] ] ,weights=y[ srt[ -len(x)//5: ] ])
    # Adapted 3/12/18, square weights to correct for long scans of sharp peaks.
    cen = centre(x, y)

    # Errors
    damp = np.sqrt(amp)
    dwid = abs(x[1] - x[0])
    # dbkg = np.sqrt(np.sum(bkgrgn**2))//len(bkgrgn)
    dbkg = np.sqrt(bkg)
    dcen = dwid

    # Integrated area
    scanwid = abs(x[-1] - x[0])
    ara = np.sum(y - bkg) * scanwid / len(x)
    dara = np.sqrt(np.sum(y)) * scanwid / len(x)

    # Print Results
    if disp is not None:
        print(' ------Simple Fit:----- ')
        print(' Amplitude = {0:10.3G} +/- {1:10.3G}'.format(amp, damp))
        print('    Centre = {0:10.3G} +/- {1:10.3G}'.format(cen, dcen))
        print('      FWHM = {0:10.3G} +/- {1:10.3G}'.format(wid, dwid))
        print('Background = {0:10.3G} +/- {1:10.3G}'.format(bkg, dbkg))
        print('      Area = {0:10.3G} +/- {1:10.3G}'.format(ara, dara))

    return amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara


def simpplt(x, height=1, cen=0, FWHM=0.5, bkg=0):
    "Plot an Illustration of simpfit"

    minpos = cen - FWHM
    maxpos = cen + FWHM
    y = np.ones(len(x)) * bkg
    y[len(x) // 5:-len(x) // 5] += height / 2
    y[np.logical_and(x > minpos, x < maxpos)] += height / 2
    return y


def gauss(x, height=1, cen=0, FWHM=0.5, bkg=0):
    "Define Gaussian"
    "From http://fityk.nieto.pl/model.html"
    return height * np.exp(-np.log(2) * ((x - cen) / (FWHM / 2)) ** 2) + bkg


def lorentz(x, height=1, cen=0, FWHM=0.5, bkg=0):
    "Define Lorentzian"
    "From http://fityk.nieto.pl/model.html"
    return height / (1 + ((x - cen) / (FWHM / 2)) ** 2) + bkg


def pvoight(x, height=1, cen=0, FWHM=0.5, LorFrac=0.5, bkg=0):
    "Define pseudo-Voight"
    "From http://fityk.nieto.pl/model.html"
    HWHM = FWHM / 2.0
    ln2 = 0.69314718055994529
    pos = x - cen
    L = LorFrac / (1 + (pos / HWHM) ** 2)
    G = (1 - LorFrac) * np.exp(-ln2 * (pos / HWHM) ** 2)
    return height * (G + L) + bkg


def create_peak_fun(text_fn, params):
    """
    Create a function from a string, return the function
     func = create_peak_fun(text_fn, params)

    text_fn = str function acting on variable 'x'
    params = list of variables in function other than 'x'

    e.g.
      func = create_peak_fun('x**2+y*z', ['y','z'])
    Returns func, which definition:
    def func(x,y,z):
        return x**2+y*z
    """
    inputs = ','.join(params)
    funcstr = 'def func(x,{}):\n    return {}'.format(inputs, text_fn)

    fitlocals = {}
    exec (funcstr, globals(), fitlocals)  # python >2.7.9
    func = fitlocals['func']
    return func


def peakfit(x, y, dy=None, type='pVoight', bkg_type='flat', peaktest=1, estvals=None,
            Nloop=10, Binit=1e-5, Tinc=2, change_factor=0.5, converge_max=100,
            min_change=0.01, interpolate=False, debug=False, plot=False, disp=False):
    """ General Peak Fitting function to fit a profile to a peak in y = f(x)
    Allows several possible profiles to be used and can try to find the best estimates for
    fitting parameters using an RMC-based least-squares routine.

    out,err = peakfit(x,y)
    out,err = peakfit(x,y,dy=None,type='pVoight',**fitoptions)

    **fitoptions:
    Basic parameters:
        x = array of the dependent variable, e.g. eta, mu, phi
        y = array of the independent variable, e.g. maxval,roi1_sum, APD
        dy = errors on y (default = None)
        type = function type. Allowed: 'pVoight' (default), 'Gauss', 'Lorentz', 'Simple'*
        bkg_type = background type. Allowed: 'flat' (default), 'slope', 'step'
    RMC options:
        Nloop = Number of iterations per temperature, default = 0 (RMC off)**
        Binit = Initial values of 1/kbT used for RMC, default = 1e-3 (lower = Higher temp)
        Tinc = After Nloop steps, the temperature is increased by factor Tinc, default = 2
        change_factor = Each parameter is multiplied by a normal distribution around 1 with width change_factor. (Default = 0.5)
        converge_max = Iterations will end when convergece reaches converge_max. (Default = 100)
        min_change = Relative variation of parameters must be below min_change to increase convergence value. (Default = 0.01)
    Output options:
        interpolate = True: The output fit will have interpolated (much finer) values in x and y. (Default = False)
        debug = True: Output of each iteration is displayed. (Default = False)
        disp = True: The final fitted parameters will be displayed in the command line. (Dafault = False)

    Output:
        out = dict with fitted parameters
        err = dict with errors on fitted paramters

        out.keys() = ['Peak Height','Peak Centre','FWHM','Lorz frac','Background','Area','CHI**2','CHI2 per dof','x','y']

    * selecting type='simple' will not fit the data, just provide a very simple estimation.
    ** Nloop must be set > 0 for the RMC routine to be used, for Nloop=0 or converge_max=0, a simple gradient decend method from a simple estimation is used.

    Notes on the RMC routine:
     - see the code

    """

    # Set dy to 1 if not given
    if dy is None: dy = np.ones(len(y))

    # Remove zeros from x - causes errors in covariance matrix
    xold = 1.0 * x
    offset = 0.
    if any(np.abs(x) < 0.0001):
        print('Zero detected - adding 0.0001 to x values')
        offset = 0.0001
        x = x + offset
    if any(np.isnan(dy)):
        print('Ignoring errors due to NaNs')
        dy = np.ones(len(y))

    # Handle zero intensities
    y[y < 0.01] = y[y < 0.01] + 0.01
    dy[dy < 0.01] = dy[dy < 0.01] + 0.01

    # Estimate starting parameters
    amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(x, y)
    frac = 0.5

    '-----------------------------------------------------------'
    '---------------------CHOOSE FUNCTIONS----------------------'
    '-----------------------------------------------------------'

    # Define background function
    if bkg_type.lower() in ['slope', 'sloping', 'grad']:
        bkgfunc = '+slope*(x-cen)'
        inpvals = ['slope']
        defestvals = [(y[-1] - y[0]) / (xold[-1] - xold[0])]
        valnames = ['Background Slope']
        minvals = [-np.inf]
        maxvals = [np.inf]
    elif bkg_type.lower() in ['step']:
        bkgfunc = '+np.append(bkg-step*np.ones(np.floor(len(x)/2.0)),bkg+step*np.ones(np.ceil(len(x)/2.0)))'
        inpvals = ['step']
        defestvals = [(y[-1] - y[0]) / 2.0]
        valnames = ['Background Step']
        minvals = [-np.inf]
        maxvals = [np.inf]
    else:
        bkgfunc = ''
        inpvals = []
        defestvals = []
        valnames = []
        minvals = []
        maxvals = []

    # Define starting parameters for choosen function
    if type.lower() in ['gauss', 'gaussian', 'g']:
        txtfunc = 'height*np.exp(-np.log(2)*((x-cen)/(FWHM/2))**2)+bkg' + bkgfunc
        inpvals = ['height', 'cen', 'FWHM', 'bkg'] + inpvals
        fitfunc = create_peak_fun(txtfunc, inpvals)
        defestvals = [amp, cen, wid, bkg] + defestvals
        valnames = ['Peak Height', 'Peak Centre', 'FWHM', 'Background'] + valnames
        minvals = [np.std(y), min(x), abs(x[1] - x[0]), -np.inf] + minvals
        maxvals = [5 * amp, max(x), 2 * (max(x) - min(x)), np.inf] + maxvals
    elif type.lower() in ['lorz', 'lorentz', 'lorentzian', 'l']:
        txtfunc = 'height/(1 + ((x-cen)/(FWHM/2))**2 )+bkg' + bkgfunc
        inpvals = ['height', 'cen', 'FWHM', 'bkg'] + inpvals
        fitfunc = create_peak_fun(txtfunc, inpvals)
        defestvals = [amp, cen, wid, bkg] + defestvals
        valnames = ['Peak Height', 'Peak Centre', 'FWHM', 'Background'] + valnames
        minvals = [0, min(x), abs(x[1] - x[0]), -np.inf] + minvals
        maxvals = [5 * amp, max(x), 2 * (max(x) - min(x)), np.inf] + maxvals
    elif type.lower() in ['simp', 'simple', 'basic', 's', 'sum', 'total', 'max', 'maxval', 'maximum']:
        fitfunc = simpplt
        defestvals = [amp, cen, wid, bkg]
        valnames = ['Peak Height', 'Peak Centre', 'FWHM', 'Background']
        minvals = [0, min(x), abs(x[1] - x[0]), -np.inf]
        maxvals = [5 * amp, max(x), 2 * (max(x) - min(x)), np.inf]
    else:
        txtfunc = 'height*( LorFrac/( 1.0 + (2.0*(x-cen)/FWHM)**2 ) + (1.0-LorFrac)*np.exp( -np.log(2)*(2.*(x-cen)/FWHM)**2 ) )+bkg' + bkgfunc
        inpvals = ['height', 'cen', 'FWHM', 'LorFrac', 'bkg'] + inpvals
        fitfunc = create_peak_fun(txtfunc, inpvals)
        defestvals = [amp, cen, wid, frac, bkg] + defestvals
        valnames = ['Peak Height', 'Peak Centre', 'FWHM', 'Lorz frac', 'Background'] + valnames
        minvals = [0, min(x), abs(x[1] - x[0]), -0.5, -np.inf] + minvals
        maxvals = [5 * amp, max(x), 2 * (max(x) - min(x)), 2, np.inf] + maxvals

    if estvals is None:
        estvals = defestvals[:]

    '-----------------------------------------------------------'
    '-------------------------FIT DATA--------------------------'
    '-----------------------------------------------------------'
    # Fitting not reuqired
    if type.lower() in ['simp', 'simple', 'basic', 's']:
        amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(xold, y)
        if ara < 0: ara = 0
        fitvals = [amp, cen, wid, bkg]
        errvals = [damp, dcen, dwid, dbkg]
        chi = 0

    elif type.lower() in ['sum', 'total']:
        amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(xold, y)
        ara = y.sum()
        dara = np.sqrt(ara)
        fitvals = [amp, cen, wid, bkg]
        errvals = [damp, dcen, dwid, dbkg]
        chi = 0

    elif type.lower() in ['max', 'maxval', 'maximum']:
        amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(xold, y)
        ara = y.max()
        dara = np.sqrt(ara)
        cen = xold[y.argmax()]
        fitvals = [amp, cen, wid, bkg]
        errvals = [damp, dcen, dwid, dbkg]
        chi = 0

    # Perform fitting
    else:
        # Check if a peak exists to fit
        peak_rat = ispeak(y, dy, test=peaktest, disp=False, return_rat=True)
        if debug: print('Peak ratio: {:1.2g} ({:1.2g})'.format(peak_rat, peaktest))
        if peak_rat < peaktest:
            if debug: print('No peak here (rat={:1.2g}). Fitting background instead!'.format(peak_rat))
            amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(xold, y)
            type = 'Background'
            fitfunc = straightline
            valnames = ['Slope', 'Background']
            estvals = [0, bkg]
            minvals = [-np.inf, -np.inf]
            maxvals = [np.inf, np.inf]

        # Perform fitting
        # Initial Fit (but don't update the estimators yet)
        try:
            fitvals, covmat = curve_fit(fitfunc, x, y, estvals, sigma=dy, absolute_sigma=True)
        except RuntimeError:
            if debug: print('Initial fit failed!')
            fitvals = 1 * estvals
            covmat = np.nan * np.eye(len(estvals))
        yfit = fitfunc(xold, *fitvals)  # New curve
        chi = np.sum((y - yfit) ** 2 / dy)  # Calculate CHI^2
        if debug: print('Initial Fit CHI**2 = ', chi)

        # Check errors are reasonable
        errvals = np.sqrt(np.diag(covmat))
        if any(np.isnan(errvals)):
            chi = np.inf

        if debug: print('Estimates: ', estvals)
        if debug: print('Initial Fit: ', list(fitvals), 'CHI**2 = ', chi)

        # Check new values are reasonable
        for n, val in enumerate(fitvals):
            if val < minvals[n] or val > maxvals[n]:
                if debug: print(
                'Initial value out of range: {} = {} ({}:{})'.format(valnames[n], val, minvals[n], maxvals[n]))
                chi = np.inf  # will not accept change if fitvalues fall out of range

        "----------------RMC-------------------"
        changes = np.zeros(len(estvals))
        converge = 0
        Ntemp = 0
        while converge < converge_max:
            beta = Binit * Tinc ** Ntemp
            if debug: print('New Temperature: ', Ntemp, beta)
            Ntemp += 1
            if Ntemp > Nloop:
                break
            for MCloop in range(Nloop):
                ini_estvals = 1 * estvals  # 1*estvals copies the array rather than links to it!
                if debug: print(Ntemp, MCloop, 'Current estimates: ', list(ini_estvals))
                # Loop over each estimator and randomly vary it
                for estn in range(len(estvals)):
                    inc_factor = np.random.normal(1, change_factor)
                    est_new = 1 * estvals
                    est_new[estn] = est_new[estn] * inc_factor
                    if debug: print('\tNew {} = {}'.format(valnames[estn], est_new[estn]))
                    try:
                        fitvals, covmat = curve_fit(fitfunc, x, y, est_new, sigma=dy, absolute_sigma=True)
                    except RuntimeError:
                        if debug: print(beta, MCloop, estn, 'Fit failed.')
                        continue
                    yfit = fitfunc(xold, *fitvals)  # New curve
                    chi_new = np.sum((y - yfit) ** 2 / dy)  # Calculate CHI^2

                    # Check errors are reasonable
                    errvals = np.sqrt(np.diag(covmat))
                    if any(np.isnan(errvals)) or any(np.isinf(errvals)):
                        chi_new = np.inf

                    # Check new values are reasonable
                    for n, val in enumerate(fitvals):
                        # if debug: print( beta,MCloop,estn,'CheckVal: ',n,val,minvals[n],maxvals[n] )
                        if val < minvals[n] or val > maxvals[n]:
                            if debug: print(
                            '\t\tValue out of range: {} = {} ({}:{})'.format(valnames[n], val, minvals[n], maxvals[n]))
                            chi_new = np.inf  # will not accept change if fitvalues fall out of range
                    if debug: print('\tFits: {}'.format(list(fitvals)))
                    if debug: print('\tErrors: {}'.format(list(errvals)))
                    if debug: print('\tCHI**2: {}'.format(chi_new))

                    # Metropolis Algorithm
                    if chi_new < chi or np.exp(beta * (chi - chi_new)) > np.random.rand():
                        if debug: print('\tFits Kept!')
                        estvals = 1 * fitvals  # = 1*est_new
                        chi = 1 * chi_new
                        changes[estn] += 1

                # Track changes
                chvals = np.divide(np.abs(np.subtract(estvals, ini_estvals)), ini_estvals)
                if np.any(chvals > min_change):
                    converge = 0
                else:
                    converge += 1

                if debug: print(beta, MCloop, chi, 'Changes: ', changes, chvals, converge)

                # break the loop if the solution has converged
                if converge >= converge_max:
                    if debug: print('Fit converged in {} temps!'.format(Ntemp - 1))
                    break

        # After the loop, perform a final check
        try:
            fitvals, covmat = curve_fit(fitfunc, x, y, estvals, sigma=dy, absolute_sigma=True)
        except RuntimeError:
            fitvals = 1 * estvals
            fitvals[0] = 0.0
            covmat = np.nan * np.eye(len(estvals))

        errvals = np.sqrt(np.diag(covmat))

    # Check fit has worked
    if any(np.isnan(errvals)) or chi == np.inf:
        print('Fit didnt work: use summation instead')
        amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(xold, y)
        if ara < 0: ara = 0
        type = 'Simple'
        fitfunc = simpplt
        valnames = ['Peak Height', 'Peak Centre', 'FWHM', 'Background']
        fitvals = [amp, cen, wid, bkg]
        errvals = [damp, dcen, dwid, dbkg]

    # create output dict
    output = dict(zip(valnames, fitvals))
    outerr = dict(zip(valnames, errvals))

    '-----------------------------------------------------------'
    '-------------Calulate area (profile dependent)-------------'
    '-----------------------------------------------------------'
    if type.lower() in ['gauss', 'gaussian', 'g']:
        output['Lorz frac'] = 0.0
        outerr['Lorz frac'] = 0.0
        output['Peak Height'] = abs(output['Peak Height'])
        output['FWHM'] = abs(output['FWHM'])
        amp, damp = abs(fitvals[0]), abs(errvals[0])
        wid, dwid = abs(fitvals[2]), abs(errvals[2])
        sig = wid / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
        dsig = dwid / ((2 * np.sqrt(2 * np.log(2))))
        ara = np.abs(amp * sig * np.sqrt(2 * np.pi))
        dara = ara * np.sqrt((damp / amp) ** 2 + (dsig / sig) ** 2)
    elif type.lower() in ['lorz', 'lorentz', 'lorentzian', 'l']:
        output['Lorz frac'] = 1.0
        outerr['Lorz frac'] = 0.0
        output['Peak Height'] = abs(output['Peak Height'])
        output['FWHM'] = abs(output['FWHM'])
        amp, damp = abs(fitvals[0]), abs(errvals[0])
        wid, dwid = abs(fitvals[2]), abs(errvals[2])
        ara = np.pi * amp * wid / 2
        dara = ara * np.sqrt((damp / amp) ** 2 + (dwid / wid) ** 2)
    elif type.lower() in ['simp', 'simple', 'basic', 's', 'max', 'maxval', 'maximum', 'sum', 'total']:
        output['Lorz frac'] = -1.0
        outerr['Lorz frac'] = 0.0
    elif type.lower() in ['voight', 'pvoight', 'pseudovoight', 'v']:
        output['Peak Height'] = abs(output['Peak Height'])
        output['FWHM'] = abs(output['FWHM'])
        amp, damp = abs(fitvals[0]), abs(errvals[0])
        wid, dwid = abs(fitvals[2]), abs(errvals[2])
        frac, dfrac = fitvals[3], errvals[3]

        # Calculated Voight area = Gaussian + Voight
        sig = wid / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
        dsig = dwid / ((2 * np.sqrt(2 * np.log(2))))
        Gara = np.abs(amp * sig * np.sqrt(2 * np.pi))
        Lara = np.pi * amp * wid / 2
        ara = frac * Lara + (1 - frac) * Gara

        # Error on area
        dGara = Gara * np.sqrt((damp / amp) ** 2 + (dsig / sig) ** 2)
        dLara = Lara * np.sqrt((damp / amp) ** 2 + (dwid / wid) ** 2)
        dVara1 = (1 - frac) * Gara * np.sqrt((dfrac / (1 - frac)) ** 2 + (dGara / Gara) ** 2)
        dVara2 = frac * Lara * np.sqrt((dfrac / frac) ** 2 + (dLara / Lara) ** 2)
        dara = np.sqrt(dVara1 ** 2 + dVara2 ** 2)
    elif type.lower() in ['background']:
        output['Lorz frac'] = np.nan
        outerr['Lorz frac'] = 0.0
        output['Peak Height'] = np.nan
        outerr['Peak Height'] = 0.0
        output['FWHM'] = np.nan
        outerr['FWHM'] = 0.0
        output['Peak Centre'] = np.nan
        outerr['Peak Centre'] = 0.0
        output['Background'] = fitfunc(xold[len(xold) // 2], *fitvals)
        outerr['Background'] = np.std(y)
        ara = 0.0
        ara = 0.0
    output['Area'] = ara
    outerr['Area'] = dara

    '-----------------------------------------------------'
    '----------------------Extra data---------------------'
    '-----------------------------------------------------'
    # Calculate fit
    if interpolate:
        xfit = np.linspace(min(xold), max(xold), 50 * len(xold))
    else:
        xfit = xold
    yfit = fitfunc(xfit, *fitvals)
    output['x'] = xfit
    output['y'] = yfit

    # Calculate CHI^2
    ycomp = fitfunc(xold, *fitvals)
    chi = np.sum((y - ycomp) ** 2 / dy ** 2)
    dof = len(y) - len(fitvals)  # Number of degrees of freedom (Nobs-Npar)
    chinfp = chi / dof
    output['CHI**2'] = chi
    output['CHI2 per dof'] = chinfp

    # Results String
    res_str = ' ------{} Fit:----- \n'.format(type)
    for estn in range(len(fitvals)):
        res_str += '{0:12s} = {1:20s}\n'.format(valnames[estn], fg.stfm(fitvals[estn], errvals[estn]))
    res_str += '        Area = {0:20s}\n'.format(fg.stfm(ara, dara))
    output['Results'] = res_str

    # Print Results
    if disp:
        res_str += '       CHI^2 = {0:10.8G}\n'.format(chi)
        res_str += 'CHI^2 per free par = {0:10.3G}\n'.format(chinfp)
        print(res_str)

    # Plot Results
    if plot:
        plt.figure()
        plt.errorbar(x, y, dy, fmt='b-o', lw=2, label='Data')
        plt.plot(xfit, yfit, 'r-', lw=2, label='Fit')
        plt.legend()
        plt.show()
    return output, outerr


def fittest(x, y, dy=None, tryall=False, disp=False):
    """
    Attempt multiple fit types and return the best
    """

    # Set dy to 1 if not given
    if dy is None: dy = np.ones(len(y))

    # Remove zeros from x - causes errors in covariance matrix
    xold = x
    offset = 0.
    if any(np.abs(x) < 0.001):
        print('Zero detected - adding 0.001 to x values')
        offset = 0.001
        x = x + offset
    if any(np.isnan(dy)):
        print('Ignoring errors due to NaNs')
        dy = np.ones(len(y))

    # Handle zero intensities
    y[y < 0.01] = y[y < 0.01] + 0.01
    dy[dy < 0.01] = dy[dy < 0.01] + 0.01

    # Estimate starting parameters
    amp, cen, wid, bkg, ara, damp, dcen, dwid, dbkg, dara = simpfit(x, y)

    # Define functions and starting paramters
    fitname = ['Line', 'Gaussian', 'Lorentzian', 'pVoight']
    fitfuncs = [straightline, gauss, lorentz, pvoight]
    estvals = [[0, bkg],
               [amp, cen, wid, bkg],
               [amp, cen, wid, bkg],
               [amp, cen, wid, 0.5, bkg]]
    valnames = [['Gradient', 'Intercept'],
                ['Peak Height', 'Peak Centre', 'FWHM', 'Background'],
                ['Peak Height', 'Peak Centre', 'FWHM', 'Background'],
                ['Peak Height', 'Peak Centre', 'FWHM', 'Lorz frac', 'Background']]
    trialvals = [[np.arange(-2, 2, 0.5), np.linspace(min(y), max(y), 5)],
                 [np.linspace(min(y), max(y), 5), np.linspace(min(x), max(x), 5),
                  np.linspace(x[1] - x[0], x[-1] - x[0], 5), np.linspace(min(y), max(y), 5)],
                 [np.linspace(min(y), max(y), 5), np.linspace(min(x), max(x), 5),
                  np.linspace(x[1] - x[0], x[-1] - x[0], 5), np.linspace(min(y), max(y), 5)],
                 [np.linspace(min(y), max(y), 5), np.linspace(min(x), max(x), 5),
                  np.linspace(x[1] - x[0], x[-1] - x[0], 5), np.linspace(0, 1, 5), np.linspace(min(y), max(y), 5)]]
    minvals = [[-np.inf, 0],
               [0, min(x), x[1] - x[0], -np.inf],
               [0, min(x), x[1] - x[0], -np.inf],
               [0, min(x), x[1] - x[0], -np.inf, -np.inf]]
    maxvals = [[np.inf, np.inf],
               [np.inf, max(x), 5 * (x[-1] - x[0]), np.inf],
               [np.inf, max(x), 5 * (x[-1] - x[0]), np.inf],
               [np.inf, max(x), 5 * (x[-1] - x[0]), np.inf, np.inf]]
    chival = np.zeros(len(fitfuncs))

    # Loop through each function and determine the best fit
    for n in range(len(fitfuncs)):
        if tryall:
            # Attemp every compbination of trial values VERY SLOW!!!
            trials = product(*trialvals[n])  # from itertools
            chival[n] = np.inf
            fitvals = np.inf * np.ones(len(valnames[n]))
            errvals = np.inf * np.ones(len(valnames[n]))
            for tt in trials:
                # attempt the fit
                try:
                    tfitvals, tcovmat = curve_fit(fitfuncs[n], x, y, tt, sigma=dy)
                except RuntimeError:
                    continue

                # Check fit is within the allowed range
                for v in range(len(valnames[n])):
                    if tfitvals[v] < minvals[n][v] or tfitvals[v] > maxvals[n][v]:
                        continue

                yfit = fitfuncs[n](xold, *tfitvals)  # New curve
                newchi = np.sum((y - yfit) ** 2 / dy)  # Calculate CHI^2
                if newchi < chival[n]:
                    fitvals = tfitvals
                    errvals = np.sqrt(np.diag(tcovmat))
                    chival[n] = 1 * newchi
        else:
            # attempt the fit
            try:
                fitvals, covmat = curve_fit(fitfuncs[n], x, y, estvals[n], sigma=dy)
            except RuntimeError:
                fitvals = 1 * estvals[n]
                covmat = np.nan * np.eye(len(estvals[n]))

            errvals = np.sqrt(np.diag(covmat))

            yfit = fitfuncs[n](xold, *fitvals)  # New curve
            chival[n] = np.sum((y - yfit) ** 2 / dy)  # Calculate CHI^2

            # Check fit is within the allowed range
            for v in range(len(valnames[n])):
                if fitvals[v] < minvals[n][v] or fitvals[v] > maxvals[n][v]:
                    valnames[n][v] += '*'
                    chival[n] = np.inf

        if disp:
            print('----{}: {}----'.format(n, fitname[n]))
            for v in range(len(valnames[n])):
                print('{0:10s} = {1:10.3G} +/- {2:10.3G}'.format(valnames[n][v], fitvals[v], errvals[v]))
            # print( '      Area = {0:10.3G} +/- {1:10.3G}'.format(ara,dara) )
            print('     CHI^2 = {0:10.8G}'.format(chival[n]))

    # Find the minimum chi
    minval = np.argmin(chival)
    return fitname[minval]


def gauss2D(XY, height=1, cen_x=0, cen_y=0, FWHM_x=.5, FWHM_y=.5, bkg=0):
    "Define 2D Gaussian"
    X, Y = XY
    G = height * np.exp(-np.log(2) * (((X - cen_x) / (FWHM_x / 2.0)) ** 2 + ((Y - cen_y) / (FWHM_y / 2.0)) ** 2)) + bkg
    return G.ravel()


def orderpar(x, Tc=100, beta=0.5, amp=1):
    "Generate an order paramter"
    # op = amp*np.real(np.power(np.complex(Tc-x),beta))
    op = amp * np.power(Tc - x, beta)
    op[np.isnan(op)] = 0.0
    return op


def orderparfit(x, y, dy=None, Tc=None, disp=None):
    "Fit an order parameter to a temperature dependence y = f(T)"

    # Set dy to 1 if not given
    if dy is None: dy = np.ones(len(y))

    # Remove zeros from x - causes errors in covariance matrix
    xold = x
    offset = 0.
    if any(np.abs(x) < 0.001):
        print('Zero detected - adding 0.001 to x values')
        offset = 0.001
        x = x + offset
    if any(np.isnan(dy)):
        print('Ignoring errors due to NaNs')
        dy = np.ones(len(y))

    # Handle zero intensities
    y[y < 0.01] = 0.01
    dy[dy < 0.01] = 0.01

    # Starting parameters
    if Tc is None:
        Tc = x[len(x) // 2]
    beta = 0.5
    amp = np.mean(y[:len(y) // 10])
    print(Tc, beta, amp)

    try:
        vals, covmat = curve_fit(orderpar, x, y, [Tc, beta, amp], sigma=dy)
    except RuntimeError:
        vals = [0, beta, amp]
        covmat = np.diag([np.nan, np.nan, np.nan])
    # Values
    Tc = vals[0] - offset
    beta = vals[1]
    amp = vals[2]
    # Errors
    perr = np.sqrt(np.diag(covmat))
    dTc = perr[0]
    dbeta = perr[1]
    damp = perr[2]

    # Calculate fit
    yfit = orderpar(xold, Tc, beta, amp)

    # Calculate CHI^2
    chi = np.sum((y - yfit) ** 2 / dy)
    dof = len(y) - 4  # Number of degrees of freedom (Nobs-Npar)
    chinfp = chi / dof

    # Check fit has worked
    if Tc <= 0 or any(np.isnan([dTc, dbeta, damp])):
        print('Fit didn''t work: oh dear')
        return

    # Print Results
    if disp:
        print(' ------Order Parameter Fit:----- ')
        print('        Tc = {0:10.3G} +/- {1:10.3G}'.format(Tc, dTc))
        print('      Beta = {0:10.3G} +/- {1:10.3G}'.format(beta, dbeta))
        print('       Amp = {0:10.3G} +/- {1:10.3G}'.format(amp, damp))
        print('     CHI^2 = {0:10.3G}'.format(chi))
        print('  CHI^2 per free par = {0:10.3G}'.format(chinfp))
    return Tc, beta, amp, dTc, dbeta, damp, yfit