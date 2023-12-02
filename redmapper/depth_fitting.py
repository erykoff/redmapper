"""Classes and routines for simple fits to galaxy catalog depth.

"""
import fitsio
import numpy as np
import esutil
import scipy.optimize


class DepthFunction(object):
    """
    Class to implement function for fitting depth.

    """
    def __init__(self,mag,magErr,zp,nSig):
        """
        Instantiate DepthFunction class.

        Parameters
        ----------
        mag: `np.array`
           Float array of magnitudes
        magErr: `np.array`
           Float array of magnitude errors
        zp: `float`
           Reference zeropoint
        nSig: `float`
           Number of sigma to compute depth limit
        """
        self.const = 2.5/np.log(10.0)

        self.mag = mag
        self.magErr = magErr
        self.zp = zp
        self.nSig = nSig

        self.max_p1 = 1e10

    def __call__(self, x):
        """
        Compute total cost function for f(x)

        Parameters
        ----------
        x: `np.array`, length 2
           Float array of fit parameters

        Returns
        -------
        t: `float`
           Total cost function at parameters x
        """

        if ((x[1] < 0.0) or
            (x[1] > self.max_p1)):
            return 1e10

        f1lim = 10.**((x[0] - self.zp)/(-2.5))
        fsky1 = ((f1lim**2. * x[1])/(self.nSig**2.) - f1lim)

        if (fsky1 < 0.0):
            return 1e10

        tflux = x[1]*10.**((self.mag - self.zp)/(-2.5))
        err = np.sqrt(fsky1*x[1] + tflux)

        # apply the constant here, not to the magErr, which was dumb
        t=np.sum(np.abs(self.const*(err/tflux) - self.magErr))

        if not np.isfinite(t):
            t=1e10

        return t

# FIXME: want to be able to reuse code from utilities!
def applyErrorModel(pars, magIn, noNoise=False, lnscat=None):
    """
    Apply error model to set of magnitudes

    Parameters
    ----------
    pars: `np.ndarray`
       Error parameter structure
    magIn: `np.array`
       Float array with input magnitudes
    noNoise: `bool`, optional
       Do not apply noise?  Default is False
    lnscat: `float`, optional
       Additional log-scatter.  Default is None.

    Returns
    -------
    mag: `np.array`
       Float array of magnitudes
    magErr: `np.array`
       Float array of magnitude errors
    """

    tFlux = pars['EXPTIME'][0]*10.**((magIn - pars['ZP'][0])/(-2.5))
    noise = np.sqrt(pars['FSKY1'][0]*pars['EXPTIME'][0] + tFlux)

    if lnscat is not None:
        noise = np.exp(np.log(noise) + lnscat * np.random.normal(size=noise.size))

    if (noNoise):
        flux = tFlux
    else:
        flux = tFlux + noise*np.random.normal(magIn.size)

    # Straight magnitudes
    mag = pars['ZP'][0] - 2.5*np.log10(flux/pars['EXPTIME'][0])
    magErr = (2.5/np.log(10.)) * (noise/flux)

    return mag, magErr


def calcErrorModel(_mag, _magErr, nSig=10.0, doPlot=False, nTrial=100, calcErr=False,
                   useBoot=False, snCut=5.0, zp=22.5, oldIDL=False):
    """
    Caluclate the error model for a given list of magnitudes and errors

    Parameters
    ----------
    _mag: `np.array`
       Float array of input magnitudes
    _magErr: `np.array`
       Float array of input magnitude errors
    nSig: `float`, optional
       Number of sigma to compute maglim.  Default is 10.0
    doPlot: `bool`, optional
       Plot results.  Default is False.
    nTrial: `int`, optional
       Number of trials for bootstrap errors.  Default is 100.
    calcErr: `bool`, optional
       Calculate parameter errors?  Default is False.
    useBoot: `bool`, optional
       Use bootstrap error estimation?  Default is False.
    snCut: `float`, optional
       Minimum signal/noise to use in the fit.  Default is 5.0
    zp: `float`, optional
       Default reference zeropoint.  Default is 22.5.
    oldIDL: `bool`, optional
       Use older (worse) IDL compatibility mode.  Default is False.

    Returns
    -------
    pars: `np.ndarray`
       Error model parameters
    val: `int`
       0 for success.  Alyways 0.
    fig: `matplotlib.Figure`, if doPlot is True
    ax: `matplotlib.Axis`, if doPlot is True
    """
    const = 2.5/np.log(10.)

    # first need to filter out really bad ones
    ok,=np.where((np.isfinite(_mag)) &
                 (np.isfinite(_magErr)) &
                 (_magErr > 0.0))

    mag=_mag[ok]
    magErr=_magErr[ok]

    if oldIDL:
        # old IDL version...
        st=np.argsort(mag)
        gd,=np.where(mag < mag[st[np.int32(0.98*mag.size)]])
    else:
        # new python
        st=np.argsort(mag)
        gd,=np.where((mag < mag[st[np.int32(0.98*mag.size)]]) &
                     (magErr < const / snCut))

    if (gd.size == 0):
        if (doPlot):
            return (-1,1,None,None)
        else:
            return (-1,1)

    # extra const here?
    dFunc = DepthFunction(mag[gd], magErr[gd], zp, nSig)

    # get the reference limiting mag
    test,=np.where((magErr[gd] > const/nSig) &
                   (magErr[gd] < 1.1*const/nSig))

    if (test.size >= 3):
        limmagStart = np.median(mag[gd[test]])
    else:
        # I don't like this alternative
        st=np.argsort(mag[gd])
        limmagStart = mag[gd[st[np.int32(0.95*st.size)]]]

    # figure out where to start the effective exposure time
    # go from 1.0 to 10000, logarithmically...
    # note that this range (from IDL code) works for zp=22.5.
    #  For other zps, need to rescale here

    expRange=np.array([1.0,10000.])/(10.**((zp - 22.5)/2.5))
    nSteps=20

    binSize=(np.log(expRange[1])-np.log(expRange[0]))/(nSteps-1)

    expTimes=np.exp(np.arange(nSteps)*binSize)*expRange[0]

    tTest=np.zeros(nSteps)
    for i,expTime in enumerate(expTimes):
        # call a function...
        dFunc.max_p1 = expTime*2.
        tTest[i] = dFunc([limmagStart, expTime])

    ind = np.argmin(tTest)

    p0=np.array([limmagStart, expTimes[ind]])

    # try single fit
    dFunc.max_p1 = 10.0*p0[1]

    ret = scipy.optimize.fmin(dFunc, p0,disp=False, full_output=True,retall=False)

    # check for convergence here...
    if (ret[-1] > 0):
        # could not converge
        if (doPlot):
            return (-1,1,None,None)
        else:
            return (-1,1)

    p = ret[0]

    pars=np.zeros(1,dtype=[('EXPTIME','f4'),
                           ('ZP','f4'),
                           ('LIMMAG','f4'),
                           ('NSIG','f4'),
                           ('FLUX1_LIM','f4'),
                           ('FSKY1','f4'),
                           ('LIMMAG_ERR','f4'),
                           ('EXPTIME_ERR','f4'),
                           ('FRAC_OUT','f4')])
    pars['EXPTIME'] = p[1]
    pars['ZP'] = dFunc.zp
    pars['LIMMAG'] = p[0]
    pars['NSIG'] = dFunc.nSig
    pars['FLUX1_LIM'] = 10.**((p[0] - dFunc.zp)/(-2.5))
    pars['FSKY1'] = (pars['FLUX1_LIM'][0]**2.*p[1])/(dFunc.nSig**2.) - pars['FLUX1_LIM'][0]

    # compute frac_out, the fraction of outliers
    testMag, testMagErr = applyErrorModel(pars, dFunc.mag, noNoise=True)

    out,=np.where(np.abs(testMagErr - dFunc.magErr) > 0.005)
    pars['FRAC_OUT'] = np.float64(out.size)/np.float64(gd.size)


    if (calcErr):
        limMags=np.zeros(nTrial,dtype=np.float32)
        expTimes=np.zeros_like(limMags)

        p0=p.copy()

        for i in range(nTrial):
            r=np.int32(np.random.random(gd.size)*gd.size)
            dFunc.mag = mag[gd[r]]
            dFunc.magErr = magErr[gd[r]]

            ret = scipy.optimize.fmin(dFunc, p0, disp=False, full_output=True,retall=False)
            if (ret[4] > 0) :
                p = p0
            else:
                p = ret[0]

            limMags[i] = p[0]
            expTimes[i] = p[1]

        # use IQD for errors
        st=np.argsort(limMags)
        pars['LIMMAG_ERR'] = (limMags[st[np.int32(0.75*nTrial)]] - limMags[st[np.int32(0.25*nTrial)]])/1.35
        st=np.argsort(expTimes)
        pars['EXPTIME_ERR'] = (expTimes[st[np.int32(0.75*nTrial)]] - expTimes[st[np.int32(0.25*nTrial)]])/1.35
        if (useBoot):
            pars['LIMMAG'] = np.median(limMags)
            pars['EXPTIME'] = np.median(expTimes)

    if (doPlot):
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(111)

        st=np.argsort(testMag)
        if (not calcErr):
            ax.plot(testMag[st], testMagErr[st], 'k-')
        else:
            testPars = pars.copy()
            alphaColor = np.zeros(4)
            alphaColor[0:3] = 0.5
            alphaColor[3] = 0.5
            for i in range(nTrial):
                testPars['LIMMAG'] = limMags[i]
                testPars['EXPTIME'] = expTimes[i]
                testPars['FLUX1_LIM'] = 10.**((limMags[i] - dFunc.zp)/(-2.5))
                testPars['FSKY1'] = (testPars['FLUX1_LIM'][0]**2.*expTimes[i])/(dFunc.nSig**2.) - testPars['FLUX1_LIM'][0]
                mTest, mErrTest = applyErrorModel(testPars, testMag[st], noNoise=True)
                ax.plot(mTest, mErrTest, '-',color=alphaColor)

            ax.plot(testMag[st],testMagErr[st],'c--')


        ax.plot(mag[gd], magErr[gd],'r.')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot([pars['LIMMAG'][0],pars['LIMMAG'][0]],[0,1],'k--')
        ax.plot([0,100],[1.086/nSig,1.086/nSig],'k--')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return (pars,0,fig,ax)
    else:
        # success
        return (pars,0)

class DepthLim(object):
    """
    Class to compute depth limits from data, if external map is not available.

    This class is used to compute depth in realtime from data.
    """
    def __init__(self, mag, mag_err, max_gals=100000):
        """
        Instantiate DepthLim object.

        Upon initialization this will compute default global parameters that
        will be used if a fit cannot be performed.

        Parameters
        ----------
        mag: `np.array`
           Float array of magnitudes from a large region of sky
        mag_err: `np.array`
           Float array of magnitude errors from a large region of sky
        max_gals: `int`
           Maximum number of galaxies to sample to get global default fit.

        Raises
        ------
        RuntimeError:
           If a global fit cannot be performed.
        """

        # This gets a global fit, to use as a fallback

        if mag.size < max_gals:
            use = np.arange(mag.size)
        else:
            use = np.random.choice(np.arange(mag.size), size=max_gals, replace=False)

        self.initpars, fail = calcErrorModel(mag[use], mag_err[use], calcErr=False)

        if fail:
            raise RuntimeError("Complete failure on getting limiting mag fit")

    def calc_maskdepth(self, maskgals, mag, mag_err):
        """
        Calculate mask depth empirically for a set of galaxies.

        This will modify maskgals.limmag, maskgals.exptime, maskgals.zp,
        maskgals.nsig to the fit values (or global if fit cannot be performed).

        Parameters
        ----------
        maskgals: `redmapper.Catalog`
           maskgals catalog
        mag: `np.array`
           Float array of local magnitudes
        mag_err: `np.array`
           Float array of local magnitude errors
        """

        limpars, fail = calcErrorModel(mag, mag_err, calcErr=False)

        if fail:
            maskgals.limmag[:] = self.initpars['LIMMAG']
            maskgals.exptime[:] = self.initpars['EXPTIME']
            maskgals.zp[0] = self.initpars['ZP']
            maskgals.nsig[0] = self.initpars['NSIG']

        else:
            maskgals.limmag[:] = limpars['LIMMAG']
            maskgals.exptime[:] = limpars['EXPTIME']
            maskgals.zp[0] = limpars['ZP'][0]
            maskgals.nsig[0] = limpars['NSIG'][0]

        return

