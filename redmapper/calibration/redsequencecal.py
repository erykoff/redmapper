from __future__ import division, absolute_import, print_function

import os
import numpy as np
import fitsio
import time
from scipy.optimize import least_squares

from ..configuration import Configuration
from ..fitters import MedZFitter, RedSequenceFitter, RedSequenceOffDiagonalFitter, CorrectionFitter
from ..redsequence import RedSequenceColorPar
from ..color_background import ColorBackground
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry
from ..utilities import make_nodes, CubicSpline, interpol

class RedSequenceCalibrator(object):
    """
    """

    def __init__(self, conf, galfile):
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self._galfile = galfile

    def run(self):
        """
        """

        gals = GalaxyCatalog.from_galfile(self._galfile)

        if self.config.calib_use_pcol:
            use, = np.where((gals.z > self.config.zrange[0]) &
                            (gals.z < self.config.zrange[1]) &
                            (gals.pcol > self.config.calib_pcut))
        else:
            use, = np.where((gals.z > self.config.zrange[0]) &
                            (gals.z < self.config.zrange[1]) &
                            (gals.pmem > self.config.calib_pcut))

        gals = gals[use]

        nmag = self.config.nmag
        ncol = nmag - 1

        # Reference mag nodes for pivot
        pivotnodes = make_nodes(self.config.zrange, self.config.calib_pivotmag_nodesize)

        # Covmat nodes
        covmatnodes = make_nodes(self.config.zrange, self.config.calib_covmat_nodesize)

        # correction nodes
        corrnodes = make_nodes(self.config.zrange, self.config.calib_corr_nodesize)

        # correction slope nodes
        corrslopenodes = make_nodes(self.config.zrange, self.config.calib_corr_slope_nodesize)

        # volume factor (hard coded)
        volnodes = make_nodes(self.config.zrange, 0.01)

        # Start building the par dtype
        dtype = [('pivotmag_z', 'f4', pivotnodes.size),
                 ('pivotmag', 'f4', pivotnodes.size),
                 ('minrefmag', 'f4', pivotnodes.size),
                 ('maxrefmag', 'f4', pivotnodes.size),
                 ('medcol', 'f4', (pivotnodes.size, ncol)),
                 ('medcol_width', 'f4', (pivotnodes.size, ncol)),
                 ('covmat_z', 'f4', covmatnodes.size),
                 ('sigma', 'f4', (ncol, ncol, covmatnodes.size)),
                 ('covmat_amp', 'f4', (ncol, ncol, covmatnodes.size)),
                 ('covmat_slope', 'f4', (ncol, ncol, covmatnodes.size)),
                 ('corr_z', 'f4', corrnodes.size),
                 ('corr', 'f4', corrnodes.size),
                 ('corr_slope_z', 'f4', corrslopenodes.size),
                 ('corr_slope', 'f4', corrslopenodes.size),
                 ('corr_r', 'f4', corrslopenodes.size),
                 ('corr2', 'f4', corrnodes.size),
                 ('corr2_slope', 'f4', corrslopenodes.size),
                 ('corr2_r', 'f4', corrslopenodes.size),
                 ('volume_factor_z', 'f4', volnodes.size),
                 ('volume_factor', 'f4', volnodes.size)]

        # And for each color, make the nodes
        node_dict = {}
        self.ztag = [None] * ncol
        self.ctag = [None] * ncol
        self.zstag = [None] * ncol
        self.stag = [None] * ncol
        for j in xrange(ncol):
            self.ztag[j] = 'z%02d' % (j)
            self.ctag[j] = 'c%02d' % (j)
            self.zstag[j] = 'zs%02d' % (j)
            self.stag[j] = 'slope%02d' % (j)

            node_dict[self.ztag[j]] = make_nodes(self.config.zrange, self.config.calib_color_nodesizes[j],
                                                 maxnode=self.config.calib_color_maxnodes[j])
            node_dict[self.zstag[j]] = make_nodes(self.config.zrane, self.config.calib_slope_nodesizes[j],
                                                  maxnode=self.config.calib_color_maxnodes[j])

            dtype.extend([(self.ztag[j], 'f4', node_dict[self.ztag[j]].size),
                          (self.ctag[j], 'f4', node_dict[self.ztag[j]].size),
                          (self.zstag[j], 'f4', node_dict[self.zstag[j]].size),
                          (self.stag[j], 'f4', node_dict[self.stag[j]].size)])

        # Make the pars ... and fill them with the defaults
        self.pars = Entry(np.zeros(1, dtype=dtype))

        self.pars.pivotmag_z = pivotnodes
        self.pars.covmat_z = covmatnodes
        self.pars.corr_z = corrnodes
        self.pars.corr_slope_z = corrslopenodes
        self.pars.volume_factor_z = volnodes

        for j in xrange(ncol):
            self.pars._ndarray[self.ztag[j]] = node_dict[self.ztag[j]]
            self.pars._ndarray[self.zstag[j]] = node_dict[self.zstag[j]]

        # And a special subset of color galaxies
        if self.config.calib_use_pcol:
            coluse, = np.where(gals.pcol > self.config.calib_color_pcut)
        else:
            coluse, = np.where(gals.pmem > self.config.calib_color_pcut)

        colgals = gals[coluse]

        # And a placeholder zredstr which allows us to do stuff
        self.zredstr = RedSequenceColorPar(None, config=self.config)

        # And read the color background
        self.bkg = ColorBackground(self.config.bkgfile_color)

        # And prepare for luptitude corrections
        if self.config.b[0] = 0.0:
            self.do_lupcorr = True
            self.bnmgy = self.config.b * 1e9
            self.lupzp = 22.5
        else:
            self.do_lupcorr = False

        # Compute pivotmags
        self._calc_pivotmags(colgals)

        # Compute median colors
        self._calc_medcols(colgals)

        # Compute diagonal parameters
        self._calc_diagonal_pars(gals)

        # Compute off-diagonal parameters
        self._calc_offdiagonal_pars(gals)

        # Compute volume factor
        self._calc_volume_factor()

        # Write out the parameter file
        self.save_pars(self.config.parfile, clobber=False)

        # Compute zreds without corrections
        # Later will want this parallelized, I think
        self._calc_zreds(gals, do_correction=False)

        # Compute correction (mode1)
        self._calc_correction(gals)

        # Compute correction (mode2)
        self._calc_correction(gals, mode2=True)

        # And re-save the parameter file
        self.save_pars(self.config.parfile, clobber=True)

        # Recompute zreds with corrections
        # Later will want this parallelized, I think
        self._calc_zreds(gals, do_correction=True)

        # Make diagnostic plots
        self._make_diagnostic_plots(gals)

    def _compute_startvals(self, nodes, z, val, err=None, median=False, fit=False, mincomp=3):
        """
        """

        def _linfunc(p, x, y):
            return (p[1] + p[0] * x) - y

        if (not median and not fit) or (median and fit):
            raise RuntimeError("Must select one and only one of median and fit")

        if median:
            mvals = np.zeros(nodes.size)
            scvals = np.zeros(nodes.size)
        else:
            cvals = np.zeros(nodes.size)
            svals = np.zeros(nodes.size)

        if err is not None:
            if err.size != val.size:
                raise ValueError("val and err must be the same length")

            # default all to 0.1
            evals = np.zeros(nodes.size) + 0.1
        else:
            evals = None

        for i in xrange(nodes.size):
            if i == 0:
                zlo = nodes[0]
            else:
                zlo = (nodes[i - 1] + nodes[i]) / 2.
            if i == nodes.size - 1:
                zhi = nodes[i]
            else:
                zhi = (nodes[i] + nodes[i + 1]) / 2.

            u, = np.where((z > zlo) & (z < zhi))

            if u.size < mincomp:
                if i > 0:
                    if median:
                        mvals[i] = mvals[i - 1]
                        scvals[i] = scvals[i - 1]
                    else:
                        cvals[i] = cvals[i - 1]
                        svals[i] = svals[i - 1]

                    if err is not None:
                        evals[i] = evals[i - 1]
            else:
                if median:
                    mvals[i] = np.median(val[u])
                    scvals[i] = np.median(np.abs(val[u] - mvals[i]))
                else:
                    fit = least_squares(_linfunc, [0.0, 0.0],  loss='soft_l1', args=(z[u], val[u]))
                    cvals[i] = fit[1]
                    svals[i] = np.clip(fit[0], None, 0.0)

                if err is not None:
                    evals[i] = np.median(err[u])

        if median:
            return mvals, scvals
        else:
            return cvals, svals, evals

    def _compute_single_lupcorr(self, cvals, svals, gals, dmags, mags, lups, mind, sign):
        spl = CubicSpline(self.pars._ndarray[self.ztag[j]], cvals)
        cv = spl(gals.z)
        spl = CubicSpline(self.pars._ndarray[self.zstag[j]], svals)
        sv = spl(gals.z)

        mags[:, mind] = mags[:, mind + sign] + sign * (cv + sv * dmags)

        flux = 10.**((mags[:, mind] - self.lupzp) / (-2.5))
        lups[:, mind] = 2.5 * np.log10(1.0 / self.config.b[mind]) - np.asinh(0.5 * flux / self.bnmgy[mind]) / (0.4 * np.log(10.0))

        magcol = mags[:, j] - mags[:, j + 1]
        lupcol = lups[:, j] - lups[:, j + 1]

        lupcorr = lupcol - magcol

        return lupcorr

    def _calc_pivotmags(self, gals):
        """
        """

        print("Calculating pivot magnitudes...")

        # With binning, approximate the positions for starting the fit
        pivmags = np.zeros_like(self.pars.pivotmag_z)

        for i in xrange(rvals.size):
            pivmags, _ = self._compute_startvals(self.pars.pivotmag_z, gals.z, gals.refmag, median=True)

        medfitter = MedZFitter(self.pars.pivotmag_z, gals.z, gals.refmag)
        pivmags = medfitter.fit(pivmags)

        self.pars.pivotmag = pivmags

        # and min and max...
        self.pars.minrefmag = self.zredstr.mstar(self.pars.pivotmag_z) - 2.5 * np.log10(30.0)
        lval_min = np.clip(self.config.lval_reference - 0.1, 0.001, None)
        self.pars.maxrefmag = self.zredstr.mstar(self.pars.pivotmag_z) - 2.5 * np.log10(lval_min)

    def _calc_medcols(self, gals):
        """
        """

        print("Calculating median colors...")

        ncol = self.config.nmag - 1

        galcolor = gals.galcol

        for j in xrange(ncol):
            col = galcolor[:, j]

            # get the start values
            mvals, scvals = self._compute_startvals(self.pars.pivotmag_z, gals.z, col, median=True)

            # compute the median
            medfitter = MedZFitter(self.pars.pivotmag_z, gals.z, col)
            mvals = medfitter.fit(mvals)

            # and the scatter
            spl = CubicSpline(self.pars.pivotmag_z, mvals)
            med = spl(gals.z)
            medfitter = MedZFitter(self.pars.pivotmag_z, gals.z, np.abs(col - med))
            scvals = medfitter.fit(scvals)

            self.pars.medcol[:, j] = mvals
            self.pars.medcol_width[:, j] = 1.4826 * scvals

    def _calc_diagonal_pars(self, gals):
        """
        """

        # The main routine to compute the red sequence on the diagonal

        ncol = self.config.nmag - 1

        galcolor = gals.galcol
        galcolor_err = gals.galcol_err

        # compute the pivot mags
        spl = CubicSpline(self.pars.pivotmag_z, self.pars.pivotmag)
        pivotmags = spl(gals.z)

        # And set the right probabilities
        if self.config.calib_use_pcol:
            probs = gals.pcol
        else:
            probs = gals.pmem

        # Figure out the order of the colors for luptitude corrections
        mags = np.zeros((gals.size, self.config.nmag))

        if self.do_lupcorr:
            col_indices = np.zeros(ncol, dtype=np.int32)
            sign_indices = np.zeros(ncol, dtype=np.int32)
            mind_indices = np.zeros(ncol, dtype=np.int32)

            c=0
            for j in xrange(self.config.ref_ind, self.config.nmag):
                col_indices[c] = j - 1
                sign_indices[c] = -1
                mind_indices[c] = j
                c += 1
            for j in xrange(self.config.ref_ind - 1, 0, -1):
                col_indices[c] = j
                sign_indices[c] = 1
                mind_indices[c] = j
                c += 1

            lups = np.zeros_like(mags)

            mags[:, self.config.ref_ind] = gals.mag[:, self.config.ref_ind]
            flux = 10.**((mags[:, self.config.ref_ind] - self.lupzp) / (-2.5))
            lups[:, self.config.ref_ind] = 2.5 * np.log10(1.0 / self.config.b[self.config.ref_ind]) - np.asinh(0.5 * flux / bnmgy[self.config.ref_ind]) / (0.4 * np.log(10.0))
        else:
            col_indices = np.arange(ncol)
            sign_indices = np.ones(ncol, dtype=np.int32)
            mind_indices = col_indices

        # One color at a time along the diagonal
        for c in xrange(ncol):
            starttime = time.time()

            # The order is given by col_indices, which ensures that we work from the
            # reference mag outward
            j = col_indices[c]
            sign = sign_indices[c]
            mind = mind_indices[c]

            print("Working on diagonal for color %d" % (j))

            col = galcolor[:, j]
            col_err = galcolor_err[:, j]

            # Need to go through the _ndarray because ztag and zstag are strings
            cvals = np.zeros(self.pars._ndarray[self.ztag[j]].size)
            svals = np.zeros(self.pars._ndarray[self.zstag[j]].size)
            scvals = np.zeros(self.pars.covmat_z.size) + 0.05
            photo_err = np.zeros_like(cvals)

            # Calculate median truncation
            spl = CubicSpline(self.pars.pivotmag_z, self.pars.medcol[:, j])
            med = spl(gals.z)
            spl = CubicSpline(self.pars.pivotmag_z, self.pars.medcol_width[:, j])
            sc = spl(gals.z)

            u, = np.where((galcolor[:, j] > (med - self.config.calib_color_nsig * sc)) &
                          (galcolor[:, j] < (med + self.config.calib_color_nsig * sc)))
            trunc = np.zeros((2, u.size))
            trunc[0, :] = med[u] - self.config.calib_color_nsig * sc[u]
            trunc[1, :] = med[u] + self.config.calib_color_nsig * sc[u]

            # And the starting values...
            # Note that this returns the slope values (svals) at the nodes from the cvals
            # but these might not be the same nodes, so we have to approximate
            cvals_temp, svals_temp, _ = self.compute_startvals(self.pars._ndarray[self.ztag[j]],
                                                                        gals.z[u], col[u],
                                                                        fit=True, mincomp=5)
            cvals[:] = cvals_temp
            inds = np.searchsorted(self.pars._ndarray[self.ztag[i]],
                                   self.pars._ndarray[self.zstag[j]])
            svals[:] = svals_temp[inds]

            dmags = gals.refmag - pivotmags

            # And do the luptitude correction if necessary.
            if self.do_lupcorr:
                lupcorr = self._compute_single_lupcorr(cvals, svals, gals, dmags, mags, lups, mind, sign)
            else:
                lupcorr = np.zeros(gals.size)

            # We fit in stages: first the mean, then the slope, then the scatter,
            # and finally all three

            rsfitter = RedSequenceFitter(self.pars._ndarray[self.ztag[i]],
                                         gals.z[u], col[u], col_err[u],
                                         dmags=dmags[u],
                                         trunc=trunc[:, u],
                                         slope_nodes=self.pars._ndarray[self.zstag[j]],
                                         scatter_nodes=self.pars.covmat_z,
                                         lupcorrs=lupcorr[u],
                                         probs=probs[u],
                                         bkgs=self.bkg.lookup_diagonal(j, col[u], gals.refmag[u]))

            # fit the mean
            cvals, = rsfitter.fit(cvals, svals, scvals, fit_mean=True)
            # Update the lupcorr...
            if self.do_lupcorr:
                rsfitter._lupcorrs[:] = self._compute_single_lupcorr(cvals, svals, gals, dmags, mags, lups, mind, sign)[u]
            # fit the slope
            svals, = rsfitter.fit(cvals, svals, scvals, fit_slope=True)
            # fit the scatter
            scvals, = rsfitter.fit(cvals, svals, scvals, fit_scatter=True)
            # fit combined
            cvals, svals, scvals = rsfitter.fit(cvals, svals, scvals,
                                                fit_mean=True, fit_slope=True, fit_scatter=True)

            # And record in the parameters
            self.pars._ndarray[self.ctag[j]] = cvals
            self.pars._ndarray[self.stag[j]] = svals
            self.pars.sigma[j, j, :] = scvals
            self.pars.covmat_amp[j, j, :] = scvals ** 2.

            # And print the time taken
            print('Done in %.2f seconds.' % (time.time() - starttime))

    def _calc_offdiagonal_pars(self, gals):
        """
        """
        # The routine to compute the off-diagonal elements

        ncol = self.config.nmag - 1

        galcolor = gals.galcol
        galcolor_err = gals.galcol_err

        # compute the pivot mags
        spl = CubicSpline(self.pars.pivotmag_z, self.pars.pivotmag)
        pivotmags = spl(gals.z)

        # And set the right probabilities
        if self.config.calib_use_pcol:
            probs = gals.pcol
        else:
            probs = gals.pmem

        



    def _calc_volume_factor(self, zref):
        """
        """

        dz = 0.01

        self.pars.volume_factor = ((self.config.cosmo.Dl(0.0, zref + dz) / (1. + (zref + dz)) -
                                   self.config.cosmo.Dl(0.0, zref) / (1. + zref)) /
                                   (self.config.cosmo.Dl(0.0, self.pars.volume_factor_z + dz) / (1. + (self.pars.volume_factor_z + dz)) -
                                    self.config.cosmo.Dl(0.0, self.pars.volume_factor_z) / (1. + self.pars.volume_factor_z)))


    def save_pars(self, filename, clobber=False):
        """
        """

        hdr = fitsio.FITSHDR()
        hdr['NCOL'] = self.config.nmag - 1
        hdr['MSTARBUR'] = self.config.mstar_survey
        hdr['MSTARBAN'] = self.config.mstar_band
        hdr['LIMMAG'] = self.config.limmag_catalog
        # Saved with arbitrary cushion that seems to work well
        hdr['ZRANGE0'] = np.clip(self.config.zrange[0] - 0.07, 0.01, None)
        hdr['ZRANGE1'] = self.config.zrange[1] + 0.07
        hdr['ALPHA'] = self.config.calib_lumfunc_alpha0
        hdr['ZBINFINE'] = self.config.zredc_binsize_fine
        hdr['ZBINCOAR'] = self.config.zredc_binsize_coarse
        hdr['LOWZMODE'] = 0
        hdr['REF_IND'] = self.config.ref_ind
        for j in xrange(self.config.b):
            hdr['BVALUE%d' % (j + 1)] = self.config.b[j]

        fitsio.write(filename, self.pars._ndarray, header=hdr, clobber=clobber)

    def _calc_zreds(self, gals, do_correction=True):
        """
        """

        # This is temporary

        zredstr = RedSequenceColorPar(self.config.parfile)

        zredc = ZredColor(zredstr, adaptive=True, do_correction=do_correction)

        gals.add_zred_fields()

        for i, g in enumerate(gals):
            try:
                zredc.compute_zred(g)
            except:
                pass

    def _calc_corrections(self, gals, mode2=False):
        """
        """

        # p or pcol
        if self.config.calib_use_pcol:
            probs = gals.pcol
        else:
            probs = gals.pmem

        # Set a threshold removing 5% worst lkhd outliers
        st = np.argsort(gals.lkhd)
        thresh = gals.lkhd[st[int(0.05 * gals.size)]]

        # This is an arbitrary 2sigma cut...
        guse, = np.where((gals.lkhd > thresh) &
                         (np.abs(gals.z - gals.zred) < 2. * gals.zred_e))

        spl = CubicSpline(self.pars.pivotmag_z, self.pars.pivotmag)
        pivotmags = spl(gals.z)

        w = 1. / (np.exp((thresh - gals.lkhd[guse]) / 0.2) + 1.0)

        # The offset cvals
        cvals = np.zeros(self.pars.corr_z.size)
        # The slope svals
        svals = np.zeros(self.pars.corr_slope_z.size)
        # And the r value to be multiplied by error
        rvals = np.ones(self.pars.corr_slope_z.size)
        # And the background vals
        bkg_cvals = np.zeros(self.pars.corr_slope_z.size)

        cvals[:], _ = self._compute_startvals(self.pars.corr_z, gals.z, gals.z - gals.zred, median=True)

        # Initial guess for bkg_cvals is trickier and not generalizable (sadly)
        for i in xrange(self.pars.corr_slope_z.size):
            if i == 0:
                zlo = self.pars.corr_slope_z[0]
            else:
                zlo = (self.pars.corr_slope_z[i - 1] + pars.corr_slope_z[i]) / 2.
            if i == (self.pars.corr_slope_z.size - 1):
                zhi = self.pars.corr_slope_z[i]
            else:
                zhi = (self.pars.corr_slope_z[i] + self.pars.corr_slope_z[i + 1]) / 2.

            if mode2:
                u, = np.where((gals.zred[guse] > zlo) & (gals.zred[guse] < zhi))
            else:
                u, = np.where((gals.z[guse] > zlo) & (gals.z[guse] < zhi))

            if u.size < 3:
                if i > 0:
                    bkg_cvals[i] = bkg_cvals[i - 1]
            else:
                st = np.argsort(probs[guse[u]])
                uu = u[st[0:nu/2]]
            bkg_cvals[i] = np.std(gals.z[guse[uu]] - gals.zred[guse[uu]])**2.

        if mode2:
            print("Fitting zred2 corrections...")
            z = gals.zred
        else:
            print("Fitting zred corrections...")
            z = gals.z

        corrfitter = CorrectionFitter(self.pars.corr_z,
                                      z[guse],
                                      gals.z[guse] - gals.zred[guse],
                                      gals.zred_e[guse],
                                      slope_nodes=self.pars.corr_slope_z,
                                      probs=np.clip(probs[guse], None, 0.99),
                                      dmags=gals.refmag[guse] - pivotmags[guse],
                                      ws=w[guse])

        # self.config.calib_corr_nocorrslope
        # first fit the mean
        cvals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_mean=True)
        # fit the slope (if desired)
        if not self.config.calib_corr_nocorrslope:
            svals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_slope=True)
        # Fit r
        rvals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_r=True)
        # Fit bkg
        bkg_cvals, = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_bkg=True)

        # Combined fit
        if not self.config.calib_corr_nocorrslope:
            cvals, svals, rvals, bkg_cvals = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_mean=True, fit_slope=True, fit_r=True, fit_bkg=True)
        else:
            cvals, rvals, bkg_cvals = corrfitter.fit(cvals, svals, rvals, bkg_cvals, fit_mean=True, fit_r=True, fit_bkg=True)

        # And record the values
        if mode2:
            self.pars.corr2 = cvals
            self.pars.corr2_slope = svals
            self.pars.corr2_r = rvals
        else:
            self.pars.corr = cvals
            self.pars.corr_slope = svals
            self.pars.corr_r = rvals

    def _make_diagnostic_plots(self, gals):
        """
        """

        import matplotlib.pyplot as plt

        # what plots do we want?
        # We want to split this out into different modules?

        # For each color, plot
        #  Color(z)
        #  Slope(z)
        #  scatter(z)
        # And a combined
        #  All off-diagonal r value plots

        fig = plt.figure(figsize=(10,6))
        fig.clf()


        plt.close(fig)

        # And two panel plot with
        #  left panel is offset, scatter, outliers as f(z)
        #  Right panel is zred vs z (whichever)
        # We need to do this for both zred and zred2.

