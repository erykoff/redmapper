"""Classes to describe clusters and cluster catalogs.

This file contains the main cluster and cluster catalog classes, including
richness computation, etc.
"""
import fitsio
import esutil
import numpy as np
import itertools
import scipy.optimize
import scipy.integrate
import copy

from .solver_nfw import Solver
from .catalog import Catalog, Entry
from .utilities import chisq_pdf, calc_theta_i, MStar, schechter_pdf, nfw_pdf
from .mask import HPMask
from .redsequence import RedSequenceColorPar
from esutil.cosmology import Cosmo
from .galaxy import GalaxyCatalog
from .depth_fitting import calcErrorModel

cluster_dtype_base = [('MEM_MATCH_ID', 'i4'),
                      ('RA', 'f8'),
                      ('DEC', 'f8'),
                      ('Z', 'f4'),
                      ('REFMAG', 'f4'),
                      ('REFMAG_ERR', 'f4'),
                      ('LAMBDA', 'f4'),
                      ('LAMBDA_E', 'f4'),
                      ('Z_LAMBDA', 'f4'),
                      ('Z_LAMBDA_E', 'f4'),
                      ('CG_SPEC_Z', 'f4'),
                      ('Z_SPEC_INIT', 'f4'),
                      ('Z_INIT', 'f4'),
                      ('R_LAMBDA', 'f4'),
                      ('R_MASK', 'f4'),
                      ('SCALEVAL', 'f4'),
                      ('MASKFRAC', 'f4'),
                      ('ZRED', 'f4'),
                      ('ZRED_E', 'f4'),
                      ('ZRED_CHISQ', 'f4'),
                      ('CHISQ', 'f4'),
                      ('Z_LAMBDA_NITER', 'i2'),
                      ('EBV_MEAN', 'f4'),
                      ('LNLAMLIKE', 'f4'),
                      ('LNCGLIKE', 'f4'),
                      ('LNLIKE', 'f4'),
                      ('RA_ORIG', 'f8'),
                      ('DEC_ORIG', 'f8'),
                      ('W', 'f4'),
                      ('DLAMBDA_DZ', 'f4'),
                      ('DLAMBDA_DZ2', 'f4'),
                      ('DLAMBDAVAR_DZ', 'f4'),
                      ('DLAMBDAVAR_DZ2', 'f4'),
                      ('Z_LAMBDA_RAW', 'f4'),
                      ('Z_LAMBDA_E_RAW', 'f4'),
                      ('BKG_LOCAL', 'f4'),
                      ('LIM_EXPTIME', 'f4'),
                      ('LIM_LIMMAG', 'f4'),
                      ('LIM_LIMMAG_HARD', 'f4'),
                      ('LAMBDA_C', 'f4'),
                      ('LAMBDA_CE', 'f4'),
                      ('NCENT_GOOD', 'i2'),
                      ('MASKGAL_INDEX', 'i2')]

member_dtype_base = [('MEM_MATCH_ID', 'i4'),
                     ('ID', 'i8'),
                     ('Z', 'f4'),
                     ('RA', 'f8'),
                     ('DEC', 'f8'),
                     ('R', 'f4'),
                     ('P', 'f4'),
                     ('PFREE', 'f4'),
                     ('PCOL', 'f4'),
                     ('THETA_I', 'f4'),
                     ('THETA_R', 'f4'),
                     ('REFMAG', 'f4'),
                     ('REFMAG_ERR', 'f4'),
                     ('ZRED', 'f4'),
                     ('ZRED_E', 'f4'),
                     ('ZRED_CHISQ','f4'),
                     ('CHISQ', 'f4'),
                     ('EBV', 'f4'),
                     ('ZSPEC', 'f4')]


class Cluster(Entry):
    """
    Class for a single galaxy cluster.

    This class includes methods to perform richness computations on individual clusters using the associated neighbor galaxy catalog.
    """

    def __init__(self, cat_vals=None, r0=None, beta=None, config=None, zredstr=None, bkg=None, cbkg=None, neighbors=None, zredbkg=None, dtype=None):
        """
        Instantiate a Cluster object.

        Note that while all the associated config, zredstr, bkg, neighbors are
        optional to instantiate the object, they are necessary in order to
        perform any calculations (but not to be able to store a representation
        of the cluster catalog.)  This also allows these to be set "lazily"
        after instantiation.

        Parameters
        ----------
        cat_vals: `np.ndarray`, optional
           Existing catalog values to pre-fill.  Default is None (zeros).
        r0: `float`, optional
           Richness/radius scale parameter.  Default to 1.0 h^-1 Mpc.
        beta: `float`, optional
           Richness/radius slope parameter.  Default to 0.2.
        config: `redmapper.Configuration`, optional
           Configuration information.  Default is None.
        zredstr: `redmapper.RedSequenceColorPar`, optional
           Red sequence parameterization.  Default is None.
        bkg: `redmapper.Background`, optional
           Galaxy background.  Default is None.
        cbkg: `redmapper.ColorBackground`, optional
           Galaxy color-only background.  Default is None.
        neighbors: `redmapper.GalaxyCatalog`, optional
           Neighbor galaxy catalog.  Default is None.
        zredbkg: `redmapper.ZredBackground`, optional
           Zred background.  Default is None.
        """

        if cat_vals is None:
            if dtype is not None:
                cat_vals = np.zeros(1, dtype=dtype)
            else:
                if config is not None:
                    cat_vals = np.zeros(1, dtype=config.cluster_dtype)
                else:
                    # This might lead to bugs down the line, but let's try
                    cat_vals = np.zeros(1, dtype=cluster_dtype_base)

        # Start by taking catalog values and stuffing them into a nice Entry format
        # we need to extend if necessary?  Or just the catalog?
        super(Cluster, self).__init__(cat_vals)

        self.r0 = 1.0 if r0 is None else r0
        self.beta = 0.2 if beta is None else beta

        # FIXME: this should explicitly set our default cosmology
        if config is None:
            self.cosmo = Cosmo()
        else:
            self.cosmo = config.cosmo
        self.config = config
        self.zredstr = zredstr
        self.bkg = bkg
        self.cbkg = cbkg
        self.zredbkg = zredbkg
        self.set_neighbors(neighbors)

        self._mstar = None
        self._mpc_scale = None

        if self.z > 0.0 and self.zredstr is not None:
            self.redshift = self.z
        else:
            self._redshift = None

    def reset(self):
        """
        Reset the cluster richness and redshift to -1.0.
        """

        # reset values to defaults
        self.Lambda = -1.0
        self.z_lambda = -1.0

    def set_neighbors(self, neighbors):
        """
        Set the neighbor galaxy catalog from a list of neighbors.  The input
        neighbor catalog is deep-copied.

        Parameters
        ----------
        neighbors: `redmapper.GalaxyCatalog`
        """

        if (neighbors.__class__ is not GalaxyCatalog and neighbors is not None):
            raise ValueError("Cluster neighbors must be a GalaxyCatalog")

        self.neighbors = None
        if (neighbors is not None):
            self.neighbors = copy.deepcopy(neighbors)

            # extra fields
            neighbor_extra_dtype = [('R', 'f8'),
                                    ('DIST', 'f8'),
                                    ('CHISQ', 'f8'),
                                    ('ZRED_CHISQ', 'f8'),
                                    ('PFREE', 'f8'),
                                    ('THETA_I', 'f8'),
                                    ('THETA_R', 'f8'),
                                    ('P', 'f8'),
                                    ('PCOL', 'f8'),
                                    ('PMEM', 'f8'),
                                    ('INDEX', 'i8')]

            dtype_augment = [dt for dt in neighbor_extra_dtype if dt[0].lower() not in self.neighbors.dtype.names]
            if len(dtype_augment) > 0:
                self.neighbors.add_fields(dtype_augment)

            if 'PFREE' in [dt[0] for dt in dtype_augment]:
                # The PFREE is new, so we must set it to 1s
                self.neighbors.pfree[:] = 1.0

            if ('ZRED_CHISQ', 'f8') in dtype_augment:
                # If we've had to add this, we want to copy the chisq values
                # since they were from the "zred" side
                self.neighbors.zred_chisq = self.neighbors.chisq

    def find_neighbors(self, radius, galcat, megaparsec=False, maxmag=None):
        """
        Find neighbors from a full galaxy catalog.

        Parameters
        ----------
        radius: `float`
           Radius in degrees or megaparsec to search for neighbors
        galcat: `redmapper.GalaxyCatalog`
           Full catalog of galaxies to look for neighbors
        megaparsec: `bool`, optional
           The radius has units of megaparsec?  Default is False.
        maxmag: `float`, optional
           The maximum refmag to store the neighbors.  Default is None (no cuts).
        """
        if radius is None:
            raise ValueError("A radius must be specified")
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")

        if megaparsec:
            radius_degrees = radius / self.mpc_scale
        else:
            radius_degrees = radius

        indices, dists = galcat.match_one(self.ra, self.dec, radius_degrees)

        if maxmag is not None:
            use, = np.where(galcat.refmag[indices] <= maxmag)
            indices = indices[use]
            dists = dists[use]

        self.set_neighbors(galcat[indices])
        self.neighbors.dist = dists
        self.neighbors.index = indices

        # And we need to compute the r values here
        self._compute_neighbor_r()

    def update_neighbors_dist(self):
        """
        Update the distance from the neighbors to the central galaxy (in degrees)
        """

        self.neighbors.dist = esutil.coords.sphdist(self.ra, self.dec,
                                                    self.neighbors.ra, self.neighbors.dec)

        self._compute_neighbor_r()

    def clear_neighbors(self):
        """
        Clear out all neighbors, to save memory.
        """

        # Clear out the memory used by the neighbors.
        self.neighbors = None

    def _calc_radial_profile(self, idx=None, rscale=0.15):
        """
        Internal method for computing radial profile weights.

        Parameters
        ----------
        idx: `np.array`, optional
           Integer indices to compute.  Default is None (all).
        rscale: `float`, optional
           r_s for nfw profile.  Default is 0.15

        Returns
        -------
        sigx: `np.array`
           sigma(x) from radial profile
        """
        if idx is None:
            idx = np.arange(len(self.neighbors))

        sigx = nfw_pdf(self.neighbors.r[idx], rscale=rscale)

        return sigx

    def _calc_luminosity(self, normmag, idx=None):
        """
        Internal method to compute luminosity filter

        Parameters
        ----------
        normmag: `float`
           Normalization magnitude
        idx: `np.array`, optional
           Integer indices to compute.  Default is None (all).

        Returns
        -------
        phi: `np.array`
           phi(x) for the cluster
        """
        if idx is None:
            idx = np.arange(len(self.neighbors))

        zind = self.zredstr.zindex(self._redshift)
        refind = self.zredstr.lumrefmagindex(normmag)
        normalization = self.zredstr.lumnorm[refind, zind]
        mstar = self.zredstr.mstar(self._redshift)
        phi = schechter_pdf(self.neighbors.refmag[idx], alpha=self.zredstr.alpha, mstar=mstar)
        return phi / normalization

    def calc_bkg_density(self, r, chisq, refmag):
        """
        Internal method to compute background filter.

        Parameters
        ----------
        r: `np.array`
           Radius (megaparsec)
        chisq: `np.array`
           Chi-squared values at redshift of the cluster
        refmag: `np.array`
           Reference magnitude of the galaxies

        Returns
        -------
        bcounts: `np.array`
           b(x) for the neighbors
        """
        sigma_g = self.bkg.sigma_g_lookup(self._redshift, chisq, refmag)
        return 2. * np.pi * r * (sigma_g/self.mpc_scale**2.)

    def calc_cbkg_density(self, r, col_index, col, refmag):
        """
        Internal method to compute color background filter.

        Parameters
        ----------
        r: `np.array`
           Radius (megaparsec)
        col_index: `int`
           Index of color used
        col: `np.array`
           Float array of colors
        refmag: `np.array`
           Reference magnitude of galaxies

        Returns
        -------
        bcounts: `np.array`
           b(x) for the neighbors
        """
        sigma_g = self.cbkg.sigma_g_diagonal(col_index, col, refmag)
        return 2. * np.pi * r * (sigma_g / self.mpc_scale**2.)

    def calc_zred_bkg_density(self, r, zred, refmag):
        """
        Internal method to compute zred background filter

        Parameters
        ----------
        r: `np.array`
           Radius (megaparsec)
        zred: `np.array`
           Zred values
        refmag: `np.array`
           Reference magnitude of galaxies

        Returns
        -------
        bcounts: `np.array`
           b(x) for the neighbors
        """

        if self.zredbkg is None:
            raise AttributeError("zredbkg has not been set for this cluster")

        sigma_g = self.zredbkg.sigma_g_lookup(zred, refmag)
        return 2. * np.pi * r * (sigma_g / self.mpc_scale**2.)

    def compute_bkg_local(self, mask, depth):
        """
        Compute the local background relative to the global.

        Parameters
        ----------
        mask: `redmapper.Mask`
           Footprint mask for survey
        depth: `redmapper.Depthmap` or `redmapper.Depthlim`
           Depth map for survey or depth fitting class

        Returns
        -------
        bkg_local: `float`
           Local background relative to global average for redshift
        """
        ras = self.ra + (mask.maskgals.x_uniform/self.mpc_scale)/np.cos(np.deg2rad(self.dec))
        decs = self.dec + mask.maskgals.y_uniform/self.mpc_scale

        maxmag = self.mstar - 2.5*np.log10(self.config.lval_reference)
        maskgals_mark = mask.compute_radmask(ras, decs)
        maskgals_refmag = self.mstar + mask.maskgals.m
        try:
            maskgals_depth = depth.get_depth_values(ras, decs)[0]
        except AttributeError:
            # We have a "Depthlim" limit.
            limpars, fail = calcErrorModel(self.neighbors.refmag, self.neighbors.refmag_err, calcErr=False)
            if fail:
                maskgals_depth = depth.initpars['LIMMAG']
            else:
                maskgals_depth = limpars['LIMMAG']

        sigma_g_maskgals = self.bkg.sigma_g_lookup(self._redshift, mask.maskgals.chisq, mask.maskgals.refmag)

        bright_enough, = np.where((mask.maskgals.refmag < maxmag) & (np.isfinite(sigma_g_maskgals))
&
                                  (mask.maskgals.chisq_pdf > 0.0) & (mask.maskgals.lum_pdf > 0.0) &
                                  (mask.maskgals.refmag < maskgals_depth))

        # Predicted number density according to global bkg
        prediction = np.sum(sigma_g_maskgals[bright_enough].astype(np.float64) / (mask.maskgals.chisq_pdf[bright_enough].astype(np.float64) * mask.maskgals.lum_pdf[bright_enough].astype(np.float64))) / float(bright_enough.size)

        # What is the annulus area?
        in_annulus, = np.where((mask.maskgals.r_uniform > self.config.bkg_local_annuli[0]) &
                               (mask.maskgals.r_uniform < self.config.bkg_local_annuli[1]))
        in_annulus_gd, = np.where(maskgals_mark[in_annulus])
        annulus_area = (np.pi*((self.config.bkg_local_annuli[1]/self.mpc_scale)**2. -
                               (self.config.bkg_local_annuli[0]/self.mpc_scale)**2.) *
                        (float(in_annulus_gd.size) / float(in_annulus.size)))

        try:
            neighbors_depth = depth.get_depth_values(self.neighbors.ra, self.neighbors.dec)[0]
        except AttributeError:
            neighbors_depth = maskgals_depth

        neighbors_in_annulus, = np.where((self.neighbors.r > self.config.bkg_local_annuli[0]) &
                                         (self.neighbors.r < self.config.bkg_local_annuli[1]) &
                                         (self.neighbors.refmag < maxmag) &
                                         (self.neighbors.chisq < mask.maskgals.chisq.max()) &
                                         (self.neighbors.refmag < neighbors_depth))
        bkg_density_in_annulus = float(neighbors_in_annulus.size) / annulus_area

        bkg_local = bkg_density_in_annulus / prediction

        return bkg_local

    def calc_richness(self, mask, calc_err=True, index=None):
        """
        Calculate the richness for the cluster.

        Parameters
        ----------
        mask: `redmapper.Mask`
           Footprint mask for survey
        calc_err: `bool`, optional
           Calculate the richness error?  Default is True.
        index: `np.array`, optional
           Integer array of neighbor indices.  Default is None (all).

        Returns
        -------
        lam: `float`
           Cluster richness.  Will be < 0 when no cluster found.
        """
        #set index for slicing self.neighbors
        if index is not None:
            idx = index
        else:
            idx = np.arange(len(self.neighbors))

        maxmag = self.mstar - 2.5 * np.log10(self.config.lval_reference)

        self.neighbors.chisq[idx] = self.zredstr.calculate_chisq(self.neighbors[idx], self._redshift)
        rho = chisq_pdf(self.neighbors.chisq[idx], self.zredstr.ncol)
        nfw = self._calc_radial_profile(idx=idx)
        phi = self._calc_luminosity(maxmag, idx=idx) #phi is lumwt in the IDL code
        ucounts = (2*np.pi*self.neighbors.r[idx]) * nfw * phi * rho
        bcounts = self.calc_bkg_density(self.neighbors.r[idx], self.neighbors.chisq[idx],
                                        self.neighbors.refmag[idx])

        theta_i = calc_theta_i(self.neighbors.refmag[idx], self.neighbors.refmag_err[idx],
                               maxmag, self.zredstr.limmag)

        cpars = mask.calc_maskcorr(self.mstar, maxmag, self.zredstr.limmag)

        try:
            w = theta_i * self.neighbors.pfree[idx]
        except AttributeError:
            w = theta_i * np.ones_like(ucounts)

        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts,
                              self.neighbors.r[idx], w,
                              cpars=cpars, rsig=self.config.rsig)

        # Call the solving routine
        # this returns five items: lam_obj, p, pmem, rlam, theta_r
        # Note that pmem used to be called "wvals" in IDL code
        # pmem = p * pfree * theta_i * theta_r
        lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()

        # reset before setting subsets
        self.neighbors.theta_i[:] = 0.0
        self.neighbors.theta_r[:] = 0.0
        self.neighbors.p[:] = 0.0
        self.neighbors.pcol[:] = 0.0
        self.neighbors.pmem[:] = 0.0

        # This also checks for crazy invalid values
        if lam < 0.0 or pmem.max() == 0.0:
            lam = -1.0
            lam_err = -1.0
            self.scaleval = -1.0
        else:
            # Only do this computation if we have a valid measurement
            bar_pmem = np.sum(pmem**2.0)/np.sum(pmem)
            cval = np.clip(np.sum(cpars * rlam**np.arange(cpars.size, dtype=float)),
                           0.0, None)

            self.scaleval = np.absolute(lam / np.sum(pmem))

            lam_unscaled = lam / self.scaleval

            if calc_err:
                lam_cerr = self.calc_lambdacerr(mask.maskgals, self.mstar,
                                                lam, rlam, pmem, cval, self.config.dldr_gamma)
                lam_err = np.sqrt((1-bar_pmem) * lam_unscaled * self.scaleval**2. + lam_cerr**2.)

            # calculate pcol -- color only.  Don't need to worry about nfw norm!
            ucounts = rho*phi

            pcol = ucounts * lam/(ucounts * lam + bcounts)
            bad, = np.where((self.neighbors.r[idx] > rlam) | (self.neighbors.refmag[idx] > maxmag) |
                            (self.neighbors.refmag[idx] > self.zredstr.limmag) | (~np.isfinite(pcol)))
            pcol[bad] = 0.0

            # and set the values
            self.neighbors.theta_i[idx] = theta_i
            self.neighbors.theta_r[idx] = theta_r
            self.neighbors.p[idx] = p
            self.neighbors.pcol[idx] = pcol
            self.neighbors.pmem[idx] = pmem

        # set values and return

        self.Lambda = lam
        self.r_lambda = rlam
        if calc_err:
            self.Lambda_e = lam_err
        else:
            self.Lambda_e = 0.0

        return lam

    def calc_lambdacerr(self, maskgals, mstar, lam, rlam, pmem, cval, gamma):
        """
        Calculate richness error from masking only.

        Parameters
        ----------
        maskgals: `redmapper.Catalog`
           Mask galaxies for monte carlo (see `redmapper.Mask`)
        mstar: `float`
           mstar at redshift of the cluster
        lam: `float`
           Richness of cluster
        rlam: `float`
           Radius of cluster
        pmem: `np.array`
           Array of pmem membership probabilities
        cval: `float`
           Total mask value c
        gamma: `float`
           Slope of the dlambda/dradius relation (on average)

        Returns
        -------
        lam_err: `float`
           Error on richness due to masking.
        """
        dof = self.zredstr.ncol
        limmag = self.zredstr.limmag

        use, = np.where(maskgals.r < rlam)

        mark    = maskgals.mark[use]
        refmag  = mstar + maskgals.m[use]
        cwt     = maskgals.cwt[use]
        nfw     = maskgals.nfw[use]
        lumwt   = maskgals.lumwt[use]
        chisq   = maskgals.chisq[use]
        r       = maskgals.r[use]

        # normalizing nfw
        logrc   = np.log(rlam)
        norm    = np.exp(1.65169 - 0.547850*logrc + 0.138202*logrc**2. -
            0.0719021*logrc**3. - 0.0158241*logrc**4.-0.000854985*logrc**5.)
        nfw     = norm*nfw

        ucounts = cwt*nfw*lumwt

        #Set too faint galaxy magnitudes close to limiting magnitude
        faint, = np.where(refmag >= limmag)
        refmag_for_bcounts = np.copy(refmag)
        refmag_for_bcounts[faint] = limmag-0.01

        bcounts = self.calc_bkg_density(r, chisq, refmag_for_bcounts)

        out, = np.where((refmag > limmag) | (mark == 0))

        if out.size == 0 or cval < 0.01:
            lam_err = 0.0
        else:
            p_out = lam*ucounts[out] / (lam*ucounts[out] + bcounts[out])
            varc0 = (1./lam) * (1./use.size) * np.sum(p_out)
            sigc = np.sqrt(varc0 - varc0**2.)
            k = lam**2. / np.sum(pmem**2.)
            lam_err = k*sigc/(1. - self.beta*gamma)

        return lam_err

    def calc_richness_fit(self, mask, col_index, centcolor_in=None, rcut=0.5, mingal=5, sigint=0.05, calc_err=False):
        """
        Compute richness for a cluster by fitting the red sequence in a single color.
        This is approximate and is used for the first iteration of training.

        Parameters
        ----------
        mask: `redmapper.Mask`
           Footprint mask for survey
        col_index: `int`
           Index of color to use
        centcolor_in: `float`, optional
           Central color to use for reference.  Default is None (use BCG color)
        rcut: `float`, optional
           Maximum radius (megaparsec).  Default is 0.5.
        mingal: `int`, optional
           Minimum number of galaxies to try fit.  Default is 5.
        sigint: `float`, optional
           Default intrinsic scatter.  Default is 0.05.
        calc_err: `bool`, optional
           Compute richness error?  Default is False.

        Returns
        -------
        lam: `float`
           cluster richness.  Will be < 0 when no cluster found.
        """

        badlam = -10.0
        s2p = np.sqrt(2. * np.pi)

        maxmag = self.mstar - 2.5 * np.log10(self.config.lval_reference)
        minmag = self.mstar - 2.5 * np.log10(20.0)

        col = self.neighbors.galcol[:, col_index]
        col_err = self.neighbors.galcol_err[:, col_index]
        colrange = self.cbkg.get_colrange(col_index)

        guse, = np.where((self.neighbors.refmag > minmag) &
                         (self.neighbors.refmag < maxmag) &
                         (self.neighbors.r < rcut) &
                         (col > colrange[0]) &
                         (col < colrange[1]))

        if guse.size < mingal:
            self.scaleval = -1.0
            return badlam

        if centcolor_in is None:
            # We were not passed a probably central color
            # Use the BCG (central) color as the peak guess
            ind = np.argmin(self.neighbors.r)
            test, = np.where(guse == ind)
            if test.size == 0:
                # Nominal BCG not in color range.  All bad!
                return badlam

            centcolor = col[ind]
        else:
            centcolor = centcolor_in

        cerr = np.sqrt(col_err**2. + sigint**2.)
        in2sig, = np.where((np.abs(col[guse] - centcolor) < 2. * cerr[guse]))
        if in2sig.size < mingal:
            self.scaleval = -1.0
            return badlam

        pivot = np.median(self.neighbors.refmag[guse])

        fit = np.polyfit(self.neighbors.refmag[guse[in2sig]] - pivot,
                         col[guse[in2sig]],
                         1,
                         w=1. / col_err[guse[in2sig]])
        mpivot = fit[0]
        bpivot = fit[1]

        d = col - (mpivot * (self.neighbors.refmag - pivot) + bpivot)
        d_err_net = np.sqrt(col_err**2. + sigint**2.)
        d_wt = (1. / (s2p * d_err_net)) * np.exp(-(d**2.) / (2. * d_err_net**2.))

        # The nfw filter as normal
        nfw = self._calc_radial_profile()

        theta_i = calc_theta_i(self.neighbors.refmag, self.neighbors.refmag_err, maxmag, self.config.limmag_catalog)
        phi = self._calc_luminosity(maxmag)

        ucounts = (2. * np.pi * self.neighbors.r) * d_wt * nfw * phi

        bcounts = self.calc_cbkg_density(self.neighbors.r, col_index, col, self.neighbors.refmag)

        cpars = mask.calc_maskcorr(self.mstar, maxmag, self.config.limmag_catalog)

        try:
            w = theta_i * self.neighbors.pfree
        except AttributeError:
            w = theta_i * np.ones_like(ucounts)

        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts, self.neighbors.r, w, cpars=cpars)

        lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()

        bar_pmem = np.sum(pmem**2.) / np.sum(pmem)

        # reset
        self.neighbors.theta_i[:] = 0.0
        self.neighbors.theta_r[:] = 0.0
        self.neighbors.p[:] = 0.0
        self.neighbors.pcol[:] = 0.0
        self.neighbors.pmem[:] = 0.0

        if lam < 0.0:
            lam_err = -1.0
            self.scaleval = -1.0
        else:
            self.scaleval = np.absolute(lam / np.sum(pmem))

            lam_unscaled = lam / self.scaleval

            if calc_err:
                lam_cerr = self.calc_lambdacerr(mask.maskgals, self.mstar,
                                               lam, rlam, pmem, cval, self.config.dldr_gamma)
                lam_err = np.sqrt((1. - bar_pmem) * lam_unscaled + lambda_cerr**2.)

            # calculate pcol (color only)
            ucounts = d_wt * phi
            bcounts = (bcounts / (2. * np.pi * self.neighbors.r)) * np.pi * rlam**2.
            pcol = ucounts * lam / (ucounts * lam + bcounts)
            bad, = np.where((self.neighbors.r > rlam) | (self.neighbors.refmag > maxmag) |
                            (self.neighbors.refmag > self.config.limmag_catalog))
            pcol[bad] = 0.0

            # And set the values
            self.neighbors.theta_i[:] = theta_i
            self.neighbors.theta_r[:] = theta_r
            self.neighbors.p[:] = p
            self.neighbors.pcol[:] = pcol
            self.neighbors.pmem[:] = pmem

        # Set values and return
        self.Lambda = lam
        self.r_lambda = rlam
        if calc_err:
            self.Lambda_e = lam_err
        else:
            self.Lambda_e = 0.0

        return lam

    @property
    def redshift(self):
        """Current cluster redshift."""
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        """Set the cluster redshift.

        Parameters
        ----------
        value: `float`
           New redshift
        """
        if (value < 0.0):
            raise ValueError("Cannot set redshift to < 0.0")

        # This forces things to not blow up...
        self._redshift = np.clip(value, 0.01, None)
        self._update_mstar()
        self._update_mpc_scale()
        self._compute_neighbor_r()


    # want to change this and mpc_scale to properties,
    # and internal update methods.  When you update the redshift,
    # all of these things should be kept in sync.  That would be pretty cool.

    @property
    def mstar(self):
        """Get the cluster mstar at self.redshift"""
        return self._mstar

    def _update_mstar(self):
        """Update the stored mstar based on self.redshift"""
        self._mstar = self.zredstr.mstar(self._redshift)

    @property
    def mpc_scale(self):
        """
        Angular scaling in units of Mpc / degree
        """
        return self._mpc_scale

    def _update_mpc_scale(self):
        """
        Compute the scaling, in units of Mpc / degree
        """
        self._mpc_scale = np.radians(1.) * self.cosmo.Da(0, self._redshift)

    def _compute_neighbor_r(self):
        """
        Compute the radius in Mpc for the neighbors, given the current redshift
        and neighbor dist (in degrees).
        """
        if self.neighbors is not None and self._redshift is not None:
            # Clipping at 1e-6 to avoid singularities.
            self.neighbors.r = np.clip(self.mpc_scale * self.neighbors.dist, 1e-6, None)


    def copy(self):
        """
        Copy the current cluster, including deep copying the neighbors.

        Returns
        -------
        cluster: `redmapper.Cluster`
           Deep copy of the cluster
        """
        cluster = self.__copy__()
        cluster.redshift = self.redshift
        return cluster

    def __copy__(self):
        # This returns a copy of the cluster, and note that the neighbors will
        # be deepcopied which is what we want.
        return Cluster(r0=self.r0,
                       beta=self.beta,
                       config=self.config,
                       zredstr=self.zredstr,
                       bkg=self.bkg,
                       cbkg=self.cbkg,
                       neighbors=self.neighbors)

class ClusterCatalog(Catalog):
    """
    Class for a catalog of clusters.

    This class comprises a set of clusters and their neighbors.
    """
    entry_class = Cluster

    def __init__(self, array, **kwargs):
        """
        Instantiate a ClusterCatalog

        Parameters
        ----------
        array: `np.ndarray`
           Array of initialization values for the cluster catalog
        r0: `float`, optional
           Richness/radius scale parameter.  Default to 1.0 h^-1 Mpc.
        beta: `float`, optional
           Richness/radius slope parameter.  Default to 0.2.
        config: `redmapper.Configuration`, optional
           Configuration information.  Default is None.
        zredstr: `redmapper.RedSequenceColorPar`, optional
           Red sequence parameterization.  Default is None.
        bkg: `redmapper.Background`, optional
           Galaxy background.  Default is None.
        cbkg: `redmapper.ColorBackground`, optional
           Galaxy color-only background.  Default is None.
        zredbkg: `redmapper.ZredBackground`, optional
           Zred background.  Default is None.
        """
        super(ClusterCatalog, self).__init__(array)

        self.r0 = kwargs.pop('r0', None)
        self.beta = kwargs.pop('beta', None)
        self.zredstr = kwargs.pop('zredstr', None)
        self.config = kwargs.pop('config', None)
        self.bkg = kwargs.pop('bkg', None)
        self.cbkg = kwargs.pop('cbkg', None)
        self.zredbkg = kwargs.pop('zredbkg', None)
        dtype = kwargs.pop('dtype', None)

        if dtype is not None:
            cluster_dtype = dtype
        else:
            if self.config is not None:
                cluster_dtype = self.config.cluster_dtype
            else:
                cluster_dtype = cluster_dtype_base

        # and if config is set then use that cluster_dtype because that
        #  will have all the other stuff filled as well.
        dtype_augment = [dt for dt in cluster_dtype if dt[0].lower() not in self._ndarray.dtype.names]
        if len(dtype_augment) > 0:
            self.add_fields(dtype_augment)

    @classmethod
    def from_catfile(cls, filename, **kwargs):
        """
        Instantiate a ClusterCatalog from a catalog file

        Parameters
        ----------
        filename: `str`
           Filename of catalog file
        r0: `float`, optional
           Richness/radius scale parameter.  Default to 1.0 h^-1 Mpc.
        beta: `float`, optional
           Richness/radius slope parameter.  Default to 0.2.
        config: `redmapper.Configuration`, optional
           Configuration information.  Default is None.
        zredstr: `redmapper.RedSequenceColorPar`, optional
           Red sequence parameterization.  Default is None.
        bkg: `redmapper.Background`, optional
           Galaxy background.  Default is None.
        cbkg: `redmapper.ColorBackground`, optional
           Galaxy color-only background.  Default is None.
        zredbkg: `redmapper.ZredBackground`, optional
           Zred background.  Default is None.
        """
        cat = fitsio.read(filename, ext=1, upper=True)

        return cls(cat, **kwargs)

    @classmethod
    def zeros(cls, size, **kwargs):
        """
        Instantiate a ClusterCatalog of a given size, filled with zeros.

        Parameters
        ----------
        size: `int`
           Length of cluster catalog
        r0: `float`, optional
           Richness/radius scale parameter.  Default to 1.0 h^-1 Mpc.
        beta: `float`, optional
           Richness/radius slope parameter.  Default to 0.2.
        config: `redmapper.Configuration`, optional
           Configuration information.  Default is None.
        zredstr: `redmapper.RedSequenceColorPar`, optional
           Red sequence parameterization.  Default is None.
        bkg: `redmapper.Background`, optional
           Galaxy background.  Default is None.
        cbkg: `redmapper.ColorBackground`, optional
           Galaxy color-only background.  Default is None.
        zredbkg: `redmapper.ZredBackground`, optional
           Zred background.  Default is None.
        """
        dtype = kwargs.get('dtype', None)
        if dtype is not None:
            cluster_dtype = dtype
        else:
            cluster_dtype = cluster_dtype_base

        return cls(np.zeros(size, dtype=cluster_dtype), **kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            # Note that if we have members, we can associate them with the cluster
            #  here.
            return Cluster(cat_vals=self._ndarray.__getitem__(key),
                           r0=self.r0,
                           beta=self.beta,
                           zredstr=self.zredstr,
                           config=self.config,
                           bkg=self.bkg,
                           cbkg=self.cbkg,
                           zredbkg=self.zredbkg)
        else:
            return ClusterCatalog(self._ndarray.__getitem__(key),
                                  r0=self.r0,
                                  beta=self.beta,
                                  zredstr=self.zredstr,
                                  config=self.config,
                                  bkg=self.bkg,
                                  cbkg=self.cbkg,
                                  zredbkg=self.zredbkg)
