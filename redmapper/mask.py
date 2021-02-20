"""Classes for describing geometry masks in redmapper.

This file contains classes for reading and using geometry masks.
"""
import esutil
import fitsio
import healpy as hp
import numpy as np
import os
from scipy.special import erf
import scipy.integrate
import healsparse

from .catalog import Catalog,Entry
from .utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere, calc_theta_i, apply_errormodels
from .utilities import make_lockfile, sample_from_pdf, chisq_pdf, schechter_pdf, nfw_pdf
from .utilities import get_healsparse_subpix_indices

CURRENT_MASKGAL_VERSION = 7

class Mask(object):
    """
    A super-class to describe geometry footpint masks.
    """

    # note: not sure how to organize this.
    #   We need a routine that looks at the mask_mode and instantiates
    #   the correct type.  How is this typically done?

    def __init__(self, config, include_maskgals=True):
        """
        Instantiate a placeholder geometry mask that will describe all galaxies
        as in the mask.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        include_maskgals: `bool`, optional
           Also read in the maskgals.  Default is True.
        """
        self.config = config

        # This will raise if maskgals aren't available
        if include_maskgals:
            self.read_maskgals(config.maskgalfile)

    def compute_radmask(self, ra, dec):
        """
        Compute the geometric mask value at a list of positions.

        Parameters
        ----------
        ra: `np.array`
           Float array of right ascensions
        dec: `np.array`
           Float array of declinations

        Returns
        -------
        maskvals: `np.array`
           Bool array of True ("in the footprint") for each ra/dec.
        """
        _ra = np.atleast_1d(ra)
        _dec = np.atleast_1d(dec)

        if (_ra.size != _dec.size):
            raise ValueError("ra, dec must be same length")

        maskvals = np.ones(_ra.size, dtype=np.bool_)
        return maskvals

    def read_maskgals(self, maskgalfile):
        """
        Read the "maskgal" file for monte carlo estimation of coverage.

        Note that this reads the file into the object.

        Parameters
        ----------
        maskgalfile: `str`
           Filename of maskgal file with monte carlo galaxies
        """

        if not os.path.isfile(maskgalfile):
            raise RuntimeError("Could not find maskgalfile %s.  Please run mask.gen_maskgals(maskgalfile)" % (maskgalfile))

        # Check version
        hdr = fitsio.read_header(maskgalfile, ext=1)
        if (hdr['version'] != CURRENT_MASKGAL_VERSION):
            raise RuntimeError("maskgalfile %s is old version.  Please run mask.gen_maskgals(maskgalfile)" % (maskgalfile))

        # Read the maskgals
        # These are going to be *all* the maskgals, but we only operate on a subset
        # at a time
        self.maskgals_all = Catalog.from_fits_file(maskgalfile)

    def select_maskgals_sample(self, maskgal_index=None):
        """
        Select a subset of maskgals by sampling.

        This will set self.maskgals to the subset in question.

        Parameters
        ----------
        maskgal_index: `int`, optional
           Pre-selected index to sample from (for reproducibility).
           Default is None (select randomly).
        """

        if maskgal_index is None:
            maskgal_index = np.random.choice(self.config.maskgal_nsamples)

        self.maskgals = self.maskgals_all[maskgal_index * self.config.maskgal_ngals:
                                              (maskgal_index + 1) * self.config.maskgal_ngals]

        return maskgal_index

    def gen_maskgals(self, maskgalfile):
        """
        Method to generate the maskgal monte carlo galaxies.

        Parameters
        ----------
        maskgalfile: `str`
           Name of maskgal file to generate.
        """

        minrad = np.clip(np.floor(10.*self.config.percolation_r0 * (3./100.)**self.config.percolation_beta) / 10., None, 0.5)
        maxrad = np.ceil(10.*self.config.percolation_r0 * (300./100.)**self.config.percolation_beta) / 10.

        nradbins = np.ceil((maxrad - minrad) / self.config.maskgal_rad_stepsize).astype(np.int32) + 1
        radbins = np.arange(nradbins, dtype=np.float32) * self.config.maskgal_rad_stepsize + minrad

        nmag = self.config.nmag
        ncol = nmag - 1

        ngals = self.config.maskgal_ngals * self.config.maskgal_nsamples

        maskgals = Catalog.zeros(ngals, dtype=[('r', 'f4'),
                                               ('phi', 'f4'),
                                               ('x', 'f4'),
                                               ('y', 'f4'),
                                               ('r_uniform', 'f4'),
                                               ('x_uniform', 'f4'),
                                               ('y_uniform', 'f4'),
                                               ('m', 'f4'),
                                               ('refmag', 'f4'),
                                               ('refmag_obs', 'f4'),
                                               ('refmag_obs_err', 'f4'),
                                               ('chisq', 'f4'),
                                               ('cwt', 'f4'),
                                               ('chisq_pdf', 'f4'),
                                               ('nfw', 'f4'),
                                               ('dzred', 'f4'),
                                               ('zwt', 'f4'),
                                               ('lumwt', 'f4'),
                                               ('lum_pdf', 'f4'),
                                               ('limmag', 'f4'),
                                               ('limmag_dered', 'f4'),
                                               ('exptime', 'f4'),
                                               ('m50', 'f4'),
                                               ('eff', 'f4'),
                                               ('w', 'f4'),
                                               ('theta_r', 'f4', nradbins),
                                               ('mark', bool),
                                               ('radbins', 'f4', nradbins),
                                               ('nin', 'f4', nradbins),
                                               ('nin_orig', 'f4', nradbins),
                                               ('zp', 'f4'),
                                               ('ebv', 'f4'),
                                               ('extinction', 'f4'),
                                               ('nsig', 'f4')])

        maskgals['radbins'] = np.tile(radbins, maskgals.size).reshape(maskgals.size, nradbins)

        # Generate chisq
        maskgals.chisq = sample_from_pdf(chisq_pdf, [0.0, self.config.chisq_max],
                                         self.config.chisq_max / 10000.,
                                         maskgals.size, k=ncol)
        # Generate mstar
        maskgals.m = sample_from_pdf(schechter_pdf,
                                     [-2.5*np.log10(10.0),
                                       -2.5*np.log10(self.config.lval_reference) + self.config.maskgal_dmag_extra],
                                     0.002, maskgals.size,
                                     alpha=self.config.calib_lumfunc_alpha, mstar=0.0)
        # Generate nfw(r)
        maskgals.r = sample_from_pdf(nfw_pdf,
                                     [0.001, maxrad],
                                     0.001, maskgals.size, radfactor=True)

        # Generate phi
        maskgals.phi = 2. * np.pi * np.random.random(size=maskgals.size)

        # Precompute x/y
        maskgals.x = maskgals.r * np.cos(maskgals.phi)
        maskgals.y = maskgals.r * np.sin(maskgals.phi)

        # And uniform x/y
        maskgals.r_uniform = self.config.bkg_local_annuli[1] * np.sqrt(np.random.uniform(size=maskgals.size))
        theta_new = np.random.uniform(size=maskgals.size)*2*np.pi
        maskgals.x_uniform = maskgals.r_uniform*np.cos(theta_new)
        maskgals.y_uniform = maskgals.r_uniform*np.sin(theta_new)

        # Compute weights to go with these values

        # Chisq weight
        maskgals.cwt = chisq_pdf(maskgals.chisq, ncol)
        maskgals.chisq_pdf = maskgals.cwt

        # Nfw weight
        maskgals.nfw = nfw_pdf(maskgals.r, radfactor=True)

        # luminosity weight

        # We just choose a reference mstar for the normalization code
        mstar = 19.0
        normmag = mstar - 2.5 * np.log10(self.config.lval_reference)
        steps = np.arange(10.0, normmag, 0.01)
        f = schechter_pdf(steps, alpha=self.config.calib_lumfunc_alpha, mstar=mstar)
        n = scipy.integrate.simps(f, steps)
        maskgals.lum_pdf = schechter_pdf(maskgals.m + mstar, mstar=mstar, alpha=self.config.calib_lumfunc_alpha)
        maskgals.lumwt = maskgals.lum_pdf / n

        # zred weight
        maskgals.dzred = np.random.normal(loc=0.0, scale=self.config.maskgal_zred_err, size=maskgals.size)
        maskgals.zwt = (1. / (np.sqrt(2.*np.pi) * self.config.maskgal_zred_err)) * np.exp(-(maskgals.dzred**2.) / (2.*self.config.maskgal_zred_err**2.))

        # And we need the radial function for each set of samples
        for j in range(self.config.maskgal_nsamples):
            indices = np.arange(j * self.config.maskgal_ngals,
                                (j + 1) * self.config.maskgal_ngals)

            # Radial function
            for i, rad in enumerate(radbins):
                inside, = np.where((maskgals.r[indices] <= rad) &
                                   (maskgals.m[indices] < -2.5*np.log10(self.config.lval_reference)))
                maskgals.nin_orig[indices, i] = inside.size

                if self.config.rsig <= 0.0:
                    theta_r = np.ones(self.config.maskgal_ngals)
                else:
                    theta_r = 0.5 + 0.5*erf((rad - maskgals.r[indices]) / (np.sqrt(2.)*self.config.rsig))
                maskgals.theta_r[indices, i] = theta_r

                inside2, = np.where(maskgals.m[indices] < -2.5*np.log10(self.config.lval_reference))
                maskgals.nin[indices, i] = np.sum(theta_r[inside2])

        # And save it

        hdr = fitsio.FITSHDR()
        hdr['version'] = CURRENT_MASKGAL_VERSION
        hdr['r0'] = self.config.percolation_r0
        hdr['beta'] = self.config.percolation_beta
        hdr['stepsize'] = self.config.maskgal_rad_stepsize
        hdr['nmag'] = self.config.nmag
        hdr['ngals'] = self.config.maskgal_ngals
        hdr['chisqmax'] = self.config.chisq_max
        hdr['lvalref'] = self.config.lval_reference
        hdr['extra'] = self.config.maskgal_dmag_extra
        hdr['alpha'] = self.config.calib_lumfunc_alpha
        hdr['rsig'] = self.config.rsig
        hdr['zrederr'] = self.config.maskgal_zred_err

        maskgals.to_fits_file(maskgalfile, clobber=True, header=hdr)


    def set_radmask(self, cluster):
        """
        Assign mask (0: out; 1: in) values to self.maskgals.mark for a given cluster.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to get position/redshift/scaling
        """
        # note this probably can be in the superclass, no?
        ras = cluster.ra + self.maskgals.x/(cluster.mpc_scale)/np.cos(np.radians(cluster.dec))
        decs = cluster.dec + self.maskgals.y/(cluster.mpc_scale)
        self.maskgals.mark = self.compute_radmask(ras,decs)

    def calc_maskcorr(self, mstar, maxmag, limmag):
        """
        Calculate mask correction cpars, a third-order polynomial which describes the
        mask fraction of a cluster as a function of radius.

        Parameters
        ----------
        mstar: `float`
           mstar (mag) at cluster redshift
        maxmag: `float`
           maximum magnitude for use in luminosity function filter
        limmag: `float`
           Survey or local limiting magnitude

        Returns
        -------
        cpars: `np.array`
           Third-order polynomial parameters describing maskfrac as function of radius
        """
        mag_in = self.maskgals.m + mstar
        self.maskgals.refmag = mag_in

        if self.maskgals.limmag[0] > 0.0:
            mag, mag_err = apply_errormodels(self.maskgals, mag_in)

            self.maskgals.refmag_obs = mag
            self.maskgals.refmag_obs_err = mag_err
        else:
            mag = mag_in
            mag_err = 0*mag_in
            raise ValueError('Survey limiting magnitude <= 0!')
            # Raise error here as this would lead to divide by zero if called.

        if (self.maskgals.w[0] < 0) or (self.maskgals.w[0] == 0 and
            np.amax(self.maskgals.m50) == 0):
            theta_i = calc_theta_i(mag, mag_err, maxmag, limmag)
        elif (self.maskgals.w[0] == 0):
            theta_i = calc_theta_i(mag, mag_err, maxmag, self.maskgals.m50)
        else:
            raise Exception('Unsupported mode!')

        p_det = theta_i*self.maskgals.mark
        c = 1 - np.dot(p_det, self.maskgals.theta_r) / self.maskgals.nin[0]

        cpars = np.polyfit(self.maskgals.radbins[0], c, 3)

        return cpars

class HPMask(Mask):
    """
    A class to use a redmapper healpix geometric mask.

    This is described as mask_mode == 3 for compatibility with the old IDL code.
    """

    def __init__(self, config, **kwargs):
        """
        Instantiate an HPMask

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object.  Reads mask from config.maskfile.
        include_maskgals: `bool`, optional
           Also read in the maskgals.  Default is True.
        """
        # record for posterity
        self.maskfile = config.maskfile

        # Check if the file is of healsparse type... if not, raise and suggest
        # the conversion code

        hdr = fitsio.read_header(self.maskfile, ext=1)
        if 'PIXTYPE' not in hdr or hdr['PIXTYPE'] != 'HEALSPARSE':
            raise RuntimeError("Need to specify mask in healsparse format.  See redmapper_convert_mask_to_healsparse.py")

        cov_hdr = fitsio.read_header(self.maskfile, ext='COV')
        nside_coverage = cov_hdr['NSIDE']

        # Which subpixels are we reading?
        if len(config.d.hpix) > 0:
            covpixels = get_healsparse_subpix_indices(config.d.nside, config.d.hpix,
                                                      config.border, nside_coverage)
        else:
            # Read in the whole thing
            covpixels = None

        self.sparse_fracgood = healsparse.HealSparseMap.read(self.maskfile, pixels=covpixels)

        self.nside = self.sparse_fracgood.nside_sparse

        super(HPMask, self).__init__(config, **kwargs)

    def compute_radmask(self, ras, decs):
        """
        Compute the geometric mask value at a list of positions.

        In the footprint is True, outside is False.

        Parameters
        ----------
        ras: `np.array`
           Float array of right ascensions
        decs: `np.array`
           Float array of declinations

        Returns
        -------
        maskvals: `np.array`
           Bool array of True (in footprint) and False (out of footprint) for
           each ra/dec.
        """

        if (ras.size != decs.size):
            raise ValueError("ra, dec must be same length")

        gd, = np.where(np.abs(decs) < 90.0)

        fracgood = np.zeros(ras.size)
        fracgood[gd] = self.sparse_fracgood.get_values_pos(ras[gd], decs[gd], lonlat=True)

        radmask = np.zeros(ras.size, dtype=bool)
        radmask[np.where(fracgood > np.random.rand(ras.size))] = True
        return radmask


def get_mask(config, include_maskgals=True):
    """
    Convenience function to look at a config file and load the appropriate type of mask.

    Uses config.mask_mode to determine mask type and config.maskfile for mask filename,

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    """

    if config.mask_mode == 0:
        # This is no mask!
        # Return a bare object with maskgal functionality
        return Mask(config, include_maskgals=include_maskgals)
    elif config.mask_mode == 3:
        # This is a healpix mask
        #  (don't ask about 1 and 2)
        return HPMask(config, include_maskgals=include_maskgals)

def convert_maskfile_to_healsparse(maskfile, healsparsefile, nsideCoverage, clobber=False):
    """
    Convert an old maskfile to a new healsparsefile

    Parameters
    ----------
    maskfile: `str`
       Input mask file
    healsparsefile: `str`
       Output healsparse file
    nsideCoverage: `int`
       Nside for sparse coverage map
    clobber: `bool`, optional
       Clobber existing healsparse file?  Default is false.
    """

    old_mask, old_hdr = fitsio.read(maskfile, ext=1, header=True, lower=True)

    nside = old_hdr['nside']

    sparseMap = healsparse.HealSparseMap.make_empty(nsideCoverage, nside, old_mask['fracgood'].dtype)
    sparseMap.update_values_pix(old_mask['hpix'], old_mask['fracgood'], nest=old_hdr['nest'])

    sparseMap.write(healsparsefile, clobber=clobber)

