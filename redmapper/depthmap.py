"""Classes to describe a redmapper depth map.

"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import healpy as hp
import numpy as np
from healpy import pixelfunc
import esutil
import scipy.optimize
import healsparse

from .utilities import astro_to_sphere, get_healsparse_subpix_indices
from .catalog import Catalog, Entry

class DepthMap(object):
    """
    A class to use a healpix-based redmapper depth map.

    FIXME: Put a description of the format here.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object, with config.depthfile set
    depthfile: `str`, optional
       Name of depthfile to use instead of config.depthfile
    """
    def __init__(self, config, depthfile=None):
        """
        Instantiate a healpix-based redmapper depth map.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object, with config.depthfile set
        depthfile: `str`, optional
           Name of depthfile to use instead of config.depthfile
        """
        # record for posterity
        if depthfile is None:
            self.depthfile = config.depthfile
        else:
            self.depthfile = depthfile

        hdr = fitsio.read_header(self.depthfile, ext=1)
        if 'PIXTYPE' not in hdr or hdr['PIXTYPE'] != 'HEALSPARSE':
            raise RuntimeError("Need to specify depthfile in healsparse format.  See redmapper_convert_depthfile_to_healsparse.py")

        cov_hdr = fitsio.read_header(self.depthfile, ext='COV')
        nside_coverage = cov_hdr['NSIDE']

        self.nsig = cov_hdr['NSIG']
        self.zp = cov_hdr['ZP']
        self.nband = cov_hdr['NBAND']
        self.w = cov_hdr['W']
        self.eff = cov_hdr['EFF']

        if len(config.d.hpix) > 0:
            covpixels = get_healsparse_subpix_indices(config.d.nside, config.d.hpix,
                                                      config.border, nside_coverage)
        else:
            covpixels = None

        self.sparse_depthmap = healsparse.HealSparseMap.read(self.depthfile, pixels=covpixels)

        self.galfile_nside = config.galfile_nside
        self.config_logger = config.logger
        self.nside = self.sparse_depthmap.nside_sparse
        self.config_area = config.area

        # Record the coverage of the subregion that we read
        self.subpix_nside = config.d.nside
        self.subpix_hpix = config.d.hpix
        self.subpix_border = config.border

    def get_depth_values(self, ras, decs):
        """
        Get the depth values for a set of positions.

        Parameters
        ----------
        ras: `np.array`
           Float array of right ascensions
        decs: `np.array`
           Float array of declinations

        Returns
        -------
        limmag: `np.array`
           Limiting magnitude values
        exptime: `np.array`
           Effective exposure times
        m50: `np.array`
           50% completeness depth values.  Should be same as limmag for now.
        """

        if ras.size != decs.size:
            raise ValueError("ra, dec must be the same length")

        values = self.sparse_depthmap.get_values_pos(ras, np.clip(decs, -90.0, 90.0), lonlat=True)

        bad, = np.where(np.abs(decs) > 90.0)
        values['limmag'][bad] = hp.UNSEEN
        values['exptime'][bad] = hp.UNSEEN
        values['m50'][bad] = hp.UNSEEN

        return (values['limmag'],
                values['exptime'],
                values['m50'])

    def get_fracgoods(self, ras, decs):
        """
        Get the fraction of good coverage of each pixel

        Parameters
        ----------
        ras: `np.array`
           Float array of right ascensions
        decs: `np.array`
           Float array of declinations

        Returns
        -------
        fracgoods: `np.array`
           Float array of fracgoods
        """

        if (ras.size != decs.size):
            raise ValueError("ra, dec must be the same length")

        values = self.sparse_depthmap.get_values_pos(ras, np.clip(decs, -90.0, 90.0), lonlat=True)

        bad, = np.where(np.abs(decs) > 90.0)
        values['fracgood'][bad] = 0.0

        return values['fracgood']

    def calc_maskdepth(self, maskgals, ra, dec, mpc_scale):
        """
        Calculate depth for maskgals structure.

        This will modify maskgals.limmag, maskgals.exptime, maskgals.zp,
        maskgals.nsig.

        Parameters
        ----------
        masgkals: `redmapper.Catalog`
           maskgals catalog
        ra: `float`
           Right ascension to center maskgals
        dec: `float`
           Declination ot center maskgals
        mpc_scale: `float`
           Scaling in Mpc / degree at cluster redshift
        """
        unseen = hp.pixelfunc.UNSEEN

        # compute ra and dec based on maskgals
        ras = ra + (maskgals.x/mpc_scale)/np.cos(dec*np.pi/180.)
        decs = dec + maskgals.y/mpc_scale

        maskgals.w[:] = self.w
        maskgals.eff = None
        maskgals.limmag[:] = unseen
        maskgals.zp[0] = self.zp
        maskgals.nsig[0] = self.nsig

        # Make sure the dec is within range, if we're going toward the pole (in sims)
        gd, = np.where(np.abs(decs) < 90.0)

        maskgals.limmag[gd], maskgals.exptime[gd], maskgals.m50[gd] = self.get_depth_values(ras[gd], decs[gd])

        bd = (maskgals.limmag < 0.0)
        ok = ~bd
        nok = ok.sum()

        if (bd.sum() > 0):
            if (nok >= 3):
                # fill them in
                maskgals.limmag[bd] = np.median(maskgals.limmag[ok])
                maskgals.exptime[bd] = np.median(maskgals.exptime[ok])
                maskgals.m50[bd] = np.median(maskgals.m50[ok])
            elif (nok > 0):
                # fill with mean
                maskgals.limmag[bd] = np.mean(maskgals.limmag[ok])
                maskgals.exptime[bd] = np.mean(maskgals.exptime[ok])
                maskgals.m50[bd] = np.mean(maskgals.m50[ok])
            else:
                # very bad (nok == 0)
                # Set this to 1.0 so it'll get used but will give giant errors.
                # And the cluster should be filtered
                maskgals.limmag[:] = 1.0
                maskgals.exptime[:] = 1000.0
                maskgals.m50[:] = 0.0
                self.config_logger.info("Warning: Bad cluster in bad region...")


    def calc_areas(self, mags):
        """
        Calculate total area from the depth map as a function of magnitude.

        Parameters
        ----------
        mags: `np.array`
           Float array of magnitudes at which to compute area

        Returns
        -------
        areas: `np.array`
           Float array of total areas for each of the mags
        """

        pixsize = hp.nside2pixarea(self.nside, degrees=True)

        if (self.w < 0.0):
            # This is just constant area
            areas = np.zeros(mags.size) + self.config_area
            return areas

        if len(self.subpix_hpix) > 0:
            # for the subregion, we need the area covered in the main pixel
            # I'm not sure what to do about border...but you shouldn't
            # be running this with a subregion with a border
            if self.subpix_border > 0.0:
                raise RuntimeError("Cannot run calc_areas() with a subregion with a border")

            bitShift = 2 * int(np.round(np.log(self.nside / self.subpix_nside) / np.log(2)))
            nFinePerSub = 2**bitShift
            ipnest = np.zeros(0, dtype=np.int64)
            for hpix in self.subpix_hpix:
                ipnest_temp = np.left_shift(hp.ring2nest(self.subpix_nside, hpix), bitShift) + np.arange(nFinePerSub)
                ipnest = np.append(ipnest, ipnest_temp)
        else:
            ipnest = self.sparse_depthmap.valid_pixels

        areas = np.zeros(mags.size)

        values = self.sparse_depthmap.get_values_pix(ipnest)

        gd, = np.where(values['m50'] > 0.0)

        depths = values['m50'][gd]
        st = np.argsort(depths)
        depths = depths[st]

        fracgoods = values['fracgood'][gd[st]]

        inds = np.clip(np.searchsorted(depths, mags) - 1, 1, depths.size - 1)

        lo = (inds < 0)
        areas[lo] = np.sum(fracgoods) * pixsize
        carea = pixsize * np.cumsum(fracgoods)
        areas[~lo] = carea[carea.size - inds[~lo]]

        return areas

# This is incomplete, since I worry the general-use depthmap will be too memory
# intensive for the volume limit mask.  TBD
"""
class MultibandDepthMap(object):

    def __init__(self, config, depthfiles, bands):

        self.nband = len(bands) + 1


        self.depthfile = config.depthfile

        # We start by reading in the primary depth file

        depthinfo, hdr = fitsio.read(self.config.depthfile, ext=1, header=True, lower=True)
        dstr = Catalog(depthinfo)

        mband = Catalog(np.zeros(dstr.size, dtype=[('hpix', 'i8'),
                                                   ('fracgood', 'f4'),
                                                   ('exptime', 'f4', self.nband),
                                                   ('limmag', 'f4', self.nband),
                                                   ('m50', 'f4', self.nband)]))
"""

def convert_depthfile_to_healsparse(depthfile, healsparsefile, nsideCoverage, clobber=False):
    """
    Convert an old depthfile to a new healsparsefile

    Parameters
    ----------
    depthfile: `str`
       Input depth file
    healsparsefile: `str`
       Output healsparse file
    nsideCoverage: `int`
       Nside for sparse coverage map
    clobber: `bool`, optional
       Clobber existing healsparse file?  Default is False.
    """

    old_depth, old_hdr = fitsio.read(depthfile, ext=1, header=True, lower=True)

    nside = old_hdr['nside']

    # Need to remove the HPIX from the dtype

    dtype_new = []
    names = []
    for d in old_depth.dtype.descr:
        if d[0] != 'hpix':
            dtype_new.append(d)
            names.append(d[0])

    sparseMap = healsparse.HealSparseMap.make_empty(nsideCoverage, nside, dtype_new, primary='fracgood')

    old_depth_sub = np.zeros(old_depth.size, dtype=dtype_new)
    for name in names:
        old_depth_sub[name] = old_depth[name]

    sparseMap.update_values_pix(old_depth['hpix'], old_depth_sub, nest=old_hdr['nest'])

    hdr = fitsio.FITSHDR()
    hdr['NSIG'] = old_hdr['NSIG']
    hdr['ZP'] = old_hdr['ZP']
    hdr['NBAND'] = old_hdr['NBAND']
    hdr['W'] = old_hdr['W']
    hdr['EFF'] = old_hdr['EFF']

    sparseMap.metadata = hdr

    sparseMap.write(healsparsefile, clobber=clobber)

