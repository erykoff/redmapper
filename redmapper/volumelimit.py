"""Classes to describe the volume limit mask.
"""
import fitsio
import hpgeom as hpg
import numpy as np
import esutil
import os
import healsparse

from .catalog import Catalog, Entry
from .depthmap import DepthMap
from .redsequence import RedSequenceColorPar
from .utilities import astro_to_sphere, get_healsparse_subpix_indices


class VolumeLimitMask(object):
    """
    A class to describe a volume limit mask.

    This is based on combining the red sequence model with depth maps in
    different bands, to determine the highest redshift that a typical
    red-sequence galaxy can be observed with the specified luminosity.

    """

    def __init__(self, config, vlim_lstar, vlimfile=None, use_geometry=False, withversion=True):
        """
        Instantiate a VolumeLimitMask

        If the mask described by maskfile already exists, it will be read in
        directly.  If it does not exist, it will be generated from the depth
        files described in the config parameters, and then stored in vlimfile.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        vlim_lstar: `float`
           Luminosity cutoff (units of L*) for red-sequence volume limit
           computation
        vlimfile: `str`, optional
           Filename to store volume limit mask.  Default is None, which
           means generate the filename of the 'vlim_zmask' type.
        use_geometry: `bool`, optional
           Use the geometric mask info only.  Only use if necessary.
           Default is False.
        withversion: `bool`, optional
           Output filename with redmapper version string.
        """
        self.config = config

        if hasattr(vlim_lstar, "__len__"):
            if len(vlim_lstar) > 1:
                raise ValueError("Can only use single value vlim_lstar.")
            vlim_lstar = vlim_lstar[0]
        self.vlim_lstar = vlim_lstar

        if vlimfile is None:
            self.vlimfile = self.config.redmapper_filename('vl%02d_vlim_zmask' %
                                                            (int(self.vlim_lstar*10)),
                                                           withversion=withversion)
        else:
            self.vlimfile = vlimfile

        if os.path.isfile(self.vlimfile):
            self._read_mask()
        else:
            if use_geometry:
                self._build_geometry_mask()
            else:
                self._build_mask()
            self._read_mask()

    def _read_mask(self):
        """
        Read an existing volume-limit mask into the VolumeLimitMask structure.
        """

        hdr = fitsio.read_header(self.vlimfile, ext=1)
        if 'PIXTYPE' not in hdr or hdr['PIXTYPE'] != 'HEALSPARSE':
            raise RuntimeError("Need to specify vlimfile in healsparse format.")

        cov_hdr = fitsio.read_header(self.vlimfile, ext='COV')
        nside_coverage = cov_hdr['NSIDE']

        if len(self.config.d.hpix) > 0:
            covpixels = get_healsparse_subpix_indices(self.config.d.nside, self.config.d.hpix,
                                                      self.config.border, nside_coverage)
        else:
            covpixels = None

        self.sparse_vlimmap = healsparse.HealSparseMap.read(self.vlimfile, pixels=covpixels)

        self.nside = self.sparse_vlimmap.nside_sparse
        self.subpix_nside = self.config.d.hpix
        self.subpix_hpix = self.config.d.nside
        self.subpix_border = self.config.border

    def _build_mask(self):
        """
        Build a VolumeLimitMask from the parameters in the config file, and
        store the mask in self.vlimfile
        """

        # Make some checks to make sure we can build a volume limit mask
        if self.config.depthfile is None or not os.path.isfile(self.config.depthfile):
            raise RuntimeError("Cannot create a volume limit mask without a depth file")
        if self.config.maskfile is None or not os.path.isfile(self.config.maskfile):
            raise RuntimeError("Cannot create a volume limit mask without a maskfile")
        for fname in self.config.vlim_depthfiles:
            if not os.path.isfile(fname):
                raise RuntimeError("Could not find specified vlim_depthfile %s" % (fname))

        # Read in the red-sequence parameters
        zredstr = RedSequenceColorPar(self.config.parfile, fine=True)

        # create the redshift bins
        zbinsize = 0.001 # arbitrary fine bin
        nzbins = int(np.ceil((self.config.zrange[1] - self.config.zrange[0]) / zbinsize))
        # Note that we want to start one step above the low redshift range
        zbins = np.arange(nzbins) * zbinsize + self.config.zrange[0] + zbinsize

        # magnitude limits
        limmags = zredstr.mstar(zbins) - 2.5 * np.log10(self.vlim_lstar)

        # get the reference index
        ref_ind = self.config.bands.index(self.config.refmag)

        # Read in the primary depth structure
        sparse_mask = healsparse.HealSparseMap.read(self.config.maskfile)
        sparse_depthmap = healsparse.HealSparseMap.read(self.config.depthfile)

        dtype_vlimmap = [('fracgood', 'f4'),
                         ('zmax', 'f4')]

        # Create the vlim map here and fill it below.
        sparse_vlimmap = healsparse.HealSparseMap.make_empty(sparse_mask.nside_coverage,
                                                             sparse_mask.nside_sparse,
                                                             dtype=dtype_vlimmap,
                                                             primary='fracgood')

        validPixels = sparse_mask.valid_pixels
        depthValues = sparse_depthmap.get_values_pix(validPixels, nside=sparse_mask.nside_sparse)
        vlimmap = np.zeros(validPixels.size, dtype=dtype_vlimmap)
        vlimmap['fracgood'] = sparse_mask[validPixels]

        lo, = np.where(depthValues['m50'] <= limmags.min())
        vlimmap['zmax'][lo] = zbins.min()
        hi, = np.where(depthValues['m50'] >= limmags.max())
        vlimmap['zmax'][hi] = zbins.max()
        mid, = np.where((depthValues['m50'] > limmags.min()) & (depthValues['m50'] < limmags.max()))
        if mid.size > 0:
            l = np.searchsorted(limmags, depthValues['m50'][mid], side='right')
            vlimmap['zmax'][mid] = zbins[l]

        # Read in any additional depth maps
        for i, depthfile in enumerate(self.config.vlim_depthfiles):
            sparse_depthmap2, hdr2 = healsparse.HealSparseMap.read(depthfile, header=True)

            depthValues2 = sparse_depthmap2.get_values_pix(validPixels, nside=sparse_mask.nside_sparse)

            nsig = hdr2['NSIG']
            zp = hdr2['ZP']

            # find mag name thing...
            # Note this is validated in the config read
            map_ind = self.config.bands.index(self.config.vlim_bands[i])

            # Only calculate with the values that are in the map.
            use = (depthValues2["fracgood"] > 0.0)

            n2 = self.config.vlim_nsigs[i]**2.
            flim_in = 10.**((depthValues2['limmag'][use] - zp) / (-2.5))
            fn = np.clip((flim_in**2. * depthValues2['exptime'][use]) / (nsig**2.) - flim_in, 0.001, None)
            flim_mask = (n2 + np.sqrt(n2**2. + 4.*depthValues2['exptime'][use] * n2 * fn)) / (2.*depthValues2['exptime'][use])

            lim_mask = np.zeros(vlimmap.size)
            lim_mask[use] = zp - 2.5*np.log10(flim_mask)

            zinds = np.searchsorted(zredstr.z, zbins, side='right')

            limmags_temp = zredstr.mstar(zbins) - 2.5*np.log10(self.vlim_lstar)
            refmag_lim = limmags_temp.copy()

            if (map_ind == ref_ind):
                self.config.logger.info('Warning: vlim_band %s is the same as the reference band!  Skipping...' % (self.config.vlim_bands[i]))
            else:
                if map_ind < ref_ind:
                    # Need to go blueward
                    for jj in range(ref_ind - 1, map_ind - 1, -1):
                        limmags_temp += (zredstr.c[zinds, jj] + zredstr.slope[zinds, jj] * (refmag_lim - zredstr.pivotmag[zinds]))
                else:
                    # Need to go redward
                    for jj in range(ref_ind, map_ind):
                        limmags_temp -= (zredstr.c[zinds, jj] + zredstr.slope[zinds, jj] * (refmag_lim - zredstr.pivotmag[zinds]))

            # adjust zmax with zmax_temp
            zmax_temp = np.zeros(vlimmap.size)

            lo, = np.where(lim_mask <= limmags_temp.min())
            zmax_temp[lo] = zbins.min()
            hi, = np.where(lim_mask >= limmags_temp.max())
            zmax_temp[hi] = zbins.max()
            mid, = np.where((lim_mask > limmags_temp.min()) & (lim_mask < limmags_temp.max()))
            if mid.size > 0:
                l = np.clip(np.searchsorted(limmags_temp, lim_mask[mid], side='right'), 0, zbins.size - 1)
                zmax_temp[mid] = zbins[l]

            limited, = np.where(zmax_temp < vlimmap['zmax'])
            vlimmap['zmax'][limited] = zmax_temp[limited]

        # Here is where we fill the pixels in the volume limit map.
        gd, = np.where(vlimmap['zmax'] > zbins[0])

        sparse_vlimmap.update_values_pix(validPixels[gd], vlimmap[gd])

        sparse_vlimmap.write(self.vlimfile)

    def _build_geometry_mask(self):
        """
        Build a VolumeLimitMask from the geometric mask from the config file,
        and store the mask in self.vlimfile
        """

        if self.config.maskfile is None or not os.path.isfile(self.config.maskfile):
            raise RuntimeError("Cannot create a geometry volume limit mask without a mask file")

        sparse_mask = healsparse.HealSparseMap.read(self.config.maskfile)

        dtype_vlimmap = [('fracgood', 'f4'),
                         ('zmax', 'f4')]

        sparse_vlimmap = healsparse.HealSparseMap.make_empty(sparse_mask.nside_coverage,
                                                             sparse_mask.nside_sparse,
                                                             dtype=dtype_vlimmap,
                                                             primary='fracgood')

        validPixels = sparse_mask.valid_pixels
        maskValues = sparse_mask.get_values_pix(validPixels)
        vlimmap = np.zeros(validPixels.size, dtype=dtype_vlimmap)
        vlimmap['fracgood'] = maskValues
        vlimmap['zmax'] = self.config.zrange[1]

        sparse_vlimmap.update_values_pix(validPixels, vlimmap)

        sparse_vlimmap.write(self.vlimfile)

    def calc_zmax(self, ras, decs, get_fracgood=False):
        """
        Calculate the maximum redshifts associated with a set of ra/decs.

        Parameters
        ----------
        ras: `np.array` or `float`
           Float array of right ascensions
        decs: `np.array` or `float`
           Float array of declinations
        get_fracgood: `bool`, optional
           Also retrieve the fracgood pixel coverage.  Default is False.

        Returns
        -------
        zmax: `np.array`
           Float array of maximum redshifts
        fracgood: `np.array`
           Float array of fracgood, if get_fracgood=True
        """

        if (len(ras) != len(decs)):
            raise ValueError("ras, decs must be same length")

        values = self.sparse_vlimmap.get_values_pos(ras, decs, lonlat=True)

        bad, = np.where(np.abs(decs) > 90.0)
        values['zmax'][bad] = hpg.UNSEEN
        values['fracgood'][bad] = 0.0

        if not get_fracgood:
            return np.clip(values['zmax'], 0.0, None)
        else:
            return (np.clip(values['zmax'], 0.0, None), values['fracgood'])

    def get_areas(self):
        """
        Retrieve the area structure (area as a function of redshift) associated
        with the volume-limit mask.

        Returns
        -------
        astr: `redmapper.Catalog`
           Area structure catalog, with .z and .area
        """

        zbinsize = self.config.area_finebin
        zbins = np.arange(self.config.zrange[0], self.config.zrange[1], zbinsize)

        astr = Catalog(np.zeros(zbins.size, dtype=[('z', 'f4'),
                                                   ('area', 'f4')]))
        astr.z = zbins

        pixsize = hpg.nside_to_pixel_area(self.nside, degrees=True)

        validPixels = self.sparse_vlimmap.valid_pixels
        zmax = self.sparse_vlimmap.get_values_pix(validPixels)['zmax']
        st = np.argsort(zmax)

        fracgoods = self.sparse_vlimmap.get_values_pix(validPixels)['fracgood'][st]

        inds = np.searchsorted(zmax[st], zbins, side='right')

        lo = (inds <= 0)
        astr.area[lo] = np.sum(fracgoods.astype(np.float64)) * pixsize

        if np.sum(~lo) > 0:
            carea = pixsize * np.cumsum(fracgoods, dtype=np.float64)
            astr.area[~lo] = carea[carea.size - inds[~lo]]

        return astr


class VolumeLimitMaskFixed(object):
    """
    A class to describe a volume limit mask with a fixed redshift maximum.

    This class is used as a placeholder when there is no depth information to
    construct a true VolumeLimitMask.
    """

    def __init__(self, config):
        """
        Instantiate a VolumeLimitMaskFixed

        The maximum redshift is set by config.zrange[1] (the max redshift in
        the config file).

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        """
        self.z_max = config.zrange[1]
        self.zrange = config.zrange
        self.zbinsize = config.area_finebin
        self.area = config.area

    def calc_zmax(self, ra, dec):
        """
        Calculate the maximum redshifts associated with a set of ra/decs.

        Parameters
        ----------
        ra: `np.array` or `float`
           Float array of right ascensions
        dec: `np.array` or `float`
           Float array of right ascensions

        Returns
        -------
        zmax: `float`
           Maximum redshift from the config file.
        """

        return self.z_max

    def get_areas(self):
        """
        Retrieve the area structure (area as a function of redshift) associated
        with the fixed-redshift mask.

        Returns
        -------
        astr: `redmapper.Catalog`
           Area structure catalog, with .z and .area
        """

        zbins = np.arange(self.zrange[0], self.zrange[1], self.zbinsize)

        astr = Catalog(np.zeros(zbins.size, dtype=[('z', 'f4'),
                                                   ('area', 'f4')]))
        astr.z = zbins

        astr.area = self.area

        return astr
