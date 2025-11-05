"""Class for making a redMaGiC galaxy selection."""
from collections import OrderedDict
import os
import numpy as np

import fitsio
import time
import scipy.optimize
import esutil

from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from ..configuration import Configuration
from ..volumelimit import VolumeLimitMask, VolumeLimitMaskFixed
from ..redsequence import RedSequenceColorPar
from ..utilities import CubicSpline

class RedmagicSelector(object):
    """
    Class to select redMaGiC galaxies.
    """
    def __init__(self, conf, vlim_masks=None):
        """
        Instantiate a RedmagicSelector

        Parameters
        ----------
        conf: `redmapper.Configuration` or `str
           Configuration object or config filename
        vlim_masks: `OrderedDict`, optional
           Dictionary of vlim_masks.  Will read in if not set.
        rng : `np.random.RandomState`, optional
            Random number generator.
        """

        if isinstance(conf, str):
            self.config = Configuration(conf)
        else:
            self.config = conf

        self.rng = np.random.RandomState(self.config.randomseed)

        redmagicfilepath = os.path.dirname(self.config.redmagicfile)

        self.calib_data = OrderedDict()
        with fitsio.FITS(self.config.redmagicfile) as fits:
            # Number of modes is number of binary extentions
            self.n_modes = len(fits) - 1

            for ext in range(self.n_modes):
                data = Entry(fits[ext + 1].read())

                try:
                    name = data.name.decode().rstrip()
                except AttributeError:
                    name = data.name.rstrip()

                self.calib_data[name] = data

        self.modes = self.calib_data.keys()

        self.zredstr = RedSequenceColorPar(self.config.parfile, fine=True)

        if vlim_masks is None:
            self.vlim_masks = OrderedDict()

            for mode in self.modes:
                try:
                    vmaskfile = self.calib_data[mode].vmaskfile.decode().rstrip()
                except AttributeError:
                    vmaskfile = self.calib_data[mode].vmaskfile.rstrip()

                if vmaskfile == '':
                    # There is no vmaskfile, we need to do a fixed area one
                    self.vlim_masks[mode] = VolumeLimitMaskFixed(self.config)
                else:
                    vmaskfile = os.path.join(redmagicfilepath,
                                             os.path.basename(vmaskfile))
                    if not os.path.isfile(vmaskfile):
                        raise RuntimeError("Could not find vmaskfile %s.  Must be in same path as redmagic calibration file %s." % (vmaskfile, os.path.abspath(self.config.redmagicfile)))

                    #if os.path.isfile(vmaskfile):
                    #    vmaskfile = vmaskfile
                    #elif os.path.isfile(os.path.join(self.config.configpath, vmaskfile)):
                    #    vmaskfile = os.path.join(self.config.configpath, vmaskfile)
                    #else:
                    #    raise RuntimeError("Could not find vmaskfile %s" % (vmaskfile))
                    self.vlim_masks[mode] = VolumeLimitMask(self.config,
                                                            self.calib_data[mode].etamin,
                                                            vlimfile=vmaskfile)
        else:
            # Check that it's an OrderedDict?  Must it be ordered?
            self.vlim_masks = vlim_masks

        self.spec = None

    def select_redmagic_galaxies(self, gals, mode, return_indices=False):
        """
        Select redMaGiC galaxies from a galaxy catalog, according to the mode.

        Parameters
        ----------
        gals: `redmapper.GalaxyCatalog`
           Catalog of galaxies for redMaPPer
        mode: `str`
           redMaGiC mode to select
        return_indices: `bool`, optional
           Return the indices of the galaxies selected.  Default is False.

        Returns
        -------
        redmagic_catalog: `redmapper.GalaxyCatalog`
           Catalog of redMaGiC galaxies
        indices: `np.ndarray`
           Integer array of selection (if return_indices is True)
        """

        # Check if we have to decode mode (py2/py3)
        if hasattr(mode, 'decode'):
            _mode = mode.decode()
        else:
            _mode = mode

        if _mode not in self.modes:
            raise RuntimeError("Requested redMaGiC mode %s not available." % (_mode))

        calstr = self.calib_data[_mode]

        # Takes in galaxies...
        # Which are the possibly red galaxies?
        lstar_cushion = calstr.lstar_cushion
        z_cushion = calstr.z_cushion
        z_buffer = calstr.buffer

        mstar_init = self.zredstr.mstar(gals.zred_uncorr)

        cut_zrange = [calstr.cost_zrange[0] - z_cushion - z_buffer,
                      calstr.cost_zrange[1] + z_cushion + z_buffer]
        minlstar = np.clip(np.min(calstr.etamin) - lstar_cushion, 0.1, None)

        red_poss_mask = ((gals.zred_uncorr > cut_zrange[0]) &
                         (gals.zred_uncorr < cut_zrange[1]) &
                         (gals.chisq < calstr.maxchi) &
                         (gals.refmag < (mstar_init - 2.5*np.log10(minlstar))))

        # Creates a new catalog...
        zredmagic = np.copy(gals.zred_uncorr)
        zredmagic_e = np.copy(gals.zred_uncorr_e)
        try:
            zredmagic_samp = np.copy(gals.zred_samp)
        except (ValueError, AttributeError) as e:
            # Sample from zred + zred_e (not optimal, for old catalogs)
            zredmagic_samp = np.zeros((zredmagic.size, 1))
            zredmagic_samp[:, 0] = self.rng.normal(loc=zredmagic,
                                                   scale=zredmagic_e,
                                                   size=zredmagic.size)

        spl = CubicSpline(calstr.nodes, calstr.cmax, fixextrap=True)
        chi2max = np.clip(spl(gals.zred_uncorr), 0.1, calstr.maxchi)

        if calstr.run_afterburner:
            spl = CubicSpline(calstr.corrnodes, calstr.bias, fixextrap=True)
            offset = spl(gals.zred_uncorr)
            zredmagic -= offset

            if calstr.apply_afterburner:
                for i in range(zredmagic_samp.shape[1]):
                    zredmagic_samp[:, i] -= offset

            spl = CubicSpline(calstr.corrnodes, calstr.eratio, fixextrap=True)
            zredmagic_e *= spl(gals.zred_uncorr)

        # Compute mstar
        mstar = self.zredstr.mstar(zredmagic)

        # Compute the maximum redshift
        vmask = self.vlim_masks[_mode]
        zmax = vmask.calc_zmax(gals.ra, gals.dec)

        # Do the redmagic selection
        gd, = np.where((gals.chisq < chi2max) &
                       (gals.refmag < (mstar - 2.5 * np.log10(calstr.etamin))) &
                       (zredmagic < zmax) &
                       (red_poss_mask))

        redmagic_catalog = GalaxyCatalog(np.zeros(gd.size, dtype=[('id', 'i8'),
                                                                  ('ra', 'f8'),
                                                                  ('dec', 'f8'),
                                                                  ('refmag', 'f4'),
                                                                  ('refmag_err', 'f4'),
                                                                  ('mag', 'f4', self.config.nmag),
                                                                  ('mag_err', 'f4', self.config.nmag),
                                                                  ('lum', 'f4'),
                                                                  ('zredmagic', 'f4'),
                                                                  ('zredmagic_e', 'f4'),
                                                                  ('zredmagic_samp', 'f4', self.config.zred_nsamp),
                                                                  ('chisq', 'f4'),
                                                                  ('zspec', 'f4')]))

        if gd.size == 0:
            if return_indices:
                return redmagic_catalog, gd
            else:
                return redmagic_catalog

        redmagic_catalog.id = gals.id[gd]
        redmagic_catalog.ra = gals.ra[gd]
        redmagic_catalog.dec = gals.dec[gd]
        redmagic_catalog.refmag = gals.refmag[gd]
        redmagic_catalog.refmag_err = gals.refmag_err[gd]
        redmagic_catalog.mag[:, :] = gals.mag[gd, :]
        redmagic_catalog.mag_err[:, :] = gals.mag_err[gd, :]
        redmagic_catalog.zredmagic = zredmagic[gd]
        redmagic_catalog.zredmagic_e = zredmagic_e[gd]
        redmagic_catalog.zredmagic_samp = zredmagic_samp[gd, :]
        redmagic_catalog.chisq = gals.chisq[gd]

        # Compute the luminosity
        redmagic_catalog.lum = 10.**((mstar[gd] - redmagic_catalog.refmag) / 2.5)

        # In the future, add absolute magnitude calculations, but that will
        # require some k-corrections.

        # Compute the zspec (check this)
        if 'ztrue' in gals.dtype.names:
            # We have truth zspec
            redmagic_catalog.zspec = gals.ztrue[gd]
        elif 'zspec' in gals.dtype.names:
            # We have already done a zspec match
            redmagic_catalog.zspec = gals.zspec[gd]
        else:
            # We need to do a zspec match here
            if self.spec is None:
                self.config.logger.info("Reading in spectroscopic information...")

                self.spec = GalaxyCatalog.from_fits_file(self.config.specfile)
                use, = np.where(self.spec.z_err < 0.001)
                self.spec = self.spec[use]
                self.config.logger.info("Done reading in spectroscopic information.")

            redmagic_catalog.zspec[:] = -1.0
            i0, i1, dists = self.spec.match_many(redmagic_catalog.ra, redmagic_catalog.dec, 3./3600., maxmatch=1)
            redmagic_catalog.zspec[i0] = self.spec.z[i1]

        if return_indices:
            return redmagic_catalog, gd
        else:
            return redmagic_catalog
