"""Class to run redmagic, scanning over all pixels"""
import os
import numpy as np
import glob
import fitsio

from ..configuration import Configuration
from .redmagic_selector import RedmagicSelector
from ..catalog import Entry
from ..galaxy import GalaxyCatalog
from ..plotting import SpecPlot, NzPlot
from .redmagic_randoms import RedmagicGenerateRandoms
from ..volumelimit import VolumeLimitMaskFixed

class RunRedmagicTask(object):
    """
    Class to run redmagic on a full catalog.
    """

    def __init__(self, configfile, path=None):
        """
        Instantiate a RunRedmagicTask

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename
        path: `str`, optional
l           Output path.  Default is None, use same absolute
           path as configfile.
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)

    def run(self, modes=None, clobber=False, do_plots=True, n_randoms=None, rng=None):
        """
        Run redMaGiC selection over a full catalog.

        The modes are optional, if not specified all the modes
        will be run.

        Parameters
        ----------
        modes: `list`, optional
           List of strings of modes to run
        clobber: `bool`, optional
           Overwrite any existing files.  Default is False.
        do_plots: `bool`, optional
           Make the output plots
        n_randoms: `int`, optional
           If None, then 10x the number of redmagic galaxies are generated.
           If 0, then no randoms are generated.
           If >0, then that many randoms are generated.
        rng : `np.random.RandomState`, optional
           Pre-set random number generator.  Default is None.
        """
        self.config.start_file_logging()

        if rng is None:
            rng = np.random.RandomState()

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        # Prepare the redMaGiC selector
        selector = RedmagicSelector(self.config)

        if modes is None:
            modes = selector.modes

        n_modes = len(modes)

        # Check if files exist, clobber is False
        filenames = [''] * n_modes
        for i, mode in enumerate(modes):
            filenames[i] = self.config.redmapper_filename('redmagic_%s' % (mode), withversion=True)

            if os.path.isfile(filenames[i]) and not clobber:
                raise RuntimeError("redMaGiC file %s already exists, and clobber is False" % (filenames[i]))

        # Loop over all pixels in the galaxy table
        tab = Entry.from_fits_file(self.config.galfile)

        started = [False] * n_modes

        self.config.logger.info("Making redMaGiC selection for %d modes and %d pixels" % (n_modes, tab.hpix.size))
        if self.config.has_truth:
            self.config.logger.info("Using truth information for zspec")

        for i, pix in enumerate(tab.hpix):
            gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                              zredfile=self.config.zredfile,
                                              nside=tab.nside,
                                              hpix=pix,
                                              border=0.0,
                                              truth=self.config.has_truth,
                                              dereddener=self.config.dereddener)

            # Loop over all modes
            for j, mode in enumerate(modes):
                # Select the red galaxies
                red_gals = selector.select_redmagic_galaxies(gals, mode)

                # Spool out redMaGiC galaxies
                if not started[j]:
                    # write a new file (and overwrite if necessary, since we
                    # already did the clobber check)
                    red_gals.to_fits_file(filenames[j], clobber=True)
                    started[j] = True
                else:
                    with fitsio.FITS(filenames[j], mode='rw') as fits:
                        fits[1].append(red_gals._ndarray)

        # Load in catalogs and make plots!
        if do_plots:
            import matplotlib.pyplot as plt

            for j, mode in enumerate(modes):
                gals = GalaxyCatalog.from_fits_file(filenames[j])

                nzplot = NzPlot(self.config, binsize=self.config.redmagic_calib_zbinsize)
                nzplot.plot_redmagic_catalog(gals, mode, selector.calib_data[mode].etamin,
                                             selector.calib_data[mode].n0,
                                             selector.vlim_masks[mode].get_areas(),
                                             zrange=selector.calib_data[mode].zrange,
                                             withversion=True)

                okspec, = np.where(gals.zspec > 0.0)
                if okspec.size > 0:
                    specplot = SpecPlot(self.config)
                    fig = specplot.plot_values(gals.zspec[okspec], gals.zredmagic[okspec],
                                               gals.zredmagic_e[okspec],
                                               name=r'z_{\mathrm{redmagic}}',
                                               title='%s: %3.1f-%02d' %
                                               (mode, selector.calib_data[mode].etamin,
                                                int(selector.calib_data[mode].n0)),
                                               figure_return=True)
                    fig.savefig(self.config.redmapper_filename('redmagic_zspec_%s_%3.1f-%02d' %
                                                               (mode, selector.calib_data[mode].etamin,
                                                                int(selector.calib_data[mode].n0)),
                                                               paths=(self.config.plotpath,),
                                                               withversion=True,
                                                               filetype='png'))
                    plt.close(fig)

        # Need a check on when to kick out
        if (n_randoms is not None and n_randoms == 0):
            # We are not generating randoms
            return

        # Random generation
        for j, mode in enumerate(modes):
            if isinstance(selector.vlim_masks[mode], VolumeLimitMaskFixed):
                self.config.logger.info("Cannot construct randoms for %s, because we don't have geometry/depth info." % (mode))
                continue

            gals = GalaxyCatalog.from_fits_file(filenames[j])

            rand_generator = RedmagicGenerateRandoms(self.config, selector.vlim_masks[mode], gals)
            if n_randoms is None:
                _n_randoms = gals.size * 10
            else:
                _n_randoms = n_randoms

            randfile = self.config.redmapper_filename('redmagic_%s_randoms' % (mode), withversion=True)

            rand_generator.generate_randoms(_n_randoms, randfile, clobber=clobber, rng=rng)

        self.config.stop_file_logging()
