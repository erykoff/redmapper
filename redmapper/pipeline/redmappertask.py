"""Class to run redmapper on a single pixel, for distributed runs.
"""
import os
import numpy as np
import glob

from ..configuration import Configuration
from ..utilities import make_lockfile
from ..run_firstpass import RunFirstPass
from ..run_likelihoods import RunLikelihoods
from ..run_percolation import RunPercolation
from ..run_randoms_zmask import RunRandomsZmask
from ..runcat import RunCatalog
from ..run_zscan import RunZScan

from ..utilities import getMemoryString

class RunRedmapperPixelTask(object):
    """
    Class to run redmapper on a single healpix pixel, for distributed runs.
    """

    def __init__(self, configfile, pixel, nside, path=None):
        """
        Instantiate a RunRedmapperPixelTask.

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename.
        pixel: `int`
           Healpix pixel to run on.
        nside: `int`
           Healpix nside associated with pixel.
        path: `str`, optional
           Output path.  Default is None, use same absolute
           path as configfile.
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """
        Run redmapper on a single healpix pixel.

        This method will check if files already exist, and will
        skip any steps that already exist.  The border radius
        will automatically be calculated based on the richest
        possible cluster at the lowest possible redshift.

        All files will be placed in self.config.outpath (see
        self.__init__)
        """

        # need to think about outpath

        # Make sure all files are here and okay...

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        self.config.check_files(check_zredfile=True, check_bkgfile=True, check_bkgfile_components=True, check_parfile=True, check_zlambdafile=True)

        # Compute the border size

        self.config.border = self.config.compute_border()

        self.config.d.hpix = [self.pixel]
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%d_%05d' % (self.config.outbase, self.nside, self.pixel)

        # Do the run
        self.config.start_file_logging()
        self.config.logger.info("Running redMaPPer on pixel %d" % (self.pixel))

        firstpass = RunFirstPass(self.config)

        if not os.path.isfile(firstpass.filename):
            firstpass.run()
            firstpass.output(savemembers=False, withversion=False)
        else:
            self.config.logger.info("Firstpass file %s already present.  Skipping..." % (firstpass.filename))

        self.config.catfile = firstpass.filename

        # Clear out the firstpass memory
        del firstpass

        like = RunLikelihoods(self.config)

        if not os.path.isfile(like.filename):
            like.run()
            like.output(savemembers=False, withversion=False)
        else:
            self.config.logger.info("Likelihood file %s already present.  Skipping..." % (like.filename))

        self.config.catfile = like.filename

        # Clear out the likelihood memory
        del like

        perc = RunPercolation(self.config)

        if not os.path.isfile(perc.filename):
            perc.run()
            perc.output(savemembers=True, withversion=False)
        else:
            self.config.logger.info("Percolation file %s already present.  Skipping..." % (perc.filename))

        self.config.stop_file_logging()


class RuncatPixelTask(object):
    """
    Class to run richness computation (runcat) on a single healpix pixel, for
    distributed runs.
    """
    def __init__(self, configfile, pixel, nside, path=None):
        """
        Instantiate a RuncatPixelTask.

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename.
        pixel: `int`
           Healpix pixel to run on.
        nside: `int`
           Healpix nside associated with pixel.
        path: `str`, optional
           Output path.  Default is None, use same absolute
           path as configfile.
        percolation_masking: `bool`, optional
           Do percolation masking when computing richnesses
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """
        Run runcat on a single healpix pixel.

        All files will be placed in self.config.outpath (see
        self.__init__)
        """
        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        self.config.check_files(check_zredfile=False, check_bkgfile=True, check_bkgfile_components=False, check_parfile=True, check_zlambdafile=True)

        # Compute the border size

        self.config.border = self.config.compute_border()

        self.config.d.hpix = [self.pixel]
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%d_%05d' % (self.config.outbase, self.nside, self.pixel)

        # Do the run
        self.config.start_file_logging()

        self.config.logger.info("Running runcat on pixel %d" % (self.pixel))

        runcat = RunCatalog(self.config)
        if not os.path.isfile(runcat.filename):
            runcat.run(do_percolation_masking=self.config.runcat_percolation_masking)
            runcat.output(savemembers=True, withversion=True)

        self.config.stop_file_logging()


class RunZmaskPixelTask(object):
    """
    Class to run redmapper zmask randoms on a single healpix pixel, for
    distributed runs.
    """
    def __init__(self, configfile, pixel, nside, path=None):
        """
        Instantiate a RunZmaskPixelTask.

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename.
        pixel: `int`
           Healpix pixel to run on.
        nside: `int`
           Healpix nside associated with pixel.
        path: `str`, optional
           Output path.  Default is None, use same absolute
           path as configfile.
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """
        Run zmask on a single healpix pixel.

        This method will check if files already exist, and will
        skip any steps that already exist.  The border radius
        will automatically be calculated based on the richest
        possible cluster at the lowest possible redshift.

        All files will be placed in self.config.outpath (see
        self.__init__)
        """
        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        self.config.check_files(check_zredfile=False, check_bkgfile=True,
                                check_parfile=True, check_randfile=True)

        # Compute the border size

        self.config.border = self.config.compute_border()

        self.config.d.hpix = [self.pixel]
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%d_%05d' % (self.config.outbase, self.nside, self.pixel)

        self.config.start_file_logging()
        self.config.logger.info("Running zmask on pixel %d" % (self.pixel))

        rand_zmask = RunRandomsZmask(self.config)

        if not os.path.isfile(rand_zmask.filename):
            rand_zmask.run()
            rand_zmask.output(savemembers=False, withversion=False)

        # All done
        self.config.stop_file_logging()


class RunZScanPixelTask(object):
    """Class to run redshift-scanning (zscan) on a single healpix pixel, for
    distributed runs.
    """
    def __init__(self, configfile, pixel, nside, path=None):
        """Instantiate a RunZScanPixelTask.

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename.
        pixel: `int`
           Healpix pixel to run on.
        nside: `int`
           Healpix nside associated with pixel.
        path: `str`, optional
           Output path.  Default is None, use same absolute
           path as configfile.
        percolation_masking: `bool`, optional
           Do percolation masking when computing richnesses
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """Run zscan on a single healpix pixel.

        All files will be placed in self.config.outpath (see
        self.__init__)
        """
        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        self.config.check_files(check_zredfile=True, check_bkgfile=True, check_bkgfile_components=True, check_parfile=True, check_zlambdafile=True)

        # Compute the border size
        self.config.border = self.config.compute_border()

        self.config.d.hpix = [self.pixel]
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%d_%05d' % (self.config.outbase, self.nside, self.pixel)

        # Do the run
        self.config.start_file_logging()

        self.config.logger.info("Running zscan on pixel %d" % (self.pixel))

        runzscan = RunZScan(self.config)
        if not os.path.isfile(runzscan.filename):
            runzscan.run()
            runzscan.output(savemembers=True, withversion=True)

        self.config.stop_file_logging()

