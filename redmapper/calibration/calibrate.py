"""Class to run the full red-sequence calibration
"""
import os
import numpy as np
import fitsio
import copy
import re

from ..configuration import Configuration
from ..color_background import ColorBackgroundGenerator
from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from .selectspecred import SelectSpecRedGalaxies
from .selectspecseeds import SelectSpecSeeds
from .redsequencecal import RedSequenceCalibrator
from .centeringcal import WcenCalibrator
from .zlambdacal import ZLambdaCalibrator
from .prepmembers import PrepMembers
from ..zred_runner import ZredRunCatalog, ZredRunPixels
from ..background import BackgroundGenerator, ZredBackgroundGenerator
from ..redmapper_run import RedmapperRun
from ..zlambda import ZlambdaCorrectionPar
from ..plotting import SpecPlot
from ..mask import get_mask
from ..run_colormem import RunColormem
from ..utilities import getMemoryString
from .._version import __version__

class RedmapperCalibrator(object):
    """
    Class to perform red-sequence calibration
    """

    def __init__(self, conf):
        """
        Instantiate a RedmapperCalibrator

        Parameters
        ----------
        conf: `str` or `redmapper.Calibration`
           Configuration yaml file or configuration object
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        print("Calibrating with version %s" % (__version__))

    def run(self):
        """
        Run the red-sequence calibration.
        """

        # Select the red galaxies to start
        self.config.redgalfile = self.config.redmapper_filename('zspec_redgals')
        self.config.redgalmodelfile = self.config.redmapper_filename('zspec_redgals_model')

        if os.path.isfile(self.config.redgalfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.redgalfile))
        else:
            self.config.logger.info("Selecting red galaxies from spectra...")
            selred = SelectSpecRedGalaxies(self.config)
            selred.run()

        # Make a color background
        self.config.bkgfile_color = self.config.redmapper_filename('bkg_color')

        if os.path.isfile(self.config.bkgfile_color):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.bkgfile_color))
        else:
            self.config.logger.info("Constructing color background...")
            cbg = ColorBackgroundGenerator(self.config)
            cbg.run()

        # Generate maskgals
        self.config.maskgalfile = self.config.redmapper_filename('maskgals')

        if os.path.isfile(self.config.maskgalfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.maskgalfile))
        else:
            self.config.logger.info("Constructing maskgals...")
            # This will generate the maskgalfile if it isn't found
            mask = get_mask(self.config, include_maskgals=False)
            mask.gen_maskgals(self.config.maskgalfile)

        # Do the color-lambda training.
        self.config.zmemfile = self.config.redmapper_filename('iter0_colormem_pgt%4.2f_lamgt%02d' % (self.config.calib_pcut, self.config.calib_colormem_minlambda))

        if os.path.isfile(self.config.zmemfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.zmemfile))
        else:
            self.config.logger.info("Doing color-lambda training...")
            rcm = RunColormem(self.config)
            rcm.run()
            rcm.output_training()

        # Generate the spec seed file
        self.config.seedfile = self.config.redmapper_filename('specseeds_train')

        if os.path.isfile(self.config.seedfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.seedfile))
        else:
            self.config.logger.info("Generating spectroscopic seeds (training spec)...")
            sss = SelectSpecSeeds(self.config)
            sss.run(usetrain=True)
        calib_iteration = RedmapperCalibrationIteration(self.config)

        for iteration in range(1, self.config.calib_niter + 1):
            # Run the calibration iteration
            calib_iteration.run(iteration)

            # Clean out the members
            # Note that the outbase is still the modified version
            redmapper_name = 'zmem_pgt%4.2f_lamgt%02d' % (self.config.calib_pcut, int(self.config.calib_minlambda))
            self.config.zmemfile = self.config.redmapper_filename(redmapper_name)
            if os.path.isfile(self.config.zmemfile):
                self.config.logger.info("%s already there.  Skipping..." % (self.config.zmemfile))
            else:
                self.config.logger.info("Preparing members for next calibration...")
                ## FIXME
                prep_members = PrepMembers(self.config)
                prep_members.run('z_init')

            # Reset outbase here
            self.config.d.outbase = self.config.outbase

            if iteration == 1:
                # If this is the first iteration, generate a new seedfile
                new_seedfile = self.config.redmapper_filename('cut_specseeds')
                if os.path.isfile(new_seedfile):
                    self.config.logger.info("%s already there.  Skipping..." % (new_seedfile))
                else:
                    self.config.logger.info("Generating cut specseeds...")
                    seeds = GalaxyCatalog.from_fits_file(self.config.seedfile)
                    cat = GalaxyCatalog.from_fits_file(self.config.catfile)

                    use, = np.where(cat.Lambda > self.config.percolation_minlambda)

                    i0, i1, dd = seeds.match_many(cat.ra[use], cat.dec[use], 0.5/3600., maxmatch=1)
                    seeds.to_fits_file(new_seedfile, indices=i1)

                self.config.seedfile = new_seedfile

        # Prep for final iteration

        # Generate full specseeds...
        self.config.seedfile = self.config.redmapper_filename('specseeds')

        if os.path.isfile(self.config.seedfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.seedfile))
        else:
            self.config.logger.info("Generating spectroscopic seeds (full spec)...")
            sss = SelectSpecSeeds(self.config)
            sss.run(usetrain=False)

        calib_iteration_final = RedmapperCalibrationIterationFinal(self.config)
        calib_iteration_final.run(self.config.calib_niter)

        # Output a configuration file
        new_bkgfile, new_zreds = self.output_configuration()

        # Generate a full background
        if new_bkgfile:
            # Note that the config file has been updated!
            self.config.logger.info("Running full background...")

            self.config.d.hpix = []
            self.config.d.nside = 0

            bkg_gen = BackgroundGenerator(self.config)
            bkg_gen.run(deepmode=True)
            self.config.logger.info("Remember to run zreds and zred background before running the full cluster finder.")
        else:
            if new_zreds:
                self.config.logger.info("Remember to run zreds before running the full cluster finder.  No need to recompute the background.")
            else:
                self.config.logger.info("Calibration done on full footprint, so background and zreds are already available.")

        # Run halos if desired (later)

    def output_configuration(self):
        """
        Output a configuration yaml file that has all the calibration products
        inserted.
        """

        new_zreds = False
        new_bkgfile = False

        # Compute the path that the cluster finder will be run in

        calpath = os.path.abspath(self.config.outpath)
        calparent = os.path.normpath(os.path.join(calpath, os.pardir))
        calpath_only = os.path.basename(os.path.normpath(calpath))

        if calpath_only == 'cal':
            runpath_only = 'run'
        elif 'cal_' in calpath_only:
            runpath_only = calpath_only.replace('cal_', 'run_')
        elif '_cal' in calpath_only:
            runpath_only = calpath_only.replace('_cal', '_run')
        else:
            runpath_only = '%s_run' % (calpath_only)

        runpath = os.path.join(calparent, runpath_only)

        if not os.path.isdir(runpath):
            os.makedirs(runpath)

        # Make sure we have absolute paths for everything that is defined
        self.config.galfile = os.path.abspath(self.config.galfile)
        self.config.specfile = os.path.abspath(self.config.specfile)

        outbase_cal = self.config.outbase

        # Compute the string to go with the final iteration
        iterstr = '%s_iter%d' % (outbase_cal, self.config.calib_niter)

        # Compute the new outbase
        if '_cal' in outbase_cal:
            outbase_run = self.config.outbase.replace('_cal', '_run')
        else:
            outbase_run = '%s_run' % (outbase_cal)

        self.config.outbase = outbase_run

        self.config.parfile = os.path.abspath(os.path.join(self.config.outpath,
                                                           '%s_pars.fit' % (iterstr)))

        # This is the default, unless we want to recompute
        self.config.bkgfile = os.path.abspath(os.path.join(calpath,
                                                           '%s_bkg.fit' % (iterstr)))

        # If we calibrated on the full survey, then we have the zredfile already
        if self.config.nside == 0:
            self.config.zredfile = os.path.abspath(os.path.join(calpath,
                                                                '%s' % (iterstr),
                                                                '%s_zreds_master_table.fit' % (iterstr)))
        else:
            new_zreds = True

            galfile_base = os.path.basename(self.config.galfile)
            zredfile = galfile_base.replace('_master', '_zreds_master')
            self.config.zredfile = os.path.abspath(os.path.join(runpath,
                                                                'zreds',
                                                                zredfile))
            if self.config.calib_make_full_bkg:
                new_bkgfile = True
                self.config.bkgfile = os.path.abspath(os.path.join(runpath, '%s_bkg.fit' % (outbase_run)))


        self.config.zlambdafile = os.path.abspath(os.path.join(calpath, '%s_zlambda.fit' % (iterstr)))
        self.config.wcenfile = os.path.abspath(os.path.join(calpath, '%s_wcen.fit' % (iterstr)))
        self.config.bkgfile_color = os.path.abspath(self.config.bkgfile_color)
        self.config.catfile = None
        self.config.maskgalfile = os.path.abspath(self.config.maskgalfile)
        self.config.redgalfile = os.path.abspath(self.config.redgalfile)
        self.config.redgalmodelfile = os.path.abspath(self.config.redgalmodelfile)
        self.config.seedfile = None
        self.config.zmemfile = None

        # and reset the running values
        self.config.nside = 0
        self.config.hpix = []
        self.config.border = 0.0

        # And finally, if we have a depth map we don't need the area...
        if self.config.depthfile is not None and os.path.isfile(self.config.depthfile):
            self.config.area = None

        self.config.output_yaml(os.path.join(runpath, 'run_default.yml'))

        return (new_bkgfile, new_zreds)

class RedmapperCalibrationIteration(object):
    """
    Class to perform a single iteration of the redmapper calibration.
    """

    def __init__(self, config):
        """
        Instantiate a RedmapperCalibrationIteration

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        """
        self.config = config

    def run(self, iteration):
        """
        Run a single iteration of the redmapper calibration.

        Files will include the name "iter%{iteration}".  Extra files are
        created in iteration 1 to start the calibration of the centering model.
        (Note that "iteration 0" is the color-based run that happens before the
        first iteration).

        Parameters
        ----------
        iteration: `int`
           Iteration number.
        """

        # Generate the name of the parfile
        self.config.d.outbase = '%s_iter%d' % (self.config.outbase, iteration)

        self.config.parfile = self.config.redmapper_filename('pars')

        # Run the red sequence calibration
        if os.path.isfile(self.config.parfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.parfile))
        else:
            self.config.logger.info("Running red sequence calibration...")
            redsequencecal = RedSequenceCalibrator(self.config, self.config.zmemfile)
            redsequencecal.run()

        # Make the "sz" file (I think maybe I can skip this)

        # Compute zreds based on the type of galaxy file
        if self.config.galfile_pixelized:
            self.config.zredfile = self.config.redmapper_filename('zreds_master_table', paths=(self.config.d.outbase,))
        else:
            self.config.zredfile = self.config.redmapper_filename('zreds')

        if os.path.isfile(self.config.zredfile):
            self.config.logger.info("%s already there.  Skipping..." % (self.config.zredfile))
        else:
            self.config.logger.info("Computing zreds for all galaxies in the training region...")
            self.config.d.hpix = self.config.hpix
            self.config.d.nside = self.config.nside

            if self.config.galfile_pixelized:
                zredRunpix = ZredRunPixels(self.config)
                zredRunpix.run()
            else:
                zredRuncat = ZredRunCatalog(self.config)
                zredRuncat.run(self.config.galfile, self.config.zredfile)

        # Compute the chisq background
        self.config.bkgfile = self.config.redmapper_filename('bkg')

        # There are two extensions here, so we need to be careful
        calc_bkg = False
        calc_zred_bkg = False
        if not os.path.isfile(self.config.bkgfile):
            calc_bkg = True
            calc_zred_bkg = True
        else:
            with fitsio.FITS(self.config.bkgfile) as fits:
                if 'CHISQBKG' not in [ext.get_extname() for ext in fits[1: ]]:
                    calc_bkg = True
                else:
                    self.config.logger.info("Found CHISQBKG in %s.  Skipping..." % (self.config.bkgfile))
                if 'ZREDBKG' not in [ext.get_extname() for ext in fits[1: ]]:
                    calc_zred_bkg = True
                else:
                    self.config.logger.info("Found ZREDBKG in %s.  Skipping..." % (self.config.bkgfile))

        if calc_bkg:
            self.config.logger.info("Generating chisq background...")
            bkg_gen = BackgroundGenerator(self.config)
            bkg_gen.run()

        if calc_zred_bkg:
            self.config.logger.info("Generating zred background...")
            zbkg_gen = ZredBackgroundGenerator(self.config)
            zbkg_gen.run()

        # Set the centering function
        centerclass = copy.deepcopy(self.config.centerclass)

        if iteration == 1:
            self.config.centerclass = self.config.firstpass_centerclass
        else:
            self.config.centerclass = 'CenteringWcenZred'

        # Generate the zreds for the specseeds
        # This is the iteration seedfile
        iter_seedfile = self.config.redmapper_filename('specseeds')

        if os.path.isfile(iter_seedfile):
            self.config.logger.info('%s already there.  Skipping...' % (iter_seedfile))
        else:
            self.config.logger.info("Generating iteration seedfile...")
            seedzredfile = self.config.redmapper_filename('specseeds_zreds')
            zredRuncat = ZredRunCatalog(self.config)
            zredRuncat.run(self.config.seedfile, seedzredfile)

            # Now combine seeds with zreds
            seeds = Catalog.from_fits_file(self.config.seedfile, ext=1)
            zreds = Catalog.from_fits_file(seedzredfile, ext=1)

            seeds.zred = zreds.zred
            seeds.zred_e = zreds.zred_e
            seeds.zred_chisq = zreds.chisq

            seeds.to_fits_file(iter_seedfile)


        # Run the cluster finder in specmode (And consolidate likelihoods)
        finalfile = self.config.redmapper_filename('final')

        if os.path.isfile(finalfile):
            self.config.logger.info('%s already there.  Skipping...' % (finalfile))
        else:
            self.config.logger.info("Running redmapper in specmode with seeds...")
            self.config.zlambdafile = None

            redmapper_run = RedmapperRun(self.config)
            catfile, likefile = redmapper_run.run(specmode=True, keepz=True, consolidate_like=True, seedfile=iter_seedfile, cleaninput=True)
            # check that catfile is the same as finalfile?
            if catfile != finalfile:
                raise RuntimeError("The output catfile %s should be the same as finalfile %s" % (catfile, finalfile))

        # If it's the first iteration, calibrate random and satellite w functions
        if iteration == 1:
            sublikefile = self.config.redmapper_filename('sub_like')

            if os.path.isfile(sublikefile):
                self.config.logger.info('%s already there.  Skipping...' % (sublikefile))
            else:
                # Generate a subset of the likelihood file...
                # Read these as GalaxyCatalogs to do matching
                lcat = GalaxyCatalog.from_fits_file(self.config.redmapper_filename('like'))
                pcat = GalaxyCatalog.from_fits_file(finalfile)

                use, = np.where(pcat.Lambda > self.config.percolation_minlambda)

                # matching...
                i0, i1, dd = lcat.match_many(pcat.ra[use], pcat.dec[use], 0.5/3600., maxmatch=1)

                sublcat = lcat[i1]

                sublcat.to_fits_file(sublikefile)

            outbase = self.config.d.outbase

            self.config.d.outbase = '%s_rand' % (outbase)
            catfile_for_rand_calib = self.config.redmapper_filename('final')
            if os.path.isfile(catfile_for_rand_calib):
                self.config.logger.info('%s already there.  Skipping...' % (catfile_for_rand_calib))
            else:
                self.config.logger.info("Running percolation for random centers...")
                self.config.catfile = sublikefile
                self.config.centerclass = 'CenteringRandom'

                redmapper_run = RedmapperRun(self.config)
                redmapper_run.run(check=True, percolation_only=True, keepz=True, cleaninput=True)

            self.config.d.outbase = '%s_randsat' % (outbase)
            catfile_for_randsat_calib = self.config.redmapper_filename('final')
            if os.path.isfile(catfile_for_randsat_calib):
                self.config.logger.info('%s already there.  Skipping...' % (catfile_for_randsat_calib))
            else:
                self.config.logger.info("Running percolation for random satellite centers...")
                self.config.catfile = sublikefile
                self.config.centerclass = 'CenteringRandomSatellite'

                redmapper_run = RedmapperRun(self.config)
                redmapper_run.run(check=True, percolation_only=True, keepz=True, cleaninput=True)

            # Reset outbase
            self.config.d.outbase = outbase
        else:
            catfile_for_rand_calib = None
            catfile_for_randsat_calib = None

        # Calibrate wcen
        self.config.centerclass = centerclass
        self.config.catfile = finalfile
        self.config.wcenfile = self.config.redmapper_filename('wcen')

        if os.path.isfile(self.config.wcenfile):
            self.config.logger.info('%s already there.  Skipping...' % (self.config.wcenfile))
        else:
            self.config.logger.info("Calibrating Wcen")
            wc = WcenCalibrator(self.config, iteration,
                                randcatfile=catfile_for_rand_calib,
                                randsatcatfile=catfile_for_randsat_calib)
            wc.run()

        self.config.set_wcen_vals()

        # Calibrate zlambda correction
        self.config.zlambdafile = self.config.redmapper_filename('zlambda')

        if os.path.isfile(self.config.zlambdafile):
            self.config.logger.info('%s already there.  Skipping...' % (self.config.zlambdafile))
        else:
            self.config.logger.info("Calibrating zlambda corrections...")
            zlambdacal = ZLambdaCalibrator(self.config, corrslope=False)
            zlambdacal.run()

        # Make pretty plots showing performance
        spec_plot = SpecPlot(self.config)
        if os.path.isfile(spec_plot.filename):
            self.config.logger.info("%s already there.  Skipping..." % (spec_plot.filename))
        else:
            self.config.logger.info("Correcting redshifts and making spec plot...")
            # We need to do a final run to apply the corrections, but this
            # seems inefficient...

            # Load in the catalog, apply the corrections, and make the plot.
            self.config.logger.info(self.config.catfile)
            cat = Catalog.from_fits_file(self.config.catfile)

            # Compute offsets for all marginal clusters to check for errors
            use, = np.where(cat.Lambda > self.config.calib_minlambda)
            cat = cat[use]

            zlambda_corr = ZlambdaCorrectionPar(parfile=self.config.zlambdafile,
                                                zlambda_pivot=self.config.zlambda_pivot)
            zlold = cat.z_lambda
            zleold = cat.z_lambda_e

            for cluster in cat:
                zlam, zlam_e = zlambda_corr.apply_correction(cluster.Lambda, cluster.z_lambda, cluster.z_lambda_e)
                cluster.z_lambda = zlam
                cluster.z_lambda_e = zlam_e

            # Check for negative values... this is a minimal check
            test, = np.where(cluster.z_lambda < 0.0)
            if (test.size > 0):
                raise RuntimeError("z_lambda correction calibration totally failed, and is yielding negative redshifts.  "
                                   "Likely there is a configuration error, see config.calib_zlambda_minlambda and "
                                   "config.calib_zlambda_nodesize.")

            # Only plot those above the specified threshold
            use, = np.where(cat.Lambda > self.config.calib_zlambda_minlambda)

            spec_plot.plot_values(cat.z_spec_init[use], cat.z_lambda[use], cat.z_lambda_e[use], title=self.config.d.outbase)


class RedmapperCalibrationIterationFinal(object):
    """
    Class to run the final iteration of the red-sequence calibration.

    This run does not start with the spectroscopic redshifts in the run to test
    the performance using zred as the cluster seed redshift.
    """

    def __init__(self, config):
        """
        Instantiate a RedmapperCalibrationIterationFinal

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        """
        self.config = config

    def run(self, iteration):
        """
        Run the final iteration of the red-sequence calibration.

        Files will be output with the iteration name "iter%{iteration}b".

        Parameters
        ----------
        iteration: `int`
           Iteration number.  Used for naming files.
        """

        # Generate the names
        self.config.d.outbase = '%s_iter%db' % (self.config.outbase, iteration)

        # Generate zreds for the specseeds
        iter_seedfile = self.config.redmapper_filename('specseeds')

        if os.path.isfile(iter_seedfile):
            self.config.logger.info('%s already there.  Skipping...'  % (iter_seedfile))
        else:
            self.config.logger.info("Creating final iteration seeds...")
            seedzredfile = self.config.redmapper_filename('specseeds_zreds')
            zredRuncat = ZredRunCatalog(self.config)
            zredRuncat.run(self.config.seedfile, seedzredfile)

            # Now combine seeds with zreds
            seeds = Catalog.from_fits_file(self.config.seedfile, ext=1)
            zreds = Catalog.from_fits_file(seedzredfile, ext=1)

            seeds.zred = zreds.zred
            seeds.zred_e = zreds.zred_e
            seeds.zred_chisq = zreds.chisq

            seeds.to_fits_file(iter_seedfile)

        # Do the full run with these seeds
        # and the previously calibrated zlambda correction (which is what's used)

        finalfile = self.config.redmapper_filename('final')

        if os.path.isfile(finalfile):
            self.config.logger.info('%s already there.  Skipping...' % (finalfile))
        else:
            self.config.logger.info("Doing final iteration run")
            redmapper_run = RedmapperRun(self.config)
            catfile = redmapper_run.run(seedfile=iter_seedfile, cleaninput=True)
            # check that catfile is the same as finalfile?
            if catfile != finalfile:
                raise RuntimeError("The output catfile %s should be the same as finalfile %s" % (catfile, finalfile))

        self.config.catfile = finalfile

        # And pretty plots
        spec_plot = SpecPlot(self.config)
        if os.path.isfile(spec_plot.filename):
            self.config.logger.info("%s already there.  Skipping..." % (spec_plot.filename))
        else:
            self.config.logger.info("Making final iteration spec plot...")
            # We do not need to do the corrections for the final run (which has them, I hope)

            cat = Catalog.from_fits_file(self.config.catfile)
            use, = np.where(cat.Lambda > self.config.calib_zlambda_minlambda)
            cat = cat[use]

            spec_plot.plot_values(cat.z_spec_init, cat.z_lambda, cat.z_lambda_e, title=self.config.d.outbase)

