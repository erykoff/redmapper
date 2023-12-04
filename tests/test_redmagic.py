import matplotlib
matplotlib.use('Agg')

import unittest
import os
import shutil
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
from numpy import random
import glob

from redmapper import Configuration
from redmapper.redmagic import RedmagicParameterFitter, RedmagicCalibrator
from redmapper.redmagic import RunRedmagicTask
from redmapper import RedSequenceColorPar
from redmapper import GalaxyCatalog
from redmapper import Catalog
from redmapper import VolumeLimitMask

class RedmagicCalTestCase(unittest.TestCase):
    def test_redmagic_fitter(self):
        """
        Test the redmagic fitting functions individually
        """
        np.random.seed(12345)

        file_path = 'data_for_tests/redmagic_test'

        # Read in the red-sequence parametrization

        # Read in the input data for comparison (no afterburner)
        calstr = fitsio.read(os.path.join(file_path, 'rcal_str_preab.fit'), ext=1, lower=True)

        # Read in the input data for testing (no afterburner)
        calstr2 = fitsio.read(os.path.join(file_path, 'rcal_str2.fit'), ext=1, lower=True)

        # Make a zred structure for mstar...
        config = Configuration(os.path.join('data_for_tests', 'testconfig.yaml'))
        zredstr = RedSequenceColorPar(None, config=config)

        # Set up the fitter...
        #randomn = np.random.normal(size=calstr2['z'][0, :].size)
        # Old IDL code did not sample for the selection, I think this was wrong
        # randomn = np.zeros(calstr2['z'][0, :].size)
        zsamp = calstr2['z'][0, :]

        ab_use = np.random.choice(np.arange(calstr2['z'][0, :].size), size=3000, replace=False)

        # Force this to be some smooth function of redshift
        zcal_raw = calstr2['z'][0, :] + 0.2 * (calstr2['z'][0, :] - 0.3)
        zcal_e = calstr2['z_err'][0, :]

        # And add some excess noise...
        scale = 1.0
        zcal = zcal_raw + np.random.normal(loc=0.0, scale=zcal_e*scale, size=zcal_raw.size)

        rmfitter = RedmagicParameterFitter(calstr['nodes'][0, :], calstr['corrnodes'][0, :],
                                           calstr2['z'][0, :], calstr2['z_err'][0, :],
                                           calstr2['chisq'][0, :], calstr2['mstar'][0, :],
                                           zcal, zcal_e,
                                           calstr2['refmag'][0, :], zsamp,
                                           calstr2['zmax'][0, :],
                                           calstr['etamin'][0], calstr['n0'][0],
                                           calstr2['volume'][0, :], calstr2['zrange'][0, :],
                                           calstr2['zbinsize'][0],
                                           zredstr, maxchi=20.0,
                                           ab_use=ab_use)

        # These match the IDL values
        testing.assert_almost_equal(rmfitter(calstr['cmax'][0, :]), 1.9331937798956758)

        p0_cval = np.zeros(calstr['nodes'][0, :].size) + 2.0
        testing.assert_almost_equal(rmfitter(p0_cval), 317.4524284321642)

        cvals = rmfitter.fit(p0_cval)

        # This does not match the IDL output, because this is doing a lot
        # better job minimizing the function, at least in this test.
        # I hope this is just because of the size of the testbed, which is
        # really way too small for something like this.
        testing.assert_almost_equal(cvals, np.array([2.61657263, 2.20376531, 1.00663991]))

        # Now we have to check the fitting with the afterburner

        biasvals = np.zeros(rmfitter._corrnodes.size)
        eratiovals = np.ones(rmfitter._corrnodes.size)
        biasvals, eratiovals = rmfitter.fit_bias_eratio(cvals, biasvals, eratiovals)

        cvals = rmfitter.fit(cvals, biaspars=biasvals, eratiopars=eratiovals, afterburner=True)

        # We have only a rough test here; the variable chisq cut version of
        # redmagic is actually a bit fragile for other reasons and constant-chi2
        # is recommended.
        testing.assert_almost_equal(cvals, np.array([2.39569338, 3.07408774, 0.8872264]), 1)
        testing.assert_almost_equal(biasvals, np.array([0.04477243, 0.00182884, -0.03398897]), 4)
        testing.assert_almost_equal(eratiovals, np.array([0.64541869, 0.94068391, 0.89967353]), 2)

    def test_redmagic_calibrate(self):
        """
        Test the redmagic calibration code
        """

        np.random.seed(12345)

        file_path = 'data_for_tests'
        conf_filename = 'testconfig_redmagic.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        testgals = GalaxyCatalog.from_fits_file(os.path.join('data_for_tests', 'redmagic_test', 'redmagic_test_input_gals.fit'))

        testgals.add_fields([('mag', 'f4', 5), ('mag_err', 'f4', 5),
                             ('zred_samp', 'f4', config.zred_nsamp)])
        testgals.zred_samp[:, 0] = testgals.zred_uncorr

        redmagic_cal = RedmagicCalibrator(config)
        # We have to have do_run=False here because we don't have a real
        # galaxy training set with associated zreds!
        redmagic_cal.run(gals=testgals, do_run=False)

        # Read in the calibrated parameters

        self.assertTrue(os.path.isfile(config.redmagicfile))

        cal = fitsio.read(config.redmagicfile, ext=1)

        # Check that they are what we think they should be
        # (these checks are arbitrary, just to make sure nothing has changed)

        testing.assert_almost_equal(cal['cmax'][0, :], np.array([3.25383848, 2.62276221, 0.17340609]), 5)
        testing.assert_almost_equal(cal['bias'][0, :], np.array([-0.00961071, -0.0281055, 0.04684099]), 4)
        testing.assert_almost_equal(cal['eratio'][0, :], np.array([1.5, 0.78679151, 0.5]), 3)

        pngs = glob.glob(os.path.join(self.test_dir, '*.png'))
        self.assertEqual(len(pngs), 3)

        # This is a hack of the volume limit mask to change from the one used for
        # calibration to the one used for the run (which uses a different footprint
        # because of reasons)

        config_regular = Configuration(os.path.join(file_path, 'testconfig.yaml'))
        maskfile = config_regular.maskfile

        config = Configuration(redmagic_cal.runfile)

        # This little dance removes the vmaskfile and creates a new one with
        # the same name in the same location
        cal, hdr = fitsio.read(config.redmagicfile, ext=1, header=True)
        config.maskfile = maskfile
        try:
            vmaskfile = cal['vmaskfile'][0].decode().rstrip()
        except AttributeError:
            vmaskfile = cal['vmaskfile'][0].rstrip()
        os.remove(vmaskfile)
        mask = VolumeLimitMask(config, cal['etamin'], use_geometry=True)

        rng = random.RandomState(12345)

        # Now test the running, using the output file which has valid galaxies/zreds
        run_redmagic = RunRedmagicTask(redmagic_cal.runfile)
        run_redmagic.run(rng=rng)

        # check that we have a redmagic catalog
        rmcatfile = config.redmapper_filename('redmagic_%s' % ('highdens'), withversion=True)
        self.assertTrue(os.path.isfile(rmcatfile))

        # And a random catalog
        rmrandfile = config.redmapper_filename('redmagic_%s_randoms' % ('highdens'), withversion=True)
        self.assertTrue(os.path.isfile(rmrandfile))

        # And check that the plot is there

        pngs = glob.glob(os.path.join(self.test_dir, '*.png'))
        self.assertEqual(len(pngs), 4)

        # And that we have the desired number of redmagic and randoms
        red_cat = GalaxyCatalog.from_fits_file(rmcatfile)
        rand_cat = GalaxyCatalog.from_fits_file(rmrandfile)

        self.assertEqual(rand_cat.size, red_cat.size * 10)

        # And confirm that all the randoms are in the footprint
        zmax = mask.calc_zmax(rand_cat.ra, rand_cat.dec)
        self.assertTrue(np.all(rand_cat.z < zmax))

        # Now run in a different directory
        relocpath = os.path.join(self.test_dir, 'redmagic_relocation')
        os.makedirs(relocpath)
        reloc_file = os.path.join(relocpath,
                                  os.path.basename(config.redmagicfile))
        os.rename(config.redmagicfile, reloc_file)
        os.rename(vmaskfile, os.path.join(relocpath,
                                          os.path.basename(vmaskfile)))

        rerun_config = Configuration(redmagic_cal.runfile)
        rerun_config.redmagicfile = reloc_file
        rerun_configfile = os.path.join(self.test_dir, 'testconfig_redmagic_rerun.yml')
        rerun_config.output_yaml(rerun_configfile)
        rerun_redmagic = RunRedmagicTask(rerun_configfile)
        rerun_redmagic.run(clobber=True)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
