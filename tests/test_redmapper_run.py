from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import copy
from numpy import random
import time
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import RedSequenceColorPar
from redmapper import RedmapperRun
from redmapper import Catalog

class RedmapperRunTestCase(unittest.TestCase):
    """
    Tests of redmapper.RedmapperRun, which runs all the redmapper steps on an
    input catalog.
    """

    def test_redmapper_run(self):
        """
        Run test of redmapper.RedmapperRun.
        """

        random.seed(seed=12345)

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        # First, test the splitting
        config.calib_run_nproc = 4

        redmapper_run = RedmapperRun(config)

        splits = redmapper_run._get_pixel_splits()

        self.assertEqual(splits[0], 64)
        testing.assert_array_equal(splits[1], np.array([2163, 2296, 2297, 2434]))

        # Now, this will just run on 1 but will test consolidation code
        config.calib_run_nproc = 2
        # Note you need these to be set to get same answer with nproc = 1 because
        # of mask rounding
        config.seedfile = os.path.join(file_path, 'test_dr8_specseeds.fit')
        config.zredfile = os.path.join(file_path, 'zreds_test', 'dr8_test_zreds_master_table.fit')

        redmapper_run = RedmapperRun(config)
        redmapper_run.run(specmode=True, consolidate_like=True, keepz=True, seedfile=config.seedfile)

        # Now let's check that we got the final file...
        self.assertTrue(os.path.isfile(os.path.join(config.outpath, '%s_final.fit' % (config.d.outbase))))
        self.assertTrue(os.path.isfile(os.path.join(config.outpath, '%s_final_members.fit' % (config.d.outbase))))
        self.assertTrue(os.path.isfile(os.path.join(config.outpath, '%s_like.fit' % (config.d.outbase))))

        cat = Catalog.from_fits_file(os.path.join(config.outpath, '%s_final.fit' % (config.d.outbase)))

        # Spot checks to look for regressions
        testing.assert_equal(cat.size, 23)
        self.assertGreater(cat.Lambda.min(), 3.0)
        testing.assert_array_almost_equal(cat.Lambda[0: 3], np.array([23.588404, 18.172888, 6.990992]))

        # And check that the members are all accounted for...
        mem = Catalog.from_fits_file(os.path.join(config.outpath, '%s_final_members.fit' % (config.d.outbase)))
        a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
        testing.assert_equal(a.size, mem.size)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
