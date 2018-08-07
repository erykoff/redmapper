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
    """

    def test_redmapper_run(self):

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = test_dir

        # First, test the splitting
        config.calib_run_nproc = 4

        redmapper_run = RedmapperRun(config)

        splits = redmapper_run._get_pixel_splits()

        self.assertFalse(splits[0])
        self.assertEqual(splits[1], 64)
        testing.assert_array_equal(splits[2], np.array([2163, 2296, 2297, 2434]))


        # Now, this will just run on 1 but will test consolidation code
        config.calib_run_nproc = 2
        config.seedfile = os.path.join(file_path, 'test_dr8_specseeds.fit')
        config.zredfile = os.path.join(file_path, 'zreds_test', 'dr8_test_zreds_master_table.fit')

        redmapper_run = RedmapperRun(config)
        redmapper_run.run(specmode=True, consolidate_like=True)

        # Now let's check that we got the final file...
        self.assertTrue(os.path.isfile(os.path.join(config.outpath, '%s_final.fit' % (config.d.outbase))))
        self.assertTrue(os.path.isfile(os.path.join(config.outpath, '%s_final_members.fit' % (config.d.outbase))))
        self.assertTrue(os.path.isfile(os.path.join(config.outpath, '%s_like.fit' % (config.d.outbase))))

        cat = Catalog.from_fits_file(os.path.join(config.outpath, '%s_final.fit' % (config.d.outbase)))

        # Spot checks to look for regressions
        testing.assert_equal(cat.size, 27)
        self.assertGreater(cat.Lambda.min(), 3.0)
        testing.assert_array_almost_equal(cat.Lambda[0: 3], np.array([22.41377068, 17.94406319, 7.73848534]))

        # And check that the members are all accounted for...
        mem = Catalog.from_fits_file(os.path.join(config.outpath, '%s_final_members.fit' % (config.d.outbase)))
        a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
        testing.assert_equal(a.size, mem.size)

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, True)

if __name__=='__main__':
    unittest.main()