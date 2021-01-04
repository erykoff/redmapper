import unittest
import numpy.testing as testing
import numpy as np
from numpy import random
import healpy as hp
import tempfile
import shutil
import fitsio
import os

from redmapper.pipeline import RedmapperConsolidateTask
from redmapper import Configuration, ClusterCatalog

class ConsolidateTestCase(unittest.TestCase):
    """
    Test the parallelized cluster catalog consolidation code in
    redmapper.pipeline.RedmapperConsolidateTask
    """
    def runTest(self):
        """
        Test redmapper.pipeline.RedmapperConsolidateTask
        """

        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.consolidate_lambda_cuts = np.array([5.0, 20.0])
        config.consolidate_vlim_lstars = np.array([])

        # Make 3 fake simple catalogs...
        random.seed(seed=12345)
        nside = 4

        test_arr = np.zeros(5, dtype=[('mem_match_id', 'i4'),
                                      ('ra', 'f8'),
                                      ('dec', 'f8'),
                                      ('lambda', 'f4'),
                                      ('lnlamlike', 'f4')])

        # Do cat0, pixel 0
        theta, phi = hp.pix2ang(nside, 0)
        test_arr['mem_match_id'] = np.arange(test_arr.size) + 1
        test_arr['lambda'][:] = 100.0
        test_arr['lnlamlike'] = random.random(size=test_arr.size) * 100
        test_arr['ra'][:] = np.degrees(phi)
        test_arr['dec'][:] = 90.0 - np.degrees(theta)
        cat0 = ClusterCatalog(test_arr)
        cat0.to_fits_file(config.redmapper_filename('cat_4_00000_final_catalog'))
        cat0.to_fits_file(config.redmapper_filename('cat_4_00000_final_catalog_members'))

        # Do cat1, pixel 1
        theta, phi = hp.pix2ang(nside, 1)
        test_arr['mem_match_id'] = np.arange(test_arr.size) + 1
        test_arr['lambda'][:] = 100.0
        test_arr['lnlamlike'] = random.random(size=test_arr.size) * 100
        test_arr['ra'][:] = np.degrees(phi)
        test_arr['dec'][:] = 90.0 - np.degrees(theta)
        cat1 = ClusterCatalog(test_arr)
        cat1.to_fits_file(config.redmapper_filename('cat_4_00001_final_catalog'))
        cat1.to_fits_file(config.redmapper_filename('cat_4_00001_final_catalog_members'))

        # Do cat2, pixel 2
        theta, phi = hp.pix2ang(nside, 2)
        test_arr['mem_match_id'] = np.arange(test_arr.size) + 1
        test_arr['lambda'][:] = 100.0
        test_arr['lnlamlike'] = random.random(size=test_arr.size) * 100
        test_arr['ra'][:] = np.degrees(phi)
        test_arr['dec'][:] = 90.0 - np.degrees(theta)
        cat2 = ClusterCatalog(test_arr)
        cat2.to_fits_file(config.redmapper_filename('cat_4_00002_final_catalog'))
        cat2.to_fits_file(config.redmapper_filename('cat_4_00002_final_catalog_members'))

        # need to write config out in test directory...
        config_file = config.redmapper_filename('testconfig', filetype='yaml')
        config.output_yaml(config_file)

        # Consolidate them together...
        consol = RedmapperConsolidateTask(config_file)
        consol.run(match_spec=False, do_plots=False)

        # Check that the ordering is correct, etc.
        catfile = config.redmapper_filename('redmapper_v%s_lgt20_catalog' % (config.version))
        memfile = config.redmapper_filename('redmapper_v%s_lgt20_catalog_members' % (config.version))
        self.assertTrue(os.path.isfile(catfile))
        self.assertTrue(os.path.isfile(memfile))

        cat = fitsio.read(catfile, ext=1)
        self.assertEqual(cat.size, cat0.size + cat1.size + cat2.size)

        # Sort by mem_match_id, these should be reverse sorted in lnlamlike
        st = np.argsort(cat['mem_match_id'])
        self.assertTrue(np.all(np.diff(cat['lnlamlike'][st[::-1]]) >= 0))

        # And check that the ra/dec/lnlamlike match...
        test0 = {}
        for cluster in cat0:
            test0[cluster.lnlamlike] = cluster.ra
        ctr = 0
        for i in range(cat.size):
            if cat['lnlamlike'][i] in test0:
                if cat['ra'][i] == test0[cat['lnlamlike'][i]]:
                    ctr += 1
        self.assertEqual(ctr, cat0.size)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
