import unittest
import numpy.testing as testing
import numpy as np
import tempfile
import shutil
import fitsio
import os

from redmapper import DepthMap
from redmapper import Configuration
from redmapper import VolumeLimitMask

class VolumeLimitMaskTestCase(unittest.TestCase):
    """
    Tests of redmapper.VolumeLimitMask volume-limit mask code.
    """
    def runTest(self):
        """
        Run tests of redmapper.VolumeLimitMask
        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        vlim = VolumeLimitMask(config, config.vlim_lstar)

        # And test the values...

        ras = np.array([140.0, 141.0, 150.0])
        decs = np.array([65.25, 65.6, 30.0])

        zmax = vlim.calc_zmax(ras, decs, get_fracgood=False)
        testing.assert_almost_equal(zmax, [0.338, 0.341, 0.0])

        # And compute a geometry mask
        vlim_geom = VolumeLimitMask(config, config.vlim_lstar + 1.0, use_geometry=True)

        zmax_geom = vlim_geom.calc_zmax(ras, decs, get_fracgood=False)
        testing.assert_almost_equal(zmax_geom, [config.zrange[1], config.zrange[1], 0.0])

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
