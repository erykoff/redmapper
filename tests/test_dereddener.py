import unittest
import numpy.testing as testing
import numpy as np
import os
import healsparse

from redmapper import Dereddener, GalaxyCatalog


class DereddenerTestCase(unittest.TestCase):
    """Test the dereddener functionality.
    """
    def test_dered(self):
        """Test running the dereddener."""
        mapfile = os.path.join('data_for_tests', 'test_sfd98_subregion.hsp')

        norm = 1.0
        constants = [1.5, 1.0]
        nmag = len(constants)
        ref_ind = 1

        dereddener = Dereddener(mapfile, norm, constants, nmag, ref_ind)

        # Check the map hasn't been loaded
        self.assertEqual(dereddener._reddening_map, None)

        # Read the map to check
        red_map = healsparse.HealSparseMap.read(mapfile)

        # Generate some positions in the region
        np.random.seed(12345)
        nrand = 1000
        ra, dec = healsparse.healSparseRandoms.make_uniform_randoms_fast(red_map, nrand)

        gal_array = np.zeros(nrand, dtype=[('ra', 'f8'),
                                           ('dec', 'f8'),
                                           ('refmag', 'f4'),
                                           ('mag', 'f4', 2)])
        gal_array['ra'] = ra
        gal_array['dec'] = dec

        # Set one to be a bad one
        gal_array['ra'][0] = 100.0
        gal_array['dec'][0] = 50.0

        # Do lookup here...
        a0 = norm*constants[0]*red_map.get_values_pos(ra, dec, lonlat=True)
        a1 = norm*constants[1]*red_map.get_values_pos(ra, dec, lonlat=True)
        aref = norm*constants[ref_ind]*red_map.get_values_pos(ra, dec, lonlat=True)

        a0[0] = np.median(a0[1: ])
        a1[0] = np.median(a1[1: ])
        aref[0] = np.median(aref[1: ])

        gal_cat = GalaxyCatalog(gal_array.copy())

        dereddener.deredden_array(gal_array)

        # Check the map has been loaded
        self.assertNotEqual(dereddener._reddening_map, None)

        testing.assert_array_almost_equal(gal_array['mag'][:, 0], -1.0*a0)
        testing.assert_array_almost_equal(gal_array['mag'][:, 1], -1.0*a1)
        testing.assert_array_almost_equal(gal_array['refmag'], -1.0*aref)

        dereddener.deredden_catalog(gal_cat)

        testing.assert_array_almost_equal(gal_cat.mag[:, 0], -1.0*a0)
        testing.assert_array_almost_equal(gal_cat.mag[:, 1], -1.0*a1)
        testing.assert_array_almost_equal(gal_cat.refmag, -1.0*aref)


if __name__=='__main__':
    unittest.main()
