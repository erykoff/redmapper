"""Class for applying dereddening.
"""
import healsparse
import numpy as np


class Dereddener(object):
    """A class for holding reddening maps and applying dereddening.

    Parameters
    ----------

    """
    def __init__(self, mapfile, norm, constants, nmag, ref_ind):
        self.mapfile = mapfile
        self.norm = norm
        self.constants = constants
        self.nmag = nmag
        self.ref_ind = ref_ind

        self._reddening_map = None

    def deredden_array(self, galaxy_array):
        """Dereddening a galaxy array in-place.

        Parameters
        ----------
        galaxy_array : `numpy.ndarray`
            Galaxy array to apply dereddening.
        """
        if self._reddening_map is None:
            # Lazy-load the map only when needed.
            self._reddening_map = healsparse.HealSparseMap.read(self.mapfile)

        vals = self._reddening_map.get_values_pos(galaxy_array['ra'], galaxy_array['dec'], lonlat=True)

        bad = (vals < -100.0)
        if np.any(bad):
            vals[bad] = np.median(vals[~bad])

        galaxy_array['refmag'] -= self.norm*self.constants[self.ref_ind]*vals

        for i in range(self.nmag):
            galaxy_array['mag'][:, i] -= self.norm*self.constants[i]*vals

    def deredden_catalog(self, galaxies):
        """Deredden a galaxy catalog in-place.

        Parameters
        ----------
        galaxies : `redmapper.GalaxyCatalog`
            Galaxy catalog to apply dereddening.
        """
        self.deredden_array(galaxies._ndarray)

        # vals = self.reddening_map.get_values_pos(galaxies.ra, galaxies.dec, lonlat=True)

        # bad = (vals < -100.0)
        # if np.any(bad):
        #     vals[bad] = np.median(vals[~bad])

        # galaxies.refmag -= self.norm*self.constants[self.config.ref_ind]*vals

        # for i in range(self.config.nmag):
        #     galaxies.mag[:, i] -= self.norm*self.constants[i]*vals
