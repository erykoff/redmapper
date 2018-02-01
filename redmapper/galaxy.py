import fitsio
import esutil
from esutil.htm import Matcher
#import smatch
import numpy as np
import itertools
from catalog import Catalog, Entry
import healpy as hp
import os


class Galaxy(Entry):
    """
    """

    @property
    def galcol(self):
        return self.mag[:-1] - self.mag[1:]


class GalaxyCatalog(Catalog):
    """
    Name:
        GalaxyCatalog
    Purpose:
        An object for holding galaxy catalogs, including all of
        the attributes that a galaxy would have.
    """

    entry_class = Galaxy

    def __init__(self, *arrays, **kwargs):
        super(GalaxyCatalog, self).__init__(*arrays)
        self._htm_matcher = None
        self.depth = 10 if 'depth' not in kwargs else kwargs['depth']
        #self._smatch_catalog = None
        #self._smatch_nside = 4096 if 'smatch_nside' not in kwargs else kwargs['smatch_nside']

    @classmethod
    def from_galfile(cls, filename, nside=None, hpix=None, border=0.0):
        """
        Name:
            from_galfile
        Purpose:
            do the actual reading in of the data fields in some galaxy
            catalog file
        Calling Squence:
            TODO
        Inputs:
            filename: a name of a galaxy catalog file
        Optional Inputs:
            nside: integer
                healpix nside of sub-pixel
            hpix: integer
                healpix pixel (ring order) of sub-pixel
        Outputs:
            A galaxy catalog object
        """
        # do we have appropriate keywords
        if hpix is not None and nside is None:
            raise ValueError("If hpix is specified, must also specify nside")
        if border < 0.0:
            raise ValueError("Border must be >= 0.0.")
        # ensure that nside is valid, and hpix is within range (if necessary)
        if nside is not None:
            if not hp.isnsideok(nside):
                raise ValueError("Nside not valid")
            if hpix is not None:
                if hpix < 0 or hpix >= hp.nside2npix(nside):
                    raise ValueError("hpix out of range.")

        # check that the file is there and the right format
        # this will raise an exception if it's not there.
        hdr = fitsio.read_header(filename, ext=1)
        pixelated, fitsformat = hdr.get("PIXELS", 0), hdr.get("FITS", 0)
        if not pixelated:
            return super(GalaxyCatalog, self).from_fits_file(filename)
        # this is to keep us from trying to use old IDL galfiles
        if not fitsformat:
            raise ValueError("Input galfile must describe fits files.")

        # now we can read in the galaxy table summary file...
        tab = fitsio.read(filename, ext=1)
        nside_tab = tab['NSIDE']
        if nside > nside_tab:
            raise ValueError("""Requested nside (%d) must not be larger than 
                                    table nside (%d).""" % (nside, nside_tab))

        # which files do we want to read?
        path = os.path.dirname(os.path.abspath(filename))
        if hpix is None:
            # all of them!
            indices = np.arange(tab[0]['FILENAMES'].size)
        else:
            # first we need all the pixels that are contained in the big pixel
            theta, phi = hp.pix2ang(nside_tab, tab[0]['HPIX'])
            ipring_big = hp.ang2pix(nside, theta, phi)
            indices, = np.where(ipring_big == hpix)
            if border > 0.0:
                # now we need to find the extra boundary...
                boundaries = hp.boundaries(nside, hpix, step=nside_tab/nside)
                inhpix = tab[0]['HPIX'][indices]
                for i in xrange(boundaries.shape[1]):
                    pixint = hp.query_disc(nside_tab, boundaries[:,i],
                                    border*np.pi/180., inclusive=True, fact=8)
                    inhpix = np.append(inhpix, pixint)
                inhpix = np.unique(inhpix)
                _, indices = esutil.numpy_util.match(inhpix, tab[0]['HPIX'])

        # create the catalog array to read into
        elt = fitsio.read('%s/%s' % (path, tab[0]['FILENAMES'][indices[0]]),ext=1, rows=0)
        cat = np.zeros(np.sum(tab[0]['NGALS'][indices]), dtype=elt.dtype)

        # read the files
        ctr = 0
        for index in indices:
            cat[ctr : ctr+tab[0]['NGALS'][index]] = fitsio.read('%s/%s' % (path, tab[0]['FILENAMES'][index]), ext=1)
            ctr += tab[0]['NGALS'][index]
        # In the IDL version this is trimmed to the precise boundary requested.
        # that's easy in simplepix.  Not sure how to do in healpix.
        return cls(cat)

    @property
    def galcol(self):
        galcol = self.mag[:,:-1] - self.mag[:,1:]
        return galcol

    def match_one(self, ra, dec, radius):
        """
        match an ra/dec to the galaxy catalog

        parameters
        ----------
        ra: input ra
        dec: input dec
        radius: float
           radius in degrees

        returns
        -------
        #(indices, dists)
        #indices: array of integers
        #    indices in galaxy catalog of matches
        #dists: array of floats
        #    match distance for each match (degrees)
        """

        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)

        _, indices, dists = self._htm_matcher.match(ra, dec, radius, maxmatch=0)

        return indices, dists

    def match_many(self, ras, decs, radius):
        """
        match many ras/decs to the galaxy catalog

        parameters
        ----------
        ras: input ras
        decs: input decs
        radius: float
           radius/radii in degrees

        returns
        -------
        (i0, i1, dists)
        i0: array of integers
            indices for input ra/dec
        i1: array of integers
            indices for galaxy catalog
        dists: array of floats
            match distance (degrees)
        """

        if self._htm_matcher is None:
            self._htm_matcher = Matcher(self.depth, self.ra, self.dec)

        return self._htm_matcher.match(ras, decs, radius, maxmatch=0)

