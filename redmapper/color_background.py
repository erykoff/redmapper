"""Color-only galaxy background class for redmapper.

This file contains classes to describe the b(x) background terms using colors only.
"""
import fitsio
import numpy as np
import esutil
import os

from .catalog import Entry
from .utilities import interpol
from .depthmap import DepthMap
from .utilities import cic
from .galaxy import GalaxyCatalog

class ColorBackground(object):
    """
    Color-only galaxy background class.

    This class describes the binned, interpolateable background term b(x), where x describes the color(s) and reference magnitude of the galaxy.

    This contains all individual colors and color/color cross terms in a
    matrix.
    """

    def __init__(self, filename, usehdrarea=False):
        """
        Instantiate a color background

        Parameters
        ----------
        filename: `string`
           Color background filename
        usehdrarea: `bool`, optional
           Use area from the header instead of as configured.  Default is False.
        """

        refmagbinsize = 0.01
        colbinsize = 0.01
        area = 1.0

        fits = fitsio.FITS(filename)

        self.bkgs = {}

        started = False

        for ext in fits[1: ]:
            extname = ext.get_extname()

            parts = extname.split('_')
            iind = int(parts[0])
            jind = int(parts[1])

            key = iind * 100 + jind

            obkg = Entry.from_fits_ext(ext)

            # Create the refmag bins
            refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], refmagbinsize)
            nrefmagbins = refmagbins.size

            # Deal with area if necessary
            if not started:
                if usehdrarea:
                    if 'areas' in obkg.dtype.names:
                        areas = interpol(obkg.areas, obkg.refmagbins, refmagbins)
                    else:
                        hdr = ext.read_header()
                        areas = np.zeros(nrefmagbins) + hdr['AREA']
                else:
                    areas = np.zeros(nrefmagbins) + area

                started = True

            if iind == jind:
                # We are on a diagonal

                # First do the refmag
                ncolbins = obkg.colbins.size
                bc_new = np.zeros((nrefmagbins, ncolbins))
                for i in range(ncolbins):
                    bc_new[:, i] = np.clip(interpol(obkg.bc[:, i], obkg.refmagbins, refmagbins), 0.0, None)

                bc = bc_new.copy()

                # Now do the color
                colbins = np.arange(obkg.colrange[0], obkg.colrange[1], colbinsize)
                ncolbins = colbins.size

                bc_new = np.zeros((nrefmagbins, ncolbins))
                for j in range(nrefmagbins):
                    bc_new[j, :] = np.clip(interpol(bc[j, :], obkg.colbins, colbins), 0.0, None)

                n = np.sum(bc, axis=1, dtype=np.float64) * (colbinsize / obkg.colbinsize)

                sigma_g = bc_new.copy()
                for j in range(ncolbins):
                    sigma_g[:, j] = bc_new[:, j] / areas

                self.bkgs[key] = {'col1': iind,
                                  'col2': jind,
                                  'refmagindex': 1,
                                  'colbins': colbins,
                                  'colrange': obkg.colrange,
                                  'colbinsize': colbinsize,
                                  'refmagbins': refmagbins,
                                  'refmagrange': obkg.refmagrange,
                                  'refmagbinsize': refmagbinsize,
                                  'bc': bc_new,
                                  'n': n,
                                  'sigma_g': sigma_g}
            else:
                # We are on an off-diagonal

                # start with the refmag
                ncol1bins = obkg.col1bins.size
                ncol2bins = obkg.col2bins.size
                bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
                for i in range(ncol1bins):
                    for j in range(ncol2bins):
                        bc_new[:, j, i] = np.clip(interpol(obkg.bc[:, j, i], obkg.refmagbins, refmagbins), 0.0, None)

                bc = bc_new.copy()

                # color1
                col1bins = np.arange(obkg.col1range[0], obkg.col1range[1], colbinsize)
                ncol1bins = col1bins.size

                bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
                for j in range(ncol2bins):
                    for k in range(nrefmagbins):
                        bc_new[k, j, :] = np.clip(interpol(bc[k, j, :], obkg.col1bins, col1bins), 0.0, None)

                bc = bc_new.copy()

                col2bins = np.arange(obkg.col2range[0], obkg.col2range[1], colbinsize)
                ncol2bins = col2bins.size

                bc_new = np.zeros((nrefmagbins, ncol2bins, ncol1bins))
                for i in range(ncol1bins):
                    for k in range(nrefmagbins):
                        bc_new[k, :, i] = np.clip(interpol(bc[k, :, i], obkg.col2bins, col2bins), 0.0, None)

                temp = np.sum(bc_new, axis=1, dtype=np.float64) * colbinsize
                n = np.sum(temp, axis=1, dtype=np.float64) * colbinsize

                sigma_g = bc_new.copy()

                for j in range(ncol1bins):
                    for k in range(ncol2bins):
                        sigma_g[:, k, j] = bc_new[:, k, j] / areas

                self.bkgs[key] = {'col1': iind,
                                  'col2': jind,
                                  'refmag_index': 2,
                                  'col1bins': col1bins,
                                  'col1range': obkg.col1range,
                                  'col1binsize': colbinsize,
                                  'col2bins': col2bins,
                                  'col2range': obkg.col2range,
                                  'col2binsize': colbinsize,
                                  'refmagbins': refmagbins,
                                  'refmagrange': obkg.refmagrange,
                                  'refmagbinsize': refmagbinsize,
                                  'bc': bc_new,
                                  'n': n,
                                  'sigma_g': sigma_g}

    def sigma_g_diagonal(self, bkg_index, colors, refmags):
        """
        Compute sigma_g(color, refmag) for a diagonal (single-color) background.

        Parameters
        ----------
        bkg_index: `int`
           Color index along the diagonal
        colors: `np.array`
           Float array of colors
        refmags: `np.array`
           Float array of reference magnitudes

        Returns
        -------
        sigma_g: `np.array`
           Sigma_g(x) for input values
        """
        bkg = self.bkgs[bkg_index * 100 + bkg_index]

        colindex = np.searchsorted(bkg['colbins'], colors - bkg['colbinsize'])
        refmagindex = np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize'])
        # check for overruns
        badcol, = np.where((colindex < 0) | (colindex >= bkg['colbins'].size))
        colindex[badcol] = 0
        badrefmag, = np.where((refmagindex < 0) | (refmagindex >= bkg['refmagbins'].size))
        refmagindex[badrefmag] = 0

        sigma_g = bkg['sigma_g'][refmagindex, colindex]
        sigma_g[badcol] = np.inf
        sigma_g[badrefmag] = np.inf
        badcombo, = np.where(sigma_g == 0.0)
        sigma_g[badcombo] = np.inf

        return sigma_g

    def lookup_diagonal(self, bkg_index, colors, refmags, doRaise=True):
        """
        Look up the normalized background value b(color, refmag) for a diagonal
        (single-color) background.

        Parameters
        ----------
        bkg_index: `int`
           Color index along the diagonal
        colors: `np.array`
           Float array of colors
        refmags: `np.array`
           Float array of reference magnitudes
        doRaise: `bool`, optional
           Raise exception if color background does not exist.  Default is True.

        Returns
        -------
        b: `np.array`
           Background(x) for input values
        """
        try:
            bkg = self.bkgs[bkg_index * 100 + bkg_index]
        except KeyError:
            if doRaise:
                raise KeyError("Could not find a color background for %d" % (bkg_index))
            else:
                return np.zeros(colors.size)

        refmagindex = np.clip(np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize']), 0, bkg['refmagbins'].size - 1)
        col_index = np.clip(np.searchsorted(bkg['colbins'], colors - bkg['colbinsize']), 0, bkg['colbins'].size - 1)

        return bkg['bc'][refmagindex, col_index] / bkg['n'][refmagindex]

    def get_colrange(self, bkg_index):
        """
        Get the background color range for a given background index.

        Parameters
        ----------
        bkg_index: `int`
           Color index along the diagonal

        Returns
        -------
        range: `np.array`
           2-element float array with color min, max
        """
        bkg = self.bkgs[bkg_index * 100 + bkg_index]
        return bkg['colrange']

    def lookup_offdiag(self, bkg_index1, bkg_index2, colors1, colors2, refmags, doRaise=True):
        """
        Look up the normalized background value b(color1, color2, refmag) for
        an off-diagonal (two-color) background.

        Parameters
        ----------
        bkg_index1: `int`
           Color index for color 1
        bkg_index2: `int`
           Color index for color 2
        colors1: `np.array`
           Float array of colors (1)
        colors2: `np.array`
           Float array of colors (2)
        refmags: `np.array`
           Float array of reference mags
        doRaise: `bool`, optional
           Raise exception if color background does not exist.  Default is True.
        """
        key = bkg_index1 * 100 + bkg_index2
        if key not in self.bkgs:
            key = bkg_index2 * 100 + bkg_index1

        try:
            bkg = self.bkgs[key]
        except KeyError:
            if doRaise:
                raise KeyError("Could not find a color background for %d, %d" % (bkg_index1, bkg_index1))
            else:
                return np.zeros(colors1.size)

        refmagindex = np.clip(np.searchsorted(bkg['refmagbins'], refmags - bkg['refmagbinsize']), 0, bkg['refmagbins'].size - 1)
        col_index1 = np.clip(np.searchsorted(bkg['col1bins'], colors1 - bkg['col1binsize']), 0, bkg['col1bins'].size - 1)
        col_index2 = np.clip(np.searchsorted(bkg['col2bins'], colors2 - bkg['col2binsize']), 0, bkg['col2bins'].size - 1)

        return bkg['bc'][refmagindex, col_index2, col_index1] / bkg['n'][refmagindex]

class ColorBackgroundGenerator(object):
    """
    Class to generate color-only galaxy background.
    """

    def __init__(self, config, minrangecheck=1000):
        """
        Instantiate a color background generator.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration information
        minrangecheck: `int`, optional
           Minimum number of galaxies in a color bin to determine full range.
           Default is 1000
        """
        self.config = config
        self.minrangecheck = minrangecheck

    def run(self, clobber=False):
        """
        Generate the color galaxyt background.  The output filename is
        specified in self.config.bkgfile_color.

        Parameters
        ----------
        clobber: `bool`, optional
           Overwrite any existing self.config.bkgfile_color file.  Default is False.
        """

        # Check if it's already there...
        if not clobber and os.path.isfile(self.config.bkgfile_color):
            self.config.logger.info("Found %s and clobber is False" % (self.config.bkgfile_color))
            return

        # read in the galaxies
        gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                          nside=self.config.d.nside,
                                          hpix=self.config.d.hpix,
                                          border=self.config.border,
                                          dereddener=self.config.dereddener)

        # Generate ranges based on the data
        refmagbinsize = 0.05

        refmagrange = np.array([12.0, self.config.limmag_ref])

        nmag = self.config.nmag
        ncol = nmag - 1

        col = gals.galcol

        colrange_default = np.array([-2.0, 5.0])

        colranges = np.zeros((2, ncol))
        colbinsize = 0.02
        for i in range(ncol):
            use, = np.where((col[:, i] > colrange_default[0]) &
                            (col[:, i] < colrange_default[1]) &
                            (gals.refmag < (self.config.limmag_ref - 0.5)))

            h = esutil.stat.histogram(col[use, i], min=colrange_default[0],
                                      max=colrange_default[1], binsize=colbinsize)
            bins = np.arange(h.size) * colbinsize + colrange_default[0]

            good, = np.where(h > self.minrangecheck)

            colranges[0, i] = np.min(bins[good])
            colranges[1, i] = np.max(bins[good]) + colbinsize

        nrefmag = np.ceil((refmagrange[1] - refmagrange[0]) / refmagbinsize).astype(np.int32)
        refmagbins = np.arange(nrefmag) * refmagbinsize + refmagrange[0]

        if self.config.depthfile is not None:
            depthstr = DepthMap(self.config)
            areas = depthstr.calc_areas(refmagbins)
        else:
            areas = np.zeros(refmagbins.size) + self.config.area

        for i in range(ncol):
            for j in range(i, ncol):
                if (i == j):
                    # diagonal
                    ncoldiag = np.ceil((colranges[1, i] - colranges[0, i]) / colbinsize).astype(np.int32)
                    coldiagbins = np.arange(ncoldiag) * colbinsize + colranges[0, i]
                    binsizes = refmagbinsize * colbinsize

                    bad = ((col[:, i] < colranges[0, i]) |
                           (col[:, i] >= colranges[1, i]) |
                           (gals.refmag < refmagrange[0]) |
                           (gals.refmag >= refmagrange[1]))

                    colpos = (col[~bad, i] - colranges[0, i]) * ncoldiag / (colranges[1, i] - colranges[0, i])
                    refmagpos = (gals[~bad].refmag - refmagrange[0]) * nrefmag / (refmagrange[1] - refmagrange[0])

                    value = np.ones(np.sum(~bad))
                    self.config.logger.info("Running CIC on %d, %d" % (i, j))
                    field = cic(value, colpos, ncoldiag, refmagpos, nrefmag, isolated=True)
                    # need to fix cic...
                    bad = ~np.isfinite(field)
                    field[bad] = 0.0

                    bc = field.astype(np.float32) / binsizes
                    n = np.sum(bc, axis=1, dtype=np.float64) * colbinsize

                    outstr = np.zeros(1, dtype=[('COL1', 'i2'),
                                                ('COL2', 'i2'),
                                                ('COL_INDEX', 'i2'),
                                                ('REFMAG_INDEX', 'i2'),
                                                ('COLBINS', 'f4', coldiagbins.size),
                                                ('COLRANGE', 'f4', 2),
                                                ('COLBINSIZE', 'f4'),
                                                ('REFMAGBINS', 'f4', refmagbins.size),
                                                ('REFMAGRANGE', 'f4', 2),
                                                ('REFMAGBINSIZE', 'f4'),
                                                ('AREAS', 'f4', areas.size),
                                                ('BC', 'f4', bc.shape),
                                                ('N', 'f4', n.size)])

                    outstr['COL1'] = i
                    outstr['COL2'] = j
                    outstr['COL_INDEX'] = 0
                    outstr['REFMAG_INDEX'] = 1
                    outstr['COLBINS'][:] = coldiagbins
                    outstr['COLRANGE'][:] = colranges[:, i]
                    outstr['COLBINSIZE'] = colbinsize
                    outstr['REFMAGBINS'][:] = refmagbins
                    outstr['REFMAGRANGE'][:] = refmagrange
                    outstr['REFMAGBINSIZE'] = refmagbinsize
                    outstr['AREAS'][:] = areas
                    outstr['BC'][:, :] = bc
                    outstr['N'][:] = n

                else:
                    # off-diagonal

                    ncol1 = np.ceil((colranges[1, i] - colranges[0, i]) / colbinsize).astype(np.int32)
                    col1bins = np.arange(ncol1) * colbinsize + colranges[0, i]
                    ncol2 = np.ceil((colranges[1, j] - colranges[0, j]) / colbinsize).astype(np.int32)
                    col2bins = np.arange(ncol2) * colbinsize + colranges[0, j]

                    binsizes = refmagbinsize * colbinsize * colbinsize

                    bad = ((col[:, i] < colranges[0, i]) |
                           (col[:, i] >= colranges[1, i]) |
                           (col[:, j] < colranges[0, j]) |
                           (col[:, j] >= colranges[1, j]) |
                           (gals.refmag < refmagrange[0]) |
                           (gals.refmag >= refmagrange[1]))

                    col1pos = (col[~bad, i] - colranges[0, i]) * ncol1 / (colranges[1, i] - colranges[0, i])
                    col2pos = (col[~bad, j] - colranges[0, j]) * ncol2 / (colranges[1, j] - colranges[0, j])
                    refmagpos = (gals[~bad].refmag - refmagrange[0]) * nrefmag / (refmagrange[1] - refmagrange[0])

                    value = np.ones(np.sum(~bad))
                    self.config.logger.info("Running CIC on %d, %d" % (i, j))
                    field = cic(value, col1pos, ncol1, col2pos, ncol2, refmagpos, nrefmag, isolated=True)
                    bad = ~np.isfinite(field)
                    field[bad] = 0.0

                    bc = field.astype(np.float32) / binsizes
                    temp = np.sum(bc, axis=2, dtype=np.float64) * colbinsize
                    n = np.sum(temp, axis=1, dtype=np.float64) * colbinsize

                    outstr = np.zeros(1, dtype=[('COL1', 'i2'),
                                                ('COL2', 'i2'),
                                                ('COL1_INDEX', 'i2'),
                                                ('COL2_INDEX', 'i2'),
                                                ('REFMAG_INDEX', 'i2'),
                                                ('COL1BINS', 'f4', col1bins.size),
                                                ('COL1RANGE', 'f4', 2),
                                                ('COL1BINSIZE', 'f4'),
                                                ('COL2BINS', 'f4', col2bins.size),
                                                ('COL2RANGE', 'f4', 2),
                                                ('COL2BINSIZE', 'f4'),
                                                ('REFMAGBINS', 'f4', refmagbins.size),
                                                ('REFMAGRANGE', 'f4', 2),
                                                ('REFMAGBINSIZE', 'f4'),
                                                ('AREAS', 'f4', areas.size),
                                                ('BC', 'f4', bc.shape),
                                                ('N', 'f4', n.size)])

                    outstr['COL1'] = i
                    outstr['COL2'] = j
                    outstr['COL1_INDEX'] = 0
                    outstr['COL2_INDEX'] = 1
                    outstr['REFMAG_INDEX'] = 2
                    outstr['COL1BINS'][:] = col1bins
                    outstr['COL1RANGE'][:] = colranges[:, i]
                    outstr['COL1BINSIZE'] = colbinsize
                    outstr['COL2BINS'][:] = col2bins
                    outstr['COL2RANGE'][:] = colranges[:, j]
                    outstr['COL2BINSIZE'] = colbinsize
                    outstr['REFMAGBINS'][:] = refmagbins
                    outstr['REFMAGRANGE'][:] = refmagrange
                    outstr['REFMAGBINSIZE'] = refmagbinsize
                    outstr['AREAS'][:] = areas
                    outstr['BC'][:, :] = bc
                    outstr['N'][:] = n


                extname = "%02d_%02d_REF" % (i, j)
                hdr = fitsio.FITSHDR()
                hdr['AREA'] = self.config.area
                if i == 0 and j == 0:
                    startClobber = True
                else:
                    startClobber = False

                fitsio.write(self.config.bkgfile_color, outstr, extname=extname, header=hdr, clobber=startClobber)



