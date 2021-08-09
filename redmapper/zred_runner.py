"""Classes to run a galaxy catalog through zred galaxy redshift computation,
using multiprocessing.
"""
import numpy as np
import copy
import fitsio
import re
import os
import time

import multiprocessing

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

from .utilities import _pickle_method

from .zred_color import ZredColor
from .galaxy import GalaxyCatalog, get_subpixel_indices, zred_extra_dtype
from .catalog import Catalog, Entry
from .redsequence import RedSequenceColorPar

class ZredRunCatalog(object):
    """
    Class to run a galaxy catalog file to compute zreds, using multiprocessing.
    """

    def __init__(self, config):
        """
        Instantiate a ZredRunCatalog object.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        """

        self.config = config.copy()
        self.config.cosmo = None

    def run(self, galaxyfile, outfile, clobber=False, nperproc=None, maxperproc=500000):
        """
        Run a galaxy file to compute zreds and output zreds to output file.

        Parameters
        ----------
        galaxyfile: `str`
           Galaxy input file
        outfile: `str`
           Output zred file
        clobber: `bool`, optional
           Clobber existing outfile?  Default is False.
        nperproc: `int`, optional
           Number of galaxies to run per processor.
           Default is None, which divides the catalog evenly with
           a maximum of maxperproc.
        maxperproc: `int`, optional
           Maximum number to run per processor, when doing automatic
           division.   Default is 500000.
        """

        self.galaxyfile = galaxyfile
        self.outfile = outfile

        hdr = fitsio.read_header(self.galaxyfile, ext=1)
        ngal = hdr['NAXIS2']

        zredstr = RedSequenceColorPar(self.config.parfile)
        self.zredc = ZredColor(zredstr)

        zreds = Catalog(np.zeros(ngal, dtype=zred_extra_dtype(self.config.zred_nsamp)))

        if nperproc is None:
            nperproc = int(float(ngal) / (self.config.calib_nproc - 0.1))
            nperproc = np.clip(nperproc, None, maxperproc)

        inds = np.arange(0, ngal, nperproc)
        worker_list = [(ind, np.clip(ind + nperproc, None, ngal)) for ind in inds]

        mp_ctx = multiprocessing.get_context("fork")
        pool = mp_ctx.Pool(processes=self.config.calib_nproc)
        retvals = pool.map(self._worker, worker_list, chunksize=1)
        pool.close()
        pool.join()

        for ind_range, zred, zred_e, zred2, zred2_e, zred_uncorr, zred_uncorr_e, zred_samp, lkhd, chisq in retvals:
            zreds.zred[ind_range[0]: ind_range[1]] = zred
            zreds.zred_e[ind_range[0]: ind_range[1]] = zred_e
            zreds.zred2[ind_range[0]: ind_range[1]] = zred2
            zreds.zred2_e[ind_range[0]: ind_range[1]] = zred2_e
            zreds.zred_uncorr[ind_range[0]: ind_range[1]] = zred_uncorr
            zreds.zred_uncorr_e[ind_range[0]: ind_range[1]] = zred_uncorr_e
            zreds.zred_samp[ind_range[0]: ind_range[1], :] = zred_samp
            zreds.lkhd[ind_range[0]: ind_range[1]] = lkhd
            zreds.chisq[ind_range[0]: ind_range[1]] = chisq

        zreds.to_fits_file(outfile, clobber=clobber)

    def _worker(self, ind_range):
        """
        Do the run on a specific list of galaxies

        Parameters
        ----------
        ind_range: `list`
           2-element list with first and last index to compute zred.

        Returns
        -------
        ind_range: `list`
           Index range that was input
        zred: `np.array`
           Float array of zred for the galaxies in ind_range
        zred_e: `np.array`
           Float array of zred errors for the galaxies in ind_range
        zred2: `np.array`
           Float array of zred2 for the galaxies in ind_range
        zred2_e: `np.array`
           Float array of zred2 errors for the galaxies in ind_range
        zred_uncorr: `np.array`
           Float array of uncorrected zred for the galaxies in ind_range
        zred_uncorr_e: `np.array`
           Float array of uncorrected zred errors for the galaxies
           in ind_range
        zred_samp: `np.array`
           Float array of sampled zred values for the galaxes in ind_range
        lkhd: `np.array`
           Float array of likelihoods for the galaxies in ind_range
        chisq: `np.array`
           Float array of chi-squared for the galaxies in ind_range
        """

        # Need a GalaxyCatalog from a fits...
        in_cat = fitsio.read(self.galaxyfile, ext=1, upper=True,
                             rows=np.arange(ind_range[0], ind_range[1]))
        if self.config.dereddener is not None:
            self.config.dereddener.deredden_array(in_cat)
        galaxies = GalaxyCatalog(in_cat)
        galaxies.add_zred_fields(self.config.zred_nsamp)

        self.zredc.compute_zreds(galaxies)

        return (ind_range,
                galaxies.zred, galaxies.zred_e,
                galaxies.zred2, galaxies.zred2_e,
                galaxies.zred_uncorr, galaxies.zred_uncorr_e,
                galaxies.zred_samp,
                galaxies.lkhd, galaxies.chisq)


class ZredRunPixels(object):
    """
    Class to run a pixelized galaxy catalog to compute zreds, using
    multiprocessing.
    """

    def __init__(self, config):
        """
        Instantiate a ZredRunPixels object.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        """
        self.config = config.copy()
        self.config.cosmo = None

    def run(self, single_process=False, no_zred_table=False, verbose=False):
        """
        Run all the galaxies in a pixelized self.config.galfile to compute
        zreds and save zreds.

        Parameters
        ----------
        single_process: `bool`, optional
           Run as a single process only.  Useful for testing.  Default is False.
        no_zred_table: `bool`, optional
           Do not output a final zred table, instead return numbers.
           Default is False.
        verbose: `bool`, optional
           Be verbose with output.  Default is False.

        Returns
        -------
        retvals: `list`
           Present only if no_zred_table is True.
           List of (index, outfile) tuples describing output files.
        """

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with a pixelized galfile.")

        self.verbose = verbose
        self.single_process = single_process

        self.zredpath = os.path.dirname(self.config.zredfile)
        self.galpath = os.path.dirname(self.config.galfile)

        test = re.search('^(.*)_zreds_master_table.fit',
                         os.path.basename(self.config.zredfile))
        if test is None:
            raise ValueError("zredfile filename not in proper format (must end with _zreds_master_table.fit)")

        self.outbase = test.groups()[0]

        # Make the output directory if necessary
        if not os.path.exists(self.zredpath):
            os.makedirs(self.zredpath)

        zredstr = RedSequenceColorPar(self.config.parfile)
        self.zredc = ZredColor(zredstr)

        self.galtable = Entry.from_fits_file(self.config.galfile)
        indices = list(get_subpixel_indices(self.galtable,
                                            hpix=self.config.d.hpix, border=self.config.border, nside=self.config.d.nside))

        starttime = time.time()

        if not self.single_process:
            mp_ctx = multiprocessing.get_context("fork")
            pool = mp_ctx.Pool(processes=self.config.calib_nproc)
            retvals = pool.map(self._worker, indices, chunksize=1)
            pool.close()
            pool.join()
        else:
            self.ctr = 0
            self.total_galaxies = np.sum(self.galtable.ngals[indices])
            if (self.verbose):
                self.config.logger.info("Computing zred for %d galaxies in %d pixels." % (self.total_galaxies, len(indices)))
            mp_ctx = multiprocessing.get_context("fork")
            pool = mp_ctx.Pool(processes=1)
            retvals = pool.map(self._worker, indices, chunksize=1)
            pool.close()
            pool.join()

        self.config.logger.info("Done computing zreds in %.2f seconds" % (time.time() - starttime))

        if no_zred_table:
            return retvals

        self.make_zred_table(retvals)

    def _worker(self, index):
        """
        Do the run on a specific pixel index from the galaxy table.

        Parameters
        ----------
        index: `int`
           Pixel index in self.galtable from self.config.galfile

        Returns
        -------
        index: `int`
           Pixel index that was run
        outfile: `str`
           zredfile that was saved
        """
        # Read in just one single pixel
        galaxies = GalaxyCatalog.from_galfile(self.config.galfile,
                                              nside=self.galtable.nside,
                                              hpix=[self.galtable.hpix[index]],
                                              border=0.0,
                                              dereddener=self.config.dereddener)
        galaxies.add_zred_fields(self.config.zred_nsamp)

        if self.single_process:
            ctr = self.ctr
            ngal = self.total_galaxies
        else:
            ctr = 0
            ngal = galaxies.size

        self.zredc.compute_zreds(galaxies)
        ctr += galaxies.size

        if self.single_process:
            self.ctr = ctr

        # And write out the pixel file ... but just the zreds
        zreds = np.zeros(galaxies.size, dtype=zred_extra_dtype(self.config.zred_nsamp))
        for dt in zred_extra_dtype(self.config.zred_nsamp):
            zreds[dt[0]][:] = galaxies._ndarray[dt[0].lower()][:]

        outfile_nopath = '%s_zreds_%07d.fit' % (self.outbase, self.galtable.hpix[index])
        outfile = os.path.join(self.zredpath, outfile_nopath)

        fitsio.write(outfile, zreds, clobber=True)

        return (index, outfile)

    def make_zred_table(self, indices_and_filenames):
        """
        Make a zred table from a list of indices and filenames

        Saves to self.config.zredfile.

        Parameters
        ----------
        indices_and_filenames: `list`
           List of (index, outfile) tuples describing zred files.
        """

        # figure out longest filename
        maxlen = 0
        for index, filename in indices_and_filenames:
            if len(os.path.basename(filename)) > maxlen:
                maxlen = len(os.path.basename(filename))


        gal_dtype = self.galtable.dtype

        zred_dtype = []
        for dt in gal_dtype.descr:
            if dt[0].lower() == 'filenames':
                dt = ('filenames', 'S%d' % (maxlen + 1), dt[2])
            zred_dtype.append(dt)

        zredtable = Entry(np.zeros(1,dtype=zred_dtype))
        for name in gal_dtype.names:
            zredtable._ndarray[name] = self.galtable._ndarray[name]

        zredtable.filenames = ''
        zredtable.ngals = 0

        for index, filename in indices_and_filenames:
            # Make sure file exists
            if not os.path.isfile(filename):
                raise ValueError("Could not find zredfile: %s" % (filename))
            # check size of file
            hdr = fitsio.read_header(filename, ext=1)
            if hdr['NAXIS2'] != self.galtable.ngals[index]:
                raise ValueError("Length mismatch for zredfile: %s" % (filename))

            zredtable.filenames[index] = os.path.basename(filename)
            zredtable.ngals[index] = self.galtable.ngals[index]

        hdr = fitsio.FITSHDR()
        hdr['PIXELS'] = 1

        zredtable.to_fits_file(self.config.zredfile, header=hdr, clobber=True)

