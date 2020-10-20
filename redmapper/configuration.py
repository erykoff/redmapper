"""Configuration class for redmapper.

This file contains the generic redmapper configuration class and associated classes.
"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import yaml
import fitsio
import copy
from esutil.cosmology import Cosmo
import numpy as np
import re
import os

from .cluster import cluster_dtype_base, member_dtype_base
from .utilities import Logger
from ._version import __version__

class ConfigField(object):
    """
    A class that describes a field that can be of various types, can specify a
    default, and can be validated.
    """

    def __init__(self, value=None, default=None, isArray=False, required=False, array_length=None, isList=False):
        """
        Instantiate a ConfigField

        Parameters
        ----------
        value: any type, optional
           The value to set to the config field.  Default is None.
        default: any type, optional
           The default value for the field.  Default is None.
        isArray: `bool`, optional
           Is the field an array type?  Default is False.
        required: `bool`, optional
           Is the field required to be set?  Default is False.
        array_length: `int`, optional
           Required array length for validation.  Default is None.
        isList: `bool`, optional
           Is the field a list type?  Default is False.
        """
        self._value = value
        self._isArray = isArray
        self._isList = isList
        self._required = required
        self._array_length = array_length

        self._default = default
        if isArray:
            if default is not None:
                self._default = np.atleast_1d(default)
            if self._value is not None:
                self._value = np.atleast_1d(self._value)
        if isList:
            if default is not None:
                self._default = list(default)
            if self._value is not None:
                self._value = list(self._value)

        if self._value is None:
            self._value = self._default

    def __get__(self, obj, type=None):
        return self._value

    def __set__(self, obj, value):
        if self._isArray:
            self._value = np.atleast_1d(value)
        elif self._isList:
            self._value = list(value)
        else:
            try:
                if len(value) > 1:
                    raise ValueError("ConfigField cannot be length > 1")
            except:
                pass
            self._value = value

    def reset(self):
        """Reset the value to the default."""
        self._value = self._default

    def set_length(self, length):
        """Set the desired array length.

        Parameters
        ----------
        length: `int`
           Desired array length to pass validation.
        """
        self._array_length = length

    def validate(self, name):
        """Validate the field.

        This needs the name because of python object weirdness.

        Parameters
        ----------
        name: `str`
           Name of the field that is being validated.

        Raises
        ------
        ValueError: if field does not validate.
        """
        if self._required:
            if self._value is None:
                raise ValueError("Required ConfigField %s is not set" % (name))

        if self._value is not None and (self._isArray or self._isList):
            if self._array_length is not None:
                if len(self._value) != self._array_length:
                    raise ValueError("ConfigField %s has the wrong length (%d != %d)" %
                                     (name, len(self._value), self._array_length))

        return True

def read_yaml(filename):
    """
    Read a yaml file into a dictionary.

    Parameters
    ----------
    filename: `str`
       File to read

    Returns
    -------
    outdict: `dict`
       Dictionary from yaml file

    """

    # The output dictionary
    outdict = {}

    # Open the yaml file and find key/value pairs
    with open(filename) as f: yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
    for tag in outdict:
        if outdict[tag] is None:
            raise ValueError('A value for the required tag \"'
                                + tag + '\" must be specified.')

    # Add the pairs to the dictionary
    for tag in yaml_data: outdict[tag] = yaml_data[tag]

    return outdict

class DuplicatableConfig(object):
    """
    Class to hold instances of variables that need to be duplicated for parallelism.
    """

    def __init__(self, config):
        """
        Instantiate a DuplicatableConfig.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration struct to copy values from

        """
        self.outbase = config.outbase
        self.hpix = config.hpix
        self.nside = config.nside


class Configuration(object):
    """
    Configuration class for redmapper.

    This class holds all the relevant configuration information, and had
    convenient methods for validating parameters as well as generating
    filenames, etc.

    """
    version = ConfigField(default=__version__, required=True)

    galfile = ConfigField(required=True)
    zredfile = ConfigField()
    halofile = ConfigField()
    randfile = ConfigField()
    catfile = ConfigField()
    specfile = ConfigField()
    specfile_train = ConfigField()
    outbase = ConfigField(required=True)
    parfile = ConfigField()
    bkgfile = ConfigField()
    bkgfile_color = ConfigField()
    zlambdafile = ConfigField()
    maskfile = ConfigField()
    depthfile = ConfigField()
    wcenfile = ConfigField()
    redgalfile = ConfigField()
    redgalmodelfile = ConfigField()
    seedfile = ConfigField()
    zmemfile = ConfigField()
    redmagicfile = ConfigField()

    calib_nproc = ConfigField(default=1, required=True)
    calib_run_nproc = ConfigField(default=1, required=True)
    calib_run_min_nside = ConfigField(default=1, required=True)

    outpath = ConfigField(default='./', required=True)
    plotpath = ConfigField(default='', required=True)

    border = ConfigField(default=0.0, required=True)
    hpix = ConfigField(default=[], required=True, isArray=True)
    nside = ConfigField(default=0, required=True)
    galfile_pixelized = ConfigField(required=True)

    nmag = ConfigField(required=True)
    area = ConfigField(required=True)
    limmag_catalog = ConfigField(required=True)
    limmag_ref = ConfigField(required=True)
    refmag = ConfigField(required=True)
    ref_ind = ConfigField(required=True)
    zeropoint = ConfigField(required=True)
    survey_mode = ConfigField(required=True)
    b = ConfigField(isArray=True)
    galfile_nside = ConfigField(required=True)
    bands = ConfigField(required=True)
    has_truth = ConfigField(default=False)

    area_finebin = ConfigField(default=0.001, required=True)
    area_coarsebin = ConfigField(default=0.005, required=True)

    zrange = ConfigField(isArray=True, array_length=2, required=True)
    lval_reference = ConfigField(default=0.2, required=True)

    maskgalfile = ConfigField(default='maskgals.fit', required=True)
    mask_mode = ConfigField(default=0, required=True)
    max_maskfrac = ConfigField(default=0.2, required=True)
    covmask_nside_default = ConfigField(default=32, required=True)

    maskgal_ngals = ConfigField(default=6000, required=True)
    maskgal_nsamples = ConfigField(default=100, required=True)
    maskgal_rad_stepsize = ConfigField(default=0.1, required=True)
    maskgal_dmag_extra = ConfigField(default=0.3, required=True)
    maskgal_zred_err = ConfigField(default=0.02, required=True)

    dldr_gamma = ConfigField(default=0.6, required=True)
    rsig = ConfigField(default=0.05, required=True)
    chisq_max = ConfigField(default=20.0, required=True)
    npzbins = ConfigField(default=21, required=True)

    zred_nsamp = ConfigField(default=4, required=True)

    mstar_survey = ConfigField(default='sdss')
    mstar_band = ConfigField(default='i03')

    calib_niter = ConfigField(default=3)
    calib_zrange_cushion = ConfigField(default=0.05)

    calib_use_pcol = ConfigField(default=True)
    calib_smooth = ConfigField(default=0.003)
    calib_minlambda = ConfigField(default=5.0)
    calib_redgal_template = ConfigField()
    calib_pivotmag_nodesize = ConfigField(default=0.1)
    calib_color_nodesizes = ConfigField(isArray=True, default=np.array([0.05]))
    calib_slope_nodesizes = ConfigField(isArray=True, default=np.array([0.1]))
    calib_color_maxnodes = ConfigField(isArray=True, default=np.array([-1.0]))
    calib_covmat_maxnodes = ConfigField(isArray=True, default=np.array([-1.0]))
    calib_covmat_nodesize = ConfigField(default=0.15)
    # calib_covmat_min_eigenvalue = ConfigField(default=0.0001)
    # calib_covmat_prior = ConfigField(default=0.45)
    calib_covmat_constant = ConfigField(default=0.9)
    calib_corr_nodesize = ConfigField(default=0.05)
    calib_corr_slope_nodesize = ConfigField(default=0.1)
    calib_corr_nocorrslope = ConfigField(default=True)
    calib_corr_pcut = ConfigField(default=0.9)
    # calib_color_order = ConfigField(isArray=True, default=np.array([-1]))

    calib_color_nsig = ConfigField(default=1.5)
    calib_redspec_nsig = ConfigField(default=2.0)

    calib_colormem_r0 = ConfigField(default=0.5)
    calib_colormem_beta = ConfigField(default=0.0)
    calib_colormem_smooth = ConfigField(default=0.003)
    calib_colormem_minlambda = ConfigField(default=10.0)
    calib_colormem_zbounds = ConfigField(isArray=True, default=np.array([0.4]))
    calib_colormem_colormodes = ConfigField(isArray=True, default=np.array([1, 2]))
    calib_colormem_sigint = ConfigField(isArray=True, default=np.array([0.05, 0.03]))
    calib_pcut = ConfigField(default=0.3)
    calib_color_pcut = ConfigField(default=0.7)

    calib_zlambda_nodesize = ConfigField(default=0.04)
    calib_zlambda_slope_nodesize = ConfigField(default=0.1)
    calib_zlambda_minlambda = ConfigField(default=20.0)
    calib_zlambda_clean_nsig = ConfigField(default=5.0)
    calib_zlambda_correct_niter = ConfigField(default=3)

    calib_lumfunc_alpha = ConfigField(default=-1.0, required=True)

    zredc_binsize_fine = ConfigField(default=0.0001)
    zredc_binsize_coarse = ConfigField(default=0.005)

    bkg_chisqbinsize = ConfigField(default=0.5)
    bkg_refmagbinsize = ConfigField(default=0.2)
    bkg_zbinsize = ConfigField(default=0.02)
    bkg_zredbinsize = ConfigField(default=0.01)
    bkg_deepmode = ConfigField(default=False)
    calib_make_full_bkg = ConfigField(default=True)

    zlambda_pivot = ConfigField(default=30.0, required=True)
    zlambda_binsize = ConfigField(default=0.002, required=True)
    zlambda_maxiter = ConfigField(default=20, required=True)
    zlambda_tol = ConfigField(default=0.0002, required=True)
    zlambda_topfrac = ConfigField(default=0.7, required=True)
    zlambda_epsilon = ConfigField(default=0.005, required=True)
    zlambda_parab_step = ConfigField(default=0.001, required=True)

    centerclass = ConfigField(default='CenteringBCG', required=True)
    wcen_rsoft = ConfigField(default=0.05, required=True)
    wcen_zred_chisq_max = ConfigField(default=100.0, required=True)
    wcen_minlambda = ConfigField(default=10.0, required=True)
    wcen_maxlambda = ConfigField(default=100.0, required=True)
    wcen_cal_zrange = ConfigField(isArray=True, default=np.array([0.0,1.0]))
    wcen_pivot = ConfigField(default=30.0, required=True)
    wcen_uselum = ConfigField(default=True, required=True)
    wcen_Delta0 = ConfigField(required=False, default=0.0)
    wcen_Delta1 = ConfigField(required=False, default=0.0)
    wcen_sigma_m = ConfigField(required=False, default=0.0)
    lnw_cen_sigma = ConfigField(required=False, default=-9999.0)
    lnw_cen_mean = ConfigField(required=False, default=-9999.0)
    lnw_sat_sigma = ConfigField(required=False, default=-9999.0)
    lnw_sat_mean = ConfigField(required=False, default=-9999.0)
    lnw_fg_sigma = ConfigField(required=False, default=-9999.0)
    lnw_fg_mean = ConfigField(required=False, default=-9999.0)
    phi1_mmstar_m = ConfigField(required=False, default=-9999.0)
    phi1_mmstar_slope = ConfigField(required=False, default=-9999.0)
    phi1_msig_m = ConfigField(required=False, default=-9999.0)
    phi1_msig_slope = ConfigField(required=False, default=-9999.0)

    firstpass_r0 = ConfigField(default=0.5, required=True)
    firstpass_beta = ConfigField(default=0.0, required=True)
    firstpass_niter = ConfigField(default=2, required=True)
    firstpass_minlambda = ConfigField(default=3.0, required=True)
    firstpass_centerclass = ConfigField(default='CenteringBCG', required=True)

    likelihoods_r0 = ConfigField(default=1.0, required=True)
    likelihoods_beta = ConfigField(default=0.2, required=True)
    likelihoods_use_zred = ConfigField(default=True, required=True)
    likelihoods_minlambda = ConfigField(default=3.0, required=True)

    percolation_r0 = ConfigField(default=1.0, required=True)
    percolation_beta = ConfigField(default=0.2, required=True)
    percolation_rmask_0 = ConfigField(default=1.5, required=True)
    percolation_rmask_beta = ConfigField(default=0.2, required=True)
    percolation_rmask_gamma = ConfigField(default=0.0, required=True)
    percolation_rmask_zpivot = ConfigField(default=0.3, required=True)
    percolation_lmask = ConfigField()
    percolation_niter = ConfigField(default=2, required=True)
    percolation_minlambda = ConfigField(default=3.0, required=True)
    percolation_pbcg_cut = ConfigField(default=0.5, required=True)
    percolation_maxcen = ConfigField(default=5, required=True)
    percolation_memradius = ConfigField()
    percolation_memlum = ConfigField()

    vlim_lstar = ConfigField(default=0.2, required=False)
    vlim_depthfiles = ConfigField(default=[], required=False, isList=True)
    vlim_bands = ConfigField(default=[], required=False, isList=True)
    vlim_nsigs = ConfigField(default=[], required=False, isArray=True)

    consolidate_lambda_cuts = ConfigField(default=[5.0, 20.0], required=False, isArray=True)
    consolidate_vlim_lstars = ConfigField(default=[0.2, 5.0], required=False, isList=True)
    select_scaleval = ConfigField(default=False, required=True)

    redmagic_calib_nodesize = ConfigField(default=0.05, required=True)
    redmagic_calib_corr_nodesize = ConfigField(default=0.05, required=True)
    redmagic_calib_buffer = ConfigField(default=0.05, required=True)
    redmagic_calib_zbinsize = ConfigField(default=0.02, required=True)
    redmagic_calib_chisqcut = ConfigField(default=20.0, required=True)
    redmagic_zrange = ConfigField(default=[], required=False, isArray=True)
    redmagic_calib_fractrain = ConfigField(default=0.5, required=True)
    redmagic_calib_redshift_buffer = ConfigField(default=0.05, required=True)
    redmagic_maxlum = ConfigField(default=100.0, required=True)
    redmagic_mock_truthspec = ConfigField(default=False, required=True)
    redmagic_run_afterburner = ConfigField(default=True, required=True)
    redmagic_apply_afterburner_zsamp = ConfigField(default=True, required=True)
    redmagic_n0s = ConfigField(default=[], required=True, isArray=True)
    redmagic_etas = ConfigField(default=[], required=True, isArray=True)
    redmagic_names = ConfigField(default=[], required=True, isList=True)
    redmagic_zmaxes = ConfigField(default=[], required=True, isArray=True)
    redmagic_deltakappa = ConfigField(default=0.0, required=False)

    def __init__(self, configfile, outpath=None):
        """
        Instantiate a Configuration object

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename
        outpath: `str`, optional
           Path for output.  Default is None, which uses yaml value.

        Raises
        ------
        RuntimeError:
           When invalid configs are used
        """
        self._reset_vars()

        # First, read in the yaml file
        confdict = read_yaml(configfile)

        self.configpath = os.path.dirname(os.path.abspath(configfile))
        self.configfile = os.path.basename(configfile)

        # And now set the config variables
        self._set_vars_from_dict(confdict)

        # Get the logger
        self.logger = Logger()

        if outpath is not None:
            self.outpath = outpath

        # validate the galfile and refmag
        type(self).__dict__['galfile'].validate('galfile')
        type(self).__dict__['refmag'].validate('refmag')

        # get galaxy file stats
        gal_stats = self._galfile_stats()
        if (self.area is not None):
            if self.depthfile is not None:
                self.logger.info("WARNING: You should not need to set area in the config file when you have a depth map.")
            if (np.abs(self.area - gal_stats['area']) > 1e-3):
                self.logger.info("Config area is not equal to galaxy file area.  Using config area.")
                gal_stats.pop('area')
        else:
            if self.depthfile is None and self.nside > 0 and len(self.hpix) > 0:
                raise RuntimeError("You must set a config file area if no depthfile is present and you are running a sub-region")
        self._set_vars_from_dict(gal_stats, check_none=True)

        if self.limmag_catalog is None:
            self.limmag_catalog = self.limmag_ref

        # Get wcen numbers if available
        self.set_wcen_vals()

        # Set some defaults
        if self.specfile_train is None:
            self.specfile_train = self.specfile

        # Record the cluster dtype for convenience
        self.cluster_dtype = copy.copy(cluster_dtype_base)
        self.cluster_dtype.extend([('MAG', 'f4', self.nmag),
                                   ('MAG_ERR', 'f4', self.nmag),
                                   ('PZBINS', 'f4', self.npzbins),
                                   ('PZ', 'f4', self.npzbins),
                                   ('RA_CENT', 'f8', self.percolation_maxcen),
                                   ('DEC_CENT', 'f8', self.percolation_maxcen),
                                   ('ID_CENT', 'i8', self.percolation_maxcen),
                                   ('LAMBDA_CENT', 'f4', self.percolation_maxcen),
                                   ('ZLAMBDA_CENT', 'f4', self.percolation_maxcen),
                                   ('P_CEN', 'f4', self.percolation_maxcen),
                                   ('Q_CEN', 'f4', self.percolation_maxcen),
                                   ('P_FG', 'f4', self.percolation_maxcen),
                                   ('Q_MISS', 'f4'),
                                   ('P_SAT', 'f4', self.percolation_maxcen),
                                   ('P_C', 'f4', self.percolation_maxcen)])
        self.member_dtype = copy.copy(member_dtype_base)
        self.member_dtype.extend([('MAG', 'f4', self.nmag),
                                  ('MAG_ERR', 'f4', self.nmag)])
        # also need pz stuff, etc, etc.  Will need to deal with defaults

        # Calibration size checking
        self._set_lengths(['calib_colormem_colormodes', 'calib_colormem_sigint'],
                          len(self.calib_colormem_zbounds) + 1)
        self._set_lengths(['calib_color_nodesizes', 'calib_slope_nodesizes',
                           'calib_color_maxnodes', 'calib_covmat_maxnodes'],
                           self.nmag - 1)
                           #'calib_color_order'], self.nmag - 1)

        # redmagic size checks
        self._set_lengths(['redmagic_n0s', 'redmagic_etas', 'redmagic_names',
                           'redmagic_zmaxes'], len(self.redmagic_names))

        # Vlim size checks
        self._set_lengths(['vlim_bands', 'vlim_nsigs'],
                          len(self.vlim_depthfiles))

        # will want to set defaults here...
        self.cosmo = Cosmo()

        # Redmagic stuff
        if len(self.redmagic_zrange) == 0 or self.redmagic_zrange[0] < 0.0 or self.redmagic_zrange[1] < 0.0:
            self.redmagic_zrange = self.zrange[:]

        self._set_lengths(['redmagic_zrange'], 2)

        # Finally, we can validate...
        self.validate()

        # Checks ...
        if self.maskfile is not None and self.mask_mode == 0:
            raise ValueError("A maskfile is set, but mask_mode is 0 (no mask).  Assuming this is not intended.")

        for vlim_band in self.vlim_bands:
            if vlim_band not in self.bands:
                raise ValueError("vlim_band %s not in list of bands!" % (vlim_band))

        # Now set the duplicatable config parameters...
        self.d = DuplicatableConfig(self)

        # Finally, once everything is here, we can make paths
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        if not os.path.exists(os.path.join(self.outpath, self.plotpath)):
            os.makedirs(os.path.join(self.outpath, self.plotpath))

    def validate(self):
        """
        Validate the configuration.

        Raises
        ------
        ValueError:
           If any config field is not legal, ValueError is raised.
        """

        for var in type(self).__dict__:
            try:
                type(self).__dict__[var].validate(var)
            except AttributeError:
                pass

    def copy(self):
        """
        Return a copy of the configuration struct
        """
        return copy.copy(self)

    def _reset_vars(self):
        """
        Internal method to reset variables to defaults
        """
        for var in type(self).__dict__:
            try:
                type(self).__dict__[var].reset()
            except AttributeError:
                pass

    def _set_vars_from_dict(self, d, check_none=False):
        """
        Internal method to set config variables from a dictionary.

        Parameters
        ----------
        d: `dict`
           Dictionary of key/value pairs to set
        check_none: `bool`, optional
           If true, don't set any variables that have a value of None in the dict.
           Default is False
        """
        for key in d:
            if check_none and d[key] is None:
                continue
            if key not in type(self).__dict__:
                raise AttributeError("Unknown config variable: %s" % (key))
            try:
                setattr(self, key, d[key])
            except TypeError:
                raise TypeError("Error with type of variable %s" % (key))

    def _set_lengths(self, l, length):
        """
        Internal method to set the validation length for a list of field names

        Parameters
        ----------
        l: `list`
           Field names to set
        length: `int`
           Validation length
        """
        for arr in l:
            if arr not in type(self).__dict__:
                raise AttributeError("Unknown config variable: %s" % (arr))
            type(self).__dict__[arr].set_length(length)


    def _galfile_stats(self):
        """
        Internal method to get statistics from the galfile

        Returns
        -------
        gal_stats: `dict`
           Dictionary with galaxy file stats

        Raises
        ------
        ValueError:
           Raise error if galaxy file is incorrect format.
        """

        hdr = fitsio.read_header(self.galfile, ext=1)
        pixelated = hdr.get("PIXELS", 0)
        fitsformat = hdr.get("FITS", 0)

        if not fitsformat:
            raise ValueError("Input galfile must describe fits files.")

        gal_stats = {}
        if not pixelated:
            # statistics are from the header
            gal_stats['galfile_pixelized'] = False

            hdrmode = hdr.get("MODE", "").rstrip()

            if hdrmode == 'SDSS':
                gal_stats['survey_mode'] = 0
            elif hdrmode == 'DES':
                gal_stats['survey_mode'] = 1
            elif hdrmode == 'LSST':
                gal_stats['survey_mode'] = 2
            else:
                raise ValueError("Input galaxy file with unknown mode: %s" % (hdrmode))

            ## FIXME: check that these are all in the header
            gal_stats['area'] = hdr.get('AREA', -100.0)
            gal_stats['limmag_ref'] = hdr.get('LIM_REF')
            gal_stats['nmag'] = hdr.get('NMAG')
            gal_stats['zeropoint'] = hdr.get('ZP')
            gal_stats['ref_ind'] = hdr.get(self.refmag.upper()+'_IND')
            gal_stats['b'] = None
            gal_stats['galfile_nside'] = 0
            gal_stats['bands'] = [None]*gal_stats['nmag']

            # figure out the bands...
            for name in hdr:
                m = re.search('(.*)_IND', name)
                if m is None:
                    continue
                if m.groups()[0] == 'REF':
                    continue
                band = m.groups()[0].lower()
                gal_stats['bands'][hdr[name]] = band

        else:
            # statistics are from the master table file
            gal_stats['galfile_pixelized'] = True

            master=fitsio.read(self.galfile, ext=1, upper=True)

            try:
                # Support for old fits reading
                mode = master['MODE'][0].rstrip().decode()
            except AttributeError:
                mode = master['MODE'][0].rstrip()
            if (mode == 'SDSS'):
                gal_stats['survey_mode'] = 0
            elif (mode == 'DES'):
                gal_stats['survey_mode'] = 1
            elif (mode == 'LSST'):
                gal_stats['survey_mode'] = 2
            else:
                raise ValueError("Input galaxy file with unknown mode: %s" % (mode))

            gal_stats['area'] = master['AREA'][0]
            gal_stats['limmag_ref'] = master['LIM_REF'][0]
            gal_stats['nmag'] = master['NMAG'][0]
            if ('B' in master.dtype.names):
                gal_stats['b'] = master['B'][0]
            else:
                gal_stats['b'] = None
            gal_stats['zeropoint'] = master['ZEROPOINT'][0]
            gal_stats['ref_ind'] = master[self.refmag.upper()+'_IND'][0]
            gal_stats['galfile_nside'] = master['NSIDE'][0]
            gal_stats['bands'] = [None]*gal_stats['nmag']

            # Figure out the bands...
            for name in master.dtype.names:
                m = re.search('(.*)_IND', name)
                if m is None:
                    continue
                if m.groups()[0] == 'REF':
                    continue
                band = m.groups()[0].lower()
                gal_stats['bands'][master[name][0]] = band

        if any(x is None for x in gal_stats['bands']):
            # Remove this, and use config values
            gal_stats.pop('bands', None)

        return gal_stats

    def set_wcen_vals(self):
        """
        Set wcen centering values.  These will be loaded from self.wcenfile if
        available.
        """

        wcen_vals = self._wcen_vals()
        if wcen_vals is not None:
            self._set_vars_from_dict(wcen_vals)

    def _wcen_vals(self):
        """
        Load in wcen values from self.wcenfile

        Returns
        -------
        vals: `dict`
           Dictionary of wcen centering parameters
        """

        if self.wcenfile is None or not os.path.isfile(self.wcenfile):
            # We don't have wcen info to load
            return None

        wcen = fitsio.read(self.wcenfile, ext=1, lower=True)

        vals = {'wcen_Delta0': wcen[0]['delta0'],
                'wcen_Delta1': wcen[0]['delta1'],
                'wcen_sigma_m': wcen[0]['sigma_m'],
                'wcen_pivot': wcen[0]['pivot'],
                'lnw_fg_mean': wcen[0]['lnw_fg_mean'],
                'lnw_fg_sigma': wcen[0]['lnw_fg_sigma'],
                'lnw_sat_mean': wcen[0]['lnw_sat_mean'],
                'lnw_sat_sigma': wcen[0]['lnw_sat_sigma'],
                'lnw_cen_mean': wcen[0]['lnw_cen_mean'],
                'lnw_cen_sigma': wcen[0]['lnw_cen_sigma']}

        # New wcen files also record the phi1 information
        if 'phi1_mmstar_m' in wcen[0].dtype.names:
            vals['phi1_mmstar_m'] = wcen[0]['phi1_mmstar_m']
            vals['phi1_mmstar_slope'] = wcen[0]['phi1_mmstar_m']
            vals['phi1_msig_m'] = wcen[0]['phi1_msig_m']
            vals['phi1_msig_slope'] = wcen[0]['phi1_msig_slope']

        return vals

    @property
    def zrange_cushioned(self):
        """Return zrange with additional cushion."""
        zrange_cushioned = self.zrange.copy()
        zrange_cushioned[0] = np.clip(zrange_cushioned[0] - self.calib_zrange_cushion, 0.05, None)
        zrange_cushioned[1] += self.calib_zrange_cushion
        return zrange_cushioned

    def redmapper_filename(self, redmapper_name, paths=None, filetype='fit'):
        """
        Generate a redmapper filename with all the appropriate infixes.

        Parameters
        ----------
        redmapper_name: `str`
           String describing the redmapper file type
        paths: `list` or `tuple`, optional
           List or tuple of path strings to join for filename.
           Default is None, just use self.outpath
        filetype: `str`, optional
           File extension.  Default is 'fit`

        Returns
        -------
        filename: `str`
           Properly formatted full-path redmapper filename
        """
        if paths is None:
            return os.path.join(self.outpath,
                                '%s_%s.%s' % (self.d.outbase, redmapper_name, filetype))
        else:
            if type(paths) is not list and type(paths) is not tuple:
                raise ValueError("paths must be a list or tuple")
            pars = [self.outpath]
            pars.extend(paths)
            pars.append('%s_%s.%s' % (self.d.outbase, redmapper_name, filetype))
            return os.path.join(*pars)

    def check_files(self, check_zredfile=False, check_bkgfile=False, check_bkgfile_components=False,
                    check_parfile=False, check_zlambdafile=False):
        """
        Check that all calibration files are available for a cluster finder run.

        Parameters
        ----------
        check_zredfile: `bool`, optional
           Check that the zred file is available.  Default is False.
        check_bkgfile: `bool`, optional
           Check that the background file is available.  Default is False.
        check_bkgfile_components: `bool`, optional
           Check that the background file zred/chisq components are available.
           Default is False.
        check_parfile: `bool`, optional
           Check that the red sequence parameter file is available.  Default is False.
        check_zlambdafile: `bool`, optional
           Check that the zlambda calibration file is available.  Default is False.

        Raises
        ------
        ValueError:
           Raise ValueError if any of the check files is missing.
        """
        if check_zredfile:
            if not os.path.isfile(self.zredfile):
                raise ValueError("zredfile %s not found." % (self.zredfile))

        if check_bkgfile:
            if not os.path.isfile(self.bkgfile):
                raise ValueError("bkgfile %s not found." % (self.bkgfile))

        if check_bkgfile_components:
            with fitsio.FITS(self.bkgfile) as fits:
                if 'CHISQBKG' not in [ext.get_extname() for ext in fits[1: ]]:
                    raise ValueError("bkgfile %s does not have CHISQBKG extension." % (self.bkgfile))
                if 'ZREDBKG' not in [ext.get_extname() for ext in fits[1: ]]:
                    raise ValueError("bkgfile %s does not have ZREDBKG extension." % (self.bkgfile))

        if check_parfile:
            if not os.path.isfile(self.parfile):
                raise ValueError("parfile %s not found." % (self.parfile))

        if check_zlambdafile:
            if not os.path.isfile(self.zlambdafile):
                raise ValueError("zlambdafile %s not found." % (self.zlambdafile))

    def output_yaml(self, filename):
        """
        Output config into a yaml file.

        Parameters
        ----------
        filename: `str`
           Output yaml filename
        """
        out_dict = {}
        for key in type(self).__dict__:
            if isinstance(type(self).__dict__[key], ConfigField):
                if type(self).__dict__[key]._isArray:
                    # Convert all elements to python types and make a list
                    out_dict[key] = np.ndarray.tolist(type(self).__dict__[key]._value)
                else:
                    # Try to use numpy to convert to scalar; if it doesn't work then
                    # it's not numpy and we can use the value directly
                    try:
                        out_dict[key] = type(self).__dict__[key]._value.item()
                    except (ValueError, AttributeError, TypeError):
                        out_dict[key] = type(self).__dict__[key]._value

        with open(filename, 'w') as f:
            yaml.dump(out_dict, stream=f)

    def compute_border(self):
        """
        Compute the border radii based on the largest expected cluster at the lowest
        redshift.

        Returns
        -------
        rad: `float`
           Border radius for overlapping tiles for parallel runs.
        """

        maxdist = 1.05 * self.percolation_rmask_0 * (300. / 100.)**self.percolation_rmask_beta
        radius = maxdist / (np.radians(1.) * self.cosmo.Da(0, self.zrange[0]))

        return 3.0 * radius
