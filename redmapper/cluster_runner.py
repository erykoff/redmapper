"""Base class to run a cluster catalog through one stage of processing.

The ClusterRunner class is the base class for looping over all the clusters in
a catalog and performing computations on each cluster.  This includes first
passes, percolation, richness computation at existing positions/redshifts, etc.

This class has numerous methods that are overridden by the derived classes.
"""
import fitsio
import numpy as np
import esutil
import os
import gc
from esutil.cosmology import Cosmo

from .configuration import Configuration
from .cluster import ClusterCatalog
from .background import Background, ZredBackground
from .color_background import ColorBackground
from .mask import get_mask
from .galaxy import GalaxyCatalog
from .catalog import Catalog
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
from .zlambda import Zlambda
from .zlambda import ZlambdaCorrectionPar
from .redsequence import RedSequenceColorPar
from .depth_fitting import DepthLim
from .utilities import getMemoryString

###################################################
# Order of operations:
#  __init__()
#    _additional_initialization() [override this]
#  run()
#    _setup()
#    _more_setup() [override this]
#    _process_cluster() [override this]
#    _postprocess() [override this]
#  output()
###################################################

class ClusterRunner(object):
    """
    The ClusterRunner class is the base class for looping over all cluster in a
    catalog and performing computations on each cluster.
    """

    def __init__(self, conf, **kwargs):
        """
        Instantiate a ClusterRunner object.

        Parameters
        ----------
        conf: `redmapper.Configuration` or `str`
           Configuration object or filename of configuration object
        **kwargs: Defined by derived classes.
        """
        if not isinstance(conf, Configuration):
            # this needs to be read
            self.config = Configuration(conf)
        else:
            self.config = conf

        # Generic defaults
        self.read_gals = True
        self.read_zreds = False
        self.zreds_required = False
        self.did_read_zreds = False
        self.zredbkg_required = False
        self.use_colorbkg = False
        self.use_parfile = True
        self.use_maxmag_in_matching = True
        self.use_rmask_settings = True
        self._filename = None

        # Will want to add stuff to check that everything needed is present?

        self._additional_initialization(**kwargs)

    def _additional_initialization(self, **kwargs):
        """
        Perform additional initialization, specific to each derived class.

        Must be overridden.
        """

        # must be overridden
        self.runmode = None
        self.filetype = None

        raise RuntimeError("Method _additional_initialization requires override")

    def _setup(self):
        """
        General setup for ClusterRunner, configuring quantities that are common
        across all derived classes.
        """
        self.r0 = self.config.__getattribute__(self.runmode + '_r0')
        self.beta = self.config.__getattribute__(self.runmode + '_beta')

        # This always uses the "percolation" settings, maybe?
        if self.use_rmask_settings:
            self.rmask_0 = self.config.percolation_rmask_0
            self.rmask_beta = self.config.percolation_rmask_beta
            self.rmask_gamma = self.config.percolation_rmask_gamma
            self.rmask_zpivot = self.config.percolation_rmask_zpivot
        else:
            self.rmask_0 = self.r0
            self.rmask_beta = self.beta
            self.rmask_gamma = self.config.percolation_rmask_gamma
            self.rmask_zpivot = self.config.percolation_rmask_zpivot

        if self.config.percolation_lmask < 0.0:
            self.percolation_lmask = self.config.lval_reference
        else:
            self.percolation_lmask = self.config.percolation_lmask

        # maxrad is the maximum dist to match neighbors.  maxrad2 is for masking
        # (note this was "maxdist" in IDL which was a bad name)
        if self.beta == 0.0:
            # add a small cushion even if we have constant radius
            self.maxrad = 1.2 * self.r0
            self.maxrad2 = 1.2 * self.rmask_0
        else:
            # maximum possible richness is 300
            self.maxrad = self.r0 * (300./100.)**self.beta
            self.maxrad2 = self.rmask_0 * (300./100.)**self.rmask_beta

        if self.maxrad2 > self.maxrad:
            self.maxrad = self.maxrad2

        if self.config.bkg_local_compute or self.config.bkg_local_use:
            if self.config.bkg_local_annuli[1] > self.maxrad:
                self.maxrad = self.config.bkg_local_annuli[1]

        # read in background
        if self.use_colorbkg:
            self.cbkg = ColorBackground(self.config.bkgfile_color, usehdrarea=True)
            self.bkg = None
        else:
            self.bkg = Background(self.config.bkgfile)
            self.cbkg = None

        if self.zredbkg_required:
            self.zredbkg = ZredBackground(self.config.bkgfile)
        else:
            self.zredbkg = None

        # read in parameters
        if self.use_parfile:
            self.zredstr = RedSequenceColorPar(self.config.parfile, fine=True)
        else:
            self.zredstr = RedSequenceColorPar(None, config=self.config)

        # And correction parameters
        try:
            self.zlambda_corr = ZlambdaCorrectionPar(parfile=self.config.zlambdafile,
                                                     zlambda_pivot=self.config.zlambda_pivot)
        except:
            self.zlambda_corr = None

        # read in mask (if available)
        # This will read in the mask and maskgals
        #  Will work with any type of mask
        self.mask = get_mask(self.config)

        # read in the depth structure
        try:
            self.depthstr = DepthMap(self.config)
        except:
            self.depthstr = None

        if self.depthstr is None and not self.read_gals:
            raise RuntimeError("Must have a valid depthstr if read_gals is False")

        self.cosmo = self.config.cosmo

        # read in the galaxies
        # Use self.read_zreds to know if we should read them!
        # And self.zreds_required to know if we *must* read them

        if self.read_zreds:
            zredfile = self.config.zredfile
        else:
            zredfile = None

        if self.zreds_required and zredfile is None:
            raise RuntimeError("zreds are required, but zredfile is None")

        self.gals = None

        if self.read_gals:
            self.gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                                   nside=self.config.d.nside,
                                                   hpix=self.config.d.hpix,
                                                   border=self.config.border,
                                                   zredfile=zredfile)

            # If the zredfile is not None and we didn't raise an exception,
            # then we successfully read in the zreds
            if zredfile is not None:
                self.did_read_zreds = True

            # Cut galaxies if desired
            if self.cutgals_bkgrange:
                refmag_low = self.bkg.refmagbins[0]
                refmag_high = self.bkg.refmagbins[-1] + (self.bkg.refmagbins[1] - self.bkg.refmagbins[0])
            else:
                refmag_low = -1000.0
                refmag_high = 1000.0

            guse = ((self.gals.refmag > refmag_low) & (self.gals.refmag < refmag_high))

            if self.did_read_zreds and self.cutgals_chisqmax:
                guse &= (self.gals.chisq < self.config.chisq_max)

            # Cut the input galaxy file
            self.gals = self.gals[guse]

            if len(self.gals) == 0:
                self.config.logger.info("No good galaxies for %s in pixel %s" %
                                        (self.runmode, self.hpix_logstr))
                return False

        # If we don't have a depth map, get ready to compute local depth
        if self.depthstr is None:
            try:
                self.depthlim = DepthLim(self.gals.refmag, self.gals.refmag_err)
            except RuntimeError:
                self.config.logger.info("Failed to obtain depth info in %s for pixel %s with %d galaxies.  Skipping pixel." % (self.runmode, self.hpix_logstr, len(self.gals)))
                return False

        # default limiting luminosity
        self.limlum = np.clip(self.config.lval_reference - 0.1, 0.01, None)

        # Defaults for whether to implement masking, etc
        self.do_percolation_masking = False
        self.do_lam_plusminus = False
        self.use_memradius = False
        self.use_memlum = False
        self.match_centers_to_galaxies = False
        self.min_lambda = -1.0
        self.record_members = False
        self.doublerun = False
        self.do_correct_zlambda = False
        self.do_pz = False

        if len(self.config.d.hpix) == 0:
            self.hpix_logstr = "All"
        else:
            self.hpix_logstr = ", ".join(str(x) for x in self.config.d.hpix)

        return True

    def _more_setup(self, *args, **kwargs):
        """
        Additional setup for derived ClusterRunner classes.

        This may be overridden if necessary.
        """
        # This is to be overridden if necessary
        # this can receive all the keywords.

        return True

    def _generate_mem_match_ids(self):
        """
        Internal method to compute unique cluster mem_match_ids
        """
        min_id = self.cat.mem_match_id.min()
        max_id = self.cat.mem_match_id.max()

        if (min_id == max_id):
            # These are unset: generate them
            self.cat.mem_match_id = np.arange(self.cat.size) + 1
        else:
            # Make sure they are unique
            if np.unique(self.cat.mem_match_id).size != self.cat.size:
                raise RuntimeError("Input values for mem_match_id are not unique (and not all unset)")

    def _reset_bad_values(self, cluster):
        """
        Internal method to reset all cluster values to "bad" values.
        """
        cluster.Lambda = -1.0
        cluster.Lambda_e = -1.0
        cluster.scaleval = -1.0
        cluster.z_lambda = -1.0
        cluster.z_lambda_e = -1.0

    def _process_cluster(self, cluster):
        """
        Process a single cluster.

        This must be overridden by the derived classes.
        """
        # This must be overridden
        raise RuntimeError("_process_cluster must have an override")

    def run(self, *args, **kwargs):
        """
        Run a catalog through the ClusterRunner.

        Loop over all clusters and perform computations as described in
        self._process_cluster(cluster) on each cluster.
        """

        # General setup
        if not self._setup():
            self.config.logger.info("Cluster initialization failed on pixel %s. No catalog will be produced." % (self.hpix_logstr))
            self.cat = None
            return

        # Setup specific for a given task.  Will read in the galaxy catalog.
        if not self._more_setup(*args, **kwargs):
            self.config.logger.info("Cluster initialization failed on pixel %s. No catalog will be produced." % (self.hpix_logstr))
            self.cat = None
            return

        # Match centers and galaxies if required
        if self.match_centers_to_galaxies:
            i0, i1, dist = self.gals.match_many(self.cat.ra, self.cat.dec, 1./3600.)
            self.cat.refmag[i0] = self.gals.refmag[i1]
            self.cat.refmag_err[i0] = self.gals.refmag_err[i1]
            self.cat.mag[i0, :] = self.gals.mag[i1, :]
            self.cat.mag_err[i0, :] = self.gals.mag_err[i1, :]

            if self.did_read_zreds:
                self.cat.zred = self.gals.zred[i1]
                self.cat.zred_e = self.gals.zred_e[i1]
                self.cat.zred_chisq = self.gals.zred_chisq[i1]

        # loop over clusters...
        # at the moment, we are doing the matching once per cluster.
        # if this proves too slow we can prematch bulk clusters as in the IDL code

        if self.do_percolation_masking or self.doublerun:
            self.pgal = np.zeros(self.gals.size, dtype=np.float32)

        self.members = None

        if self.doublerun:
            nruniter = 2
        else:
            nruniter = 1

        for it in range(nruniter):

            if self.doublerun:
                # This mode allows two passes, with a sort in between.
                if it == 0:
                    self.config.logger.info("First iteration...")
                    self.do_percolation_masking = False
                else:
                    self.config.logger.info("Second iteration with percolation...")
                    self._doublerun_sort()
                    self.do_percolation_masking = True
                    self.record_members = True

            cctr = 0

            for cluster in self.cat:
                cluster.maskgal_index = self.mask.select_maskgals_sample()

                if ((cctr % 1000) == 0):
                    self.config.logger.info("%s: Working on cluster %d of %d" % (self.hpix_logstr, cctr, self.cat.size))
                cctr += 1

                # Note that the cluster is set with .z if available! (which becomes .redshift)
                if self.use_maxmag_in_matching:
                    maxmag = cluster.mstar - 2.5*np.log10(self.limlum)
                else:
                    maxmag = None

                if self.read_gals:
                    cluster.find_neighbors(self.maxrad, self.gals, megaparsec=True, maxmag=maxmag)

                    if cluster.neighbors.size == 0:
                        self._reset_bad_values(cluster)
                        continue

                    if self.do_percolation_masking:
                        cluster.neighbors.pfree[:] = 1.0 - self.pgal[cluster.neighbors.index]
                    else:
                        cluster.neighbors.pfree[:] = 1.0

                # FIXME: add mean ebv computation here.

                if self.depthstr is None:
                    # must approximate the limiting magnitude

                    self.depthlim.calc_maskdepth(self.mask.maskgals,
                                                 cluster.neighbors.refmag, cluster.neighbors.refmag_err)
                else:
                    # get from the depth structure
                    self.depthstr.calc_maskdepth(self.mask.maskgals,
                                                 cluster.ra, cluster.dec, cluster.mpc_scale)

                cluster.lim_exptime = np.median(self.mask.maskgals.exptime)
                cluster.lim_limmag = np.median(self.mask.maskgals.limmag)
                cluster.lim_limmag_hard = self.config.limmag_catalog

                # And survey masking (this may be a dummy)
                self.mask.set_radmask(cluster)

                # And compute maskfrac here...approximate first computation
                inside, = np.where(self.mask.maskgals.r < 1.0)
                bad, = np.where(self.mask.maskgals.mark[inside] == 0)
                cluster.maskfrac = float(bad.size) / float(inside.size)

                if cluster.maskfrac == 1.0 or cluster.lim_limmag <= 1.0:
                    # This is a very bad cluster, and should not be used
                    bad_cluster = True
                else:
                    # Do the cluster processing
                    bad_cluster = self._process_cluster(cluster)

                if bad_cluster:
                    # This is a bad cluster and we can't continue
                    self._reset_bad_values(cluster)
                    continue

                if self.read_gals:
                    if self.config.bkg_local_compute and not self.config.bkg_local_use:
                        if self.depthstr is None:
                            depth = self.depthlim
                        else:
                            depth = self.depthstr
                        cluster.bkg_local = cluster.compute_bkg_local(self.mask, depth)

                if self.do_correct_zlambda and self.zlambda_corr is not None and self.read_gals:
                    if self.do_pz:
                        zlam, zlam_e, pzbins, pzvals = self.zlambda_corr.apply_correction(cluster.Lambda,
                                                                                          cluster.z_lambda,
                                                                                          cluster.z_lambda_e,
                                                                                          pzbins=cluster.pzbins,
                                                                                          pzvals=cluster.pz)
                        cluster.pzbins = pzbins
                        cluster.pzvals = pzvals
                    else:
                        zlam, zlam_e = self.zlambda_corr.apply_correction(cluster.Lambda,
                                                                          cluster.z_lambda,
                                                                          cluster.z_lambda_e)
                    cluster.z_lambda = zlam
                    cluster.z_lambda_e = zlam_e

                # compute updated maskfrac (always)
                inside, = np.where(self.mask.maskgals.r < cluster.r_lambda)
                bad, = np.where(self.mask.maskgals.mark[inside] == 0)
                if inside.size == 0:
                    cluster.maskfrac = 1.0
                else:
                    cluster.maskfrac = float(bad.size) / float(inside.size)

                # compute additional dlambda bits (if desired)
                if self.do_lam_plusminus and self.read_gals:
                    cluster_temp = cluster.copy()

                    cluster_temp.redshift = cluster.z_lambda - self.config.zlambda_epsilon
                    lam_zmeps = cluster_temp.calc_richness(self.mask)
                    elambda_zmeps = cluster_temp.lambda_e
                    cluster_temp.redshift = cluster.z_lambda + self.config.zlambda_epsilon
                    lam_zpeps = cluster_temp.calc_richness(self.mask)
                    elambda_zpeps = cluster_temp.lambda_e

                    if (lam_zmeps > 0 and lam_zpeps > 0):
                        # Only compute if these are valid
                        # During training, when we use the seed redshifts,
                        #  we could fall out of the good range for a cluster
                        cluster.dlambda_dz = (np.log(lam_zpeps) - np.log(lam_zmeps)) / (2. * self.config.zlambda_epsilon)
                        cluster.dlambda_dz2 = (np.log(lam_zpeps) + np.log(lam_zmeps) - 2.*np.log(cluster.Lambda)) / (self.config.zlambda_epsilon**2.)

                        cluster.dlambdavar_dz = (elambda_zpeps**2. - elambda_zmeps**2.) / (2.*self.config.zlambda_epsilon)
                        cluster.dlambdavar_dz2 = (elambda_zpeps**2. + elambda_zmeps**2. - 2.*cluster.Lambda_e**2.) / (self.config.zlambda_epsilon**2.)

                # and record pfree if desired
                if self.do_percolation_masking and self.read_gals:
                    # FIXME
                    r_mask = (self.rmask_0 * (cluster.Lambda/100.)**self.rmask_beta *
                              ((1. + cluster.redshift)/(1. + self.rmask_zpivot))**self.rmask_gamma)
                    if (r_mask < cluster.r_lambda):
                        r_mask = cluster.r_lambda
                    cluster.r_mask = r_mask

                    lim = cluster.mstar - 2.5*np.log10(self.percolation_lmask)

                    u, = np.where((cluster.neighbors.refmag < lim) &
                                  (cluster.neighbors.r < r_mask) &
                                  (cluster.neighbors.p > 0.0))
                    if (u.size > 0):
                        self.pgal[cluster.neighbors.index[u]] += cluster.neighbors.p[u]

                # and save members
                # Note that this is probably horribly inefficient for memory
                #  usage right now, but will start here and fix later if it
                #  is a problem.

                if self.read_gals:
                    pfree_temp = cluster.neighbors.pfree[:]

                if (self.use_memradius or self.use_memlum) and self.read_gals:
                    ok = (cluster.neighbors.p > 0.01)

                    if self.use_memradius:
                        ok &= (cluster.neighbors.r < self.config.percolation_memradius * cluster.r_lambda)
                    if self.use_memlum:
                        ok &= (cluster.neighbors.refmag < (cluster.mstar - 2.5*np.log10(self.config.percolation_memlum)))

                    # And set pfree_temp to zero when it is not okay
                    pfree_temp[~ok] = 0.0
                elif self.read_gals:
                    # Only save members where pmem > 0.01 (for space)
                    ok = (cluster.neighbors.pmem > 0.01)
                    pfree_temp[~ok] = 0.0

                if self.record_members and self.read_gals:
                    pfree_temp = cluster.neighbors.pfree[:]

                    if self.use_memradius or self.use_memlum:
                        ok = (cluster.neighbors.p > 0.01)

                        if self.use_memradius:
                            ok &= (cluster.neighbors.r < self.config.percolation_memradius * cluster.r_lambda)
                        if self.use_memlum:
                            ok &= (cluster.neighbors.refmag < (cluster.mstar - 2.5*np.log10(self.config.percolation_memlum)))

                        # And set pfree_temp to zero when it is not okay
                        pfree_temp[~ok] = 0.0
                    else:
                        # Only save members where pmem > 0.01 (for space)
                        ok = (cluster.neighbors.pmem > 0.01)
                        pfree_temp[~ok] = 0.0

                    memuse, = np.where(pfree_temp > 0.01)
                    mem_temp = Catalog.zeros(memuse.size, dtype=self.config.member_dtype)

                    mem_temp.mem_match_id[:] = cluster.mem_match_id
                    mem_temp.id[:] = cluster.neighbors.id[memuse]
                    mem_temp.z[:] = cluster.redshift
                    mem_temp.ra[:] = cluster.neighbors.ra[memuse]
                    mem_temp.dec[:] = cluster.neighbors.dec[memuse]
                    mem_temp.r[:] = cluster.neighbors.r[memuse]
                    mem_temp.p[:] = cluster.neighbors.p[memuse]
                    mem_temp.pfree[:] = pfree_temp[memuse]
                    mem_temp.pcol[:] = cluster.neighbors.pcol[memuse]
                    mem_temp.theta_i[:] = cluster.neighbors.theta_i[memuse]
                    mem_temp.theta_r[:] = cluster.neighbors.theta_r[memuse]
                    mem_temp.refmag[:] = cluster.neighbors.refmag[memuse]
                    mem_temp.refmag_err[:] = cluster.neighbors.refmag_err[memuse]
                    if (self.did_read_zreds):
                        mem_temp.zred[:] = cluster.neighbors.zred[memuse]
                        mem_temp.zred_e[:] = cluster.neighbors.zred_e[memuse]
                    mem_temp.chisq[:] = cluster.neighbors.chisq[memuse]
                    mem_temp.ebv[:] = cluster.neighbors.ebv[memuse]
                    mem_temp.mag[:, :] = cluster.neighbors.mag[memuse, :]
                    mem_temp.mag_err[:, :] = cluster.neighbors.mag_err[memuse, :]

                    # Worried this is going to be slow...
                    if self.members is None:
                        self.members = mem_temp
                    else:
                        self.members.append(mem_temp)

        self._postprocess()
        self._cleanup()

    def _postprocess(self):
        """
        Perform cluster catalog post-processing.

        This may be overridden in derived classes.
        """

        # default post-processing...
        use, = np.where(self.cat.Lambda >= self.min_lambda)

        # Make a new catalog that doesn't have the extra memory usage
        # from catalogs and neighbors
        self.cat = ClusterCatalog(self.cat._ndarray[use])

        # And crop down members?
        if self.members is not None:
            a, b = esutil.numpy_util.match(self.cat.mem_match_id,
                                           self.members.mem_match_id)
            self.members = Catalog(self.members._ndarray[b])

    def _cleanup(self):
        """
        Clean up memory usage.
        """

        # Release references to allow garbage collection to run
        del self.gals
        del self.bkg
        del self.cbkg
        del self.zredbkg
        del self.zredstr
        del self.depthstr
        del self.zlambda_corr
        del self.mask
        del self.cosmo

        gc.collect()

    def output(self, savemembers=True, withversion=True, clobber=False, outbase=None):
        """
        Output the cluster catalog.

        Parameters
        ----------
        savemembers: `bool`, optional
           Save the members along with the catalog?  Default is True.
        withversion: `bool`, optional
           Should the filename contain the redmapper code version?  Default is True.
        clobber: `bool`, optional
           Clobber existing file?  Default is False.
        outbase: `str`, optional
           Override file output base (from self.config.d.outbase).  Default is None.
        """

        # Try with a universal method.
        # It will save the underlying _ndarrays of the Cluster and
        # (optionally) Member catalogs.

        self._filename = self.config.redmapper_filename(self.filetype + '_catalog', withversion=withversion, outbase=outbase)

        if self.cat is None:
            self.config.logger.info("Warning: no catalog generated for %s" % (self._filename))
            return

        self.config.logger.info("Writing catalog to file: %s" % (self._filename))
        self.cat.to_fits_file(self._filename, clobber=clobber)

        if savemembers:
            if self.members is None:
                self.config.logger.info("Warning: no members generated for %s" % (self._filename))
                return
            memfilename = self.config.redmapper_filename(self.filetype + '_catalog_members', withversion=withversion, outbase=outbase)
            self.members.to_fits_file(memfilename, clobber=clobber)

    @property
    def filename(self):
        """
        Get filename used to save the derived catalog.
        """
        if self._filename is None:
            # Return the unversioned, default filename
            return self.config.redmapper_filename(self.filetype + '_catalog')
        else:
            return self._filename
