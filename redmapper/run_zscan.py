"""Class to compute richness for a set of ra/dec positions by scanning redshift.
"""
from functools import reduce
import sys

import fitsio
import numpy as np
import esutil

from .cluster import ClusterCatalog
from .background import Background
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
from .zlambda import Zlambda
from .zlambda import ZlambdaCorrectionPar
from .cluster_runner import ClusterRunner
from .centering import CenteringBCG, CenteringWcenZred, CenteringRandom, CenteringRandomSatellite


class RunZScan(ClusterRunner):
    """
    The RunZScan class is derived from ClusterRunner, and will compute
    richness, redshift (z_lambda), and centering for an input catalog
    that has ra/dec, by scanning through all redshifts.
    """

    def _additional_initialization(self, **kwargs):
        """Additional initialization for RunZScan."""
        # This is the runmode where we get the mask/radius config vars from
        self.runmode = 'zscan'
        self.read_zreds = True
        self.zreds_required = True
        self.zredbkg_required = True
        self.cutgals_bkgrange = True
        self.cutgals_chisqmax = False
        self.use_rmask_settings = False
        self.filetype = 'zscan'

    def run(self, *args, **kwargs):
        """Run a catalog through RunZScan.

        Loops over all positions and perform RunZScan computations on each pos.
        """
        return super(RunZScan, self).run(*args, **kwargs)

    def _more_setup(self, *args, **kwargs):
        """More setup for RunZScan.
        """
        # Read in the catalog, etc.
        self.config.logger.info("Reading in catalog file...")
        self.cat = ClusterCatalog.from_catfile(self.config.catfile,
                                               zredstr=self.zredstr,
                                               config=self.config,
                                               bkg=self.bkg,
                                               zredbkg=self.zredbkg,
                                               cosmo=self.cosmo,
                                               r0=self.r0,
                                               beta=self.beta)

        # Add additional columns to the catalog.
        self.nzstep = int(np.ceil((self.config.zrange[1] - self.config.zrange[0])/self.config.zscan_zstep))
        self.z_array = np.arange(self.nzstep, dtype=np.float64)*self.config.zscan_zstep + self.config.zrange[0]

        zscan_dtype = [('z_steps', 'f4', self.nzstep),
                       ('lambda_steps', 'f4', self.nzstep),
                       ('likelihood_steps', 'f4', self.nzstep),
                       ('lmax', 'f4'),
                       ('max_ind', 'i4'),
                       ('zmax', 'f4'),
                       ('ra_opt', 'f8'),
                       ('dec_opt', 'f8'),
                       ('lambda_opt', 'f4'),
                       ('lambda_opt_e', 'f4'),
                       ('z_lambda_opt', 'f4'),
                       ('z_lambda_opt_e', 'f4')]

        self.cat.add_fields(zscan_dtype)

        # check if we need to generate mem_match_ids
        self._generate_mem_match_ids()

        self.cat.ra_orig = self.cat.ra
        self.cat.dec_orig = self.cat.dec

        # The initial redshift, for matching, is the lowest redshift
        self.cat.z_init = self.z_array[0]
        self.cat.z = self.z_array[0]

        self.do_percolation_masking = False
        self.do_lam_plusminus = True
        self.match_centers_to_galaxies = False
        self.record_members = True
        self.do_correct_zlambda = True
        self.do_pz = True
        self.use_maxmag_in_matching = False

        self.min_lambda = self.config.zscan_minlambda

        self.refine_r0 = self.config.percolation_r0
        self.refine_beta = self.config.percolation_beta

        if self.refine_beta == 0.0:
            self.refine_maxrad = 1.2*self.refine_r0
        else:
            self.refine_maxrad = self.refine_r0*(300./100.)**self.refine_beta

        return True

    def _process_cluster(self, cluster):
        """Process a single position with RunZScan.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to compute richness.
        """
        bad = False
        done = False

        cluster.z_steps[:] = self.z_array
        cluster.lambda_steps[:] = -1.0
        cluster.likelihood_steps[:] = -1.0
        cluster.lmax = -1.0
        cluster.max_ind = -1
        cluster.zmax = -1.0

        if self.depthstr is None:
            # Compute the mask depth from the galaxies in the region
            # This sets all the maskgals limmag to the same value
            self.depthlim.calc_maskdepth(self.mask.maskgals,
                                         cluster.neighbors.refmag,
                                         cluster.neighbors.refmag_err)

        # First pass, do the full scan
        for zb, zuse in enumerate(self.z_array):
            # Set the cluster redshift, and update distances
            cluster.redshift = zuse
            cluster.update_neighbors_dist()

            # Select galaxies that are bright enough, within the radius
            maxmag = cluster.mstar - 2.5*np.log10(self.limlum)

            lc, = np.where((cluster.neighbors.refmag < maxmag) &
                           (cluster.neighbors.r < self.maxrad))

            if lc.size < 2:
                # There is nothing at this redshift
                continue

            # Update the mask radii
            self.mask.set_radmask(cluster)

            if self.depthstr is not None:
                self.depthstr.calc_maskdepth(self.mask.maskgals,
                                             cluster.ra, cluster.dec, cluster.mpc_scale)

            # Compute the richness
            lam = cluster.calc_richness(self.mask, index=lc, calc_err=False)

            if lam < self.min_lambda:
                # There is nothing at this redshift
                continue

            # Compute likelihood, since we have a cluster here;
            # we want to avoid the central galaxy for this computation
            # (if there is one).
            incut, = np.where((cluster.neighbors.pmem > 0.0) &
                              (cluster.neighbors.r > 1e-6) &
                              (cluster.neighbors.pmem < 1.0))
            if incut.size > 0:
                like = -lam/cluster.scaleval - np.sum(np.log(1.0 - cluster.neighbors.pmem[incut]))
            else:
                like = -1.0

            # Record
            cluster.lambda_steps[zb] = lam
            cluster.likelihood_steps[zb] = like

        # Find max likelihood
        cluster.max_ind = np.argmax(cluster.likelihood_steps)
        cluster.lmax = cluster.likelihood_steps[cluster.max_ind]
        cluster.zmax = cluster.z_steps[cluster.max_ind]

        # If bad, quit out.
        bad = False
        if cluster.lambda_steps[cluster.max_ind] < self.min_lambda:
            bad = True
            return bad

        # Iterate to get refinement.
        zuse = cluster.zmax
        cluster.r0 = self.refine_r0
        cluster.beta = self.refine_beta

        # Re-match neighbors using larger radius, new redshift
        # We expand the radius here to allow for recentering below
        cluster.redshift = zuse
        maxmag = cluster.mstar - 2.5*np.log10(self.limlum)

        cluster.find_neighbors(2.0*self.refine_maxrad, self.gals, megaparsec=True, maxmag=maxmag)

        # Iterate to refine the redshift/richness using the full radius
        bad = False
        for i in range(2):
            if bad:
                continue

            # Set the redshift and update values
            cluster.redshift = zuse

            # Do mask computations
            self.mask.set_radmask(cluster)
            if self.depthstr is not None:
                self.depthstr.calc_maskdepth(self.mask.maskgals,
                                             cluster.ra, cluster.dec, cluster.mpc_scale)

            # Compute richness and error
            lam = cluster.calc_richness(self.mask)

            if (lam/cluster.scaleval < self.min_lambda):
                bad = True
                continue

            # Compute z_lambda
            if i == 0:
                zlam = Zlambda(cluster)
                z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, self.mask,
                                                         calc_err=True, calcpz=True)
                cluster.z_lambda = z_lambda
                cluster.z_lambda_e = z_lambda_e
                cluster.pzbins[:] = zlam.pzbins
                cluster.pz[:] = zlam.pz

                if z_lambda > 0.0:
                    zuse = z_lambda

            if cluster.z_lambda < 0.0:
                bad = True
                continue

        if bad:
            self._reset_bad_values(cluster)
            return bad

        cluster.redshift = cluster.z_lambda

        # Compute optical center and statistics
        cent = reduce(getattr, self.config.centerclass.split('.'), sys.modules[__name__])(cluster)

        if not cent.find_center() or cent.ngood == 0:
            self.config.logger.info("Could not find optical center on a cluster.")
            # Note that this is not a _bad_ cluster per-se.
            return False

        cluster.ra_opt = cent.ra[0]
        cluster.dec_opt = cent.dec[0]

        # only update central galaxy values if we centered on a galaxy
        #  (this is typical, but not required for a centering module)
        if cent.index[0] >= 0:
            cluster.mag[:] = cluster.neighbors.mag[cent.index[0], :]
            cluster.mag_err[:] = cluster.neighbors.mag_err[cent.index[0], :]
            cluster.refmag = cluster.neighbors.refmag[cent.index[0]]
            cluster.refmag_err = cluster.neighbors.refmag_err[cent.index[0]]
            cluster.ebv_mean = cluster.neighbors.ebv[cent.index[0]]
            if self.did_read_zreds:
                cluster.zred = cluster.neighbors.zred[cent.index[0]]
                cluster.zred_e = cluster.neighbors.zred_e[cent.index[0]]
                cluster.zred_chisq = cluster.neighbors.zred_chisq[cent.index[0]]

            cluster.id_cent[:] = cluster.neighbors.id[cent.index]

        # And update the center info...
        cluster.ncent_good = cent.ngood
        cluster.ra_cent[:] = cent.ra
        cluster.dec_cent[:] = cent.dec
        cluster.p_cen[:] = cent.p_cen
        cluster.q_cen[:] = cent.q_cen
        cluster.p_fg[:] = cent.p_fg
        cluster.q_miss = cent.q_miss
        cluster.p_sat[:] = cent.p_sat
        cluster.p_c[:] = cent.p_c

        # Make a copy of the cluster for computing optical properties
        opt_cluster = cluster.copy()
        opt_cluster.ra = cluster.ra_opt
        opt_cluster.dec = cluster.dec_opt
        opt_cluster.update_neighbors_dist()

        # Compute optical-center richness and redshift
        lam = opt_cluster.calc_richness(self.mask)

        cluster.lambda_opt = opt_cluster.Lambda
        cluster.lambda_opt_e = opt_cluster.Lambda_e

        zlam = Zlambda(opt_cluster)
        z_lambda_opt, z_lambda_opt_e = zlam.calc_zlambda(opt_cluster.redshift, self.mask,
                                                         calc_err=True, calcpz=False)

        cluster.z_lambda_opt = z_lambda_opt
        cluster.z_lambda_opt_e = z_lambda_opt_e
