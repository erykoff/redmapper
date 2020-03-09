"""Class to run the redmapper randoms using zmask randoms.
"""


from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import esutil
import os

from .cluster import ClusterCatalog
from .catalog import Catalog
from .background import Background
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .randoms import RandomCatalog
from .cluster import Cluster
from .depthmap import DepthMap
from .cluster_runner import ClusterRunner

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

class RunRandomsZmask(ClusterRunner):
    """
    The RunRandomsZmask class is derived from ClusterRunner, and will
    compute the masked fraction in the vicinity of a fake cluster.
    """

    def _additional_initialization(self):
        """
        Additional initialization for RunCatalog.
        """

        # This is used to select the masking parameters
        self.runmode = 'percolation'

        if self.config.depthfile is None:
            # Only read galaxies if we don't have a real depth file
            self.read_gals = True
        else:
            self.read_gals = False
        self.read_zreds = False
        self.zreds_required = False
        self.filetype = 'randoms_zmask'

    def run(self, *args, **kwargs):
        """
        Run a catalog through RunCatalog.

        Loop over all clusters and perform RunCatalog computations on each cluster.

        Parameters
        ----------
        cleaninput: `bool`, optional
           Clean seed clusters that are out of the footprint?  Default is False.
        """

        return super(RunRandomsZmask, self).run(*args, **kwargs)

    def _more_setup(self, *args, **kwargs):
        """
        More setup for RunRandomsZmask.
        """

        # Read in the random catalog
        incat = RandomCatalog.from_randfile(self.config.randfile,
                                            nside=self.config.d.nside,
                                            hpix=self.config.d.hpix,
                                            border=self.config.border)

        self.cat = ClusterCatalog.zeros(incat.size,
                                        zredstr=self.zredstr,
                                        config=self.config,
                                        bkg=self.bkg,
                                        cosmo=self.cosmo,
                                        r0=self.r0,
                                        beta=self.beta)

        self.cat.ra = incat.ra
        self.cat.dec = incat.dec
        self.cat.mem_match_id = incat.mem_match_id
        self.cat.z = incat.z
        self.cat.Lambda = incat.Lambda
        self.cat.id_input = incat.id_input

        self.do_lam_plusminus = False
        self.match_centers_to_galaxies = False
        self.do_percolation_masking = False

    def _process_cluster(self, cluster):
        """
        Process a single random cluster with RunRandomsZmask.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to compute mask info.

        Returns
        -------
        fail: `bool`
        """

        # This is a no-op because the generic runner does the work!

        # False means we did not fail.
        return False
