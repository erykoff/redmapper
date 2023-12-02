"""Classes to compute cluster centers with different algorithms.

This file contains the generic centering class and various different
implementations of different centering algorithms.  The main algorithm is
CenteringWcenZred, but others are included as part of the centering training
procedure.
"""
import fitsio
import esutil
import numpy as np

from .utilities import gaussFunction
from .utilities import interpol

class Centering(object):
    """
    Generic centering base class for computing cluster centers.
    """

    def __init__(self, cluster, zlambda_corr=None):
        """
        Instantiate a Centering object for a cluster.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to compute centering
        zlambda_corr: `redmapper.ZlambdaCorrectionPar`, optional
           z_lambda correction parameters, if desired.  Default is None.
        """
        # Reference to the cluster; may need to copy
        self.cluster = cluster

        # And the zlambda_corr structure
        self.zlambda_corr = zlambda_corr

        # For convenience, make references to these structures
        self.zredstr = cluster.zredstr
        self.config = cluster.config
        self.cosmo = cluster.cosmo

        # Reset values
        self.ra = np.zeros(self.config.percolation_maxcen) - 400.0
        self.dec = np.zeros(self.config.percolation_maxcen) - 400.0
        self.ngood = 0
        self.index = np.zeros(self.config.percolation_maxcen, dtype=np.int32) - 1
        self.maxind = -1
        self.lnlamlike = -1.0
        self.lnbcglike = -1.0
        self.p_cen = np.zeros(self.config.percolation_maxcen)
        self.q_cen = np.zeros(self.config.percolation_maxcen)
        self.p_fg = np.zeros(self.config.percolation_maxcen)
        self.q_miss = 0.0
        self.p_sat = np.zeros(self.config.percolation_maxcen)
        self.p_c = np.zeros(self.config.percolation_maxcen)

    def find_center(self):
        """
        Stub to override to find center
        """
        return False

class CenteringBCG(Centering):
    """
    Centering class using the brightest cluster galaxy (BCG) algorithm.
    """

    def find_center(self):
        """
        Find the center using the CenteringBCG algorithm.

        This algorithm takes the brightest member with pmem > 0.8 and calls it
        the central galaxy.

        Will set self.maxind (index of best center); self.ra, self.dec
        (position of best center); self.ngood (number of good candidates);
        self.index[:] (indices of all the candidates); self.p_cen[:] (pcen
        centering probabilities); self.q_cen[:] (qcen unused miss
        probabilities); self.p_sat[:] (p_sat satellite probabilities).

        Returns
        -------
        success: `bool`
           True when a center is successfully found. (Always True).
        """
        # This is somewhat arbitrary, and is not yet configurable
        pmem_cut = 0.8

        z_neighbors = self.cluster.neighbors.zred
        z_neighbors_e = self.cluster.neighbors.zred_e
        if self.config.centering_use_zspec:
            has_zspec, = np.where(self.cluster.neighbors.zspec > 0.0)
            z_neighbors[has_zspec] = self.cluster.neighbors.zspec[has_zspec]
            z_neighbors_e[has_zspec] = self.cluster.neighbors.zspec_err[has_zspec]

        use, = np.where((self.cluster.neighbors.r < self.cluster.r_lambda) &
                        ((self.cluster.neighbors.pmem > pmem_cut) |
                         (np.abs(z_neighbors - self.cluster.redshift) < 2.0 * z_neighbors_e)))

        if use.size == 0:
            return False

        mind = np.argmin(self.cluster.neighbors.refmag[use])

        self.maxind = use[mind]
        self.ra = np.array([self.cluster.neighbors.ra[self.maxind]])
        self.dec = np.array([self.cluster.neighbors.dec[self.maxind]])
        self.ngood = 1
        self.index[0] = self.maxind
        self.p_cen[0] = 1.0
        self.q_cen[0] = 1.0
        self.p_sat[0] = 0.0

        return True

class CenteringWcenZred(Centering):
    """
    Centering class using the "wcen-zred" algorithm.

    This algorithm computes the primary centering likelihood algorithm by
    computing the connectivity of the members, as well as ensuring
    consistency between zred of the candidates and the cluster redshift.
    """

    def find_center(self):
        """
        Find the center using the CenteringWcenZred algorithm.

        This algorithm computes the primary centering likelihood algorithm by
        computing the connectivity of the members, as well as ensuring
        consistency between zred of the candidates and the cluster redshift.

        Will set self.maxind (index of best center); self.ra, self.dec
        (position of best center); self.ngood (number of good candidates);
        self.index[:] (indices of all the candidates); self.p_cen[:] (pcen
        centering probabilities); self.q_cen[:] (qcen unused miss
        probabilities); self.p_sat[:] (p_sat satellite probabilities).

        Returns
        -------
        success: `bool`
           True when a center is successfully found. (Always True).
        """

        z_neighbors = self.cluster.neighbors.zred
        z_neighbors_e = self.cluster.neighbors.zred_e
        chisq_neighbors = self.cluster.neighbors.zred_chisq
        if self.config.centering_use_zspec:
            has_zspec, = np.where(self.cluster.neighbors.zspec > 0.0)
            z_neighbors[has_zspec] = self.cluster.neighbors.zspec[has_zspec]
            z_neighbors_e[has_zspec] = self.cluster.neighbors.zspec_err[has_zspec]
            chisq_neighbors[has_zspec] = 1.0

        # These are the galaxies considered as candidate centers
        use, = np.where((self.cluster.neighbors.r < self.cluster.r_lambda) &
                        (self.cluster.neighbors.pfree >= self.config.percolation_pbcg_cut) &
                        (chisq_neighbors < self.config.wcen_zred_chisq_max) &
                        ((self.cluster.neighbors.pmem > 0.0) |
                         (np.abs(self.cluster.redshift - z_neighbors) < 5.0 * z_neighbors_e)))

        # Do the phi_cen filter
        mbar = self.cluster.mstar + self.config.wcen_Delta0 + self.config.wcen_Delta1 * np.log(self.cluster.Lambda / self.config.wcen_pivot)
        phi_cen = gaussFunction(self.cluster.neighbors.refmag[use],
                                1. / (np.sqrt(2. * np.pi) * self.config.wcen_sigma_m),
                                mbar,
                                self.config.wcen_sigma_m)

        if self.zlambda_corr is not None:
            zrmod = interpol(self.zlambda_corr.zred_uncorr, self.zlambda_corr.z, self.cluster.redshift)
            gz = gaussFunction(z_neighbors[use],
                               1. / (np.sqrt(2. * np.pi) * z_neighbors_e[use]),
                               zrmod,
                               z_neighbors_e[use])
        else:
            gz = gaussFunction(z_neighbors[use],
                               1. / (np.sqrt(2. * np.pi) * z_neighbors_e[use]),
                               self.cluster.redshift,
                               z_neighbors_e[use])

        # and the w filter.  We need w for each galaxy that is considered a candidate center.
        # Note that in order to calculate w we need to know all the galaxies that are
        # around it, but only within r_lambda *of that galaxy*.  This is tricky.

        u, = np.where(self.cluster.neighbors.p > 0.0)

        # This is the maximum radius in units of degrees (r_lambda is Mpc; mpc_scale is Mpc / degree)
        maxrad = 1.1 * self.cluster.r_lambda / self.cluster.mpc_scale

        htm_matcher = esutil.htm.Matcher(self.cluster.neighbors.depth,
                                         self.cluster.neighbors.ra[use],
                                         self.cluster.neighbors.dec[use])
        i2, i1, dist = htm_matcher.match(self.cluster.neighbors.ra[u],
                                         self.cluster.neighbors.dec[u],
                                         maxrad, maxmatch=0)

        subdifferent, = np.where(~(use[i1] == u[i2]))
        i1 = i1[subdifferent]
        i2 = i2[subdifferent]
        pdis = dist[subdifferent] * self.cluster.mpc_scale
        pdis = np.sqrt(pdis**2. + self.config.wcen_rsoft**2.)

        lum = 10.**((self.cluster.mstar - self.cluster.neighbors.refmag) / (2.5))

        # Put a floor on w when we have a strange candidate at the edge that doesn't
        # match any good galaxies
        w = np.zeros(use.size) + 1e-3
        for i in range(use.size):
            # need to filter on r_lambda...
            subgal, = np.where(i1 == i)
            if subgal.size > 0:
                inside, = np.where(pdis[subgal] < self.cluster.r_lambda)
                if inside.size > 0:
                    indices = u[i2[subgal[inside]]]
                    if self.config.wcen_uselum:
                        w[i] = np.log(np.sum(self.cluster.neighbors.p[indices] * lum[indices] /
                                             pdis[subgal[inside]]) /
                                      ((1. / self.cluster.r_lambda) *
                                       np.sum(self.cluster.neighbors.p[indices] * lum[indices])))
                    else:
                        w[i] = np.log(np.sum(self.cluster.neighbors.p[indices] /
                                             pdis[subgal[inside]]) /
                                      ((1. / self.cluster.r_lambda) *
                                       np.sum(self.cluster.neighbors.p[indices])))

        sigscale = np.sqrt((np.clip(self.cluster.Lambda, None, self.config.wcen_maxlambda) / self.cluster.scaleval) / self.config.wcen_pivot)

        # scale with richness for Poisson errors
        sig = self.config.lnw_cen_sigma / sigscale

        fw = gaussFunction(np.log(w),
                           1. / (np.sqrt(2. * np.pi) * sig),
                           self.config.lnw_cen_mean,
                           sig)

        ucen = phi_cen * gz * fw

        lo, = np.where(ucen < 1e-10)
        ucen[lo] = 0.0

        # and the satellite function
        maxmag = self.cluster.mstar - 2.5 * np.log10(self.config.lval_reference)
        phi_sat = self.cluster._calc_luminosity(maxmag, idx=use)

        satsig = self.config.lnw_sat_sigma / sigscale
        fsat = gaussFunction(np.log(w),
                             1. / (np.sqrt(2. * np.pi) * satsig),
                             self.config.lnw_sat_mean,
                             satsig)

        usat = phi_sat * gz * fsat

        lo, = np.where(usat < 1e-10)
        usat[lo] = 0.0

        # and the background/foreground
        fgsig = self.config.lnw_fg_sigma / sigscale
        ffg = gaussFunction(np.log(w),
                            1. / (np.sqrt(2. * np.pi) * fgsig),
                            self.config.lnw_fg_mean,
                            fgsig)

        # we want to divide out the r, and we don't want small r's messing this up
        rtest = np.zeros(use.size) + 0.1

        bcounts = ffg * (self.cluster.calc_zred_bkg_density(rtest,
                                                            z_neighbors[use],
                                                            self.cluster.neighbors.refmag[use]) /
                         (2. * np.pi * rtest)) * np.pi * self.cluster.r_lambda**2.

        # The start of Pcen
        Pcen_basic = np.clip(self.cluster.neighbors.pfree[use] * (ucen / (ucen + (self.cluster.Lambda / self.cluster.scaleval - 1.0) * usat + bcounts)),None, 0.99999)

        # make sure we don't have any bad values
        bad, = np.where(~np.isfinite(Pcen_basic))
        Pcen_basic[bad] = 0.0

        okay, = np.where(Pcen_basic > 0.0)
        if okay.size == 0:
            # There are literally NO centers
            self.q_miss = 1.0

            # Set the same as the input galaxy...
            # We need this to be an array of length 1
            good = np.atleast_1d(np.argmin(self.cluster.neighbors.r[use]))

            maxind = use[good[0]]

            Pcen = np.zeros(use.size)
            Qcen = np.zeros(use.size)

        else:
            # Do the renormalization

            Pcen_unnorm = np.zeros(use.size)

            # Only consider centrals that have a non-zero probability
            ok, = np.where(Pcen_basic > 0)

            st = np.argsort(Pcen_basic[ok])[::-1]
            if st.size < self.config.percolation_maxcen:
                good = ok[st]
            else:
                good = ok[st[0: self.config.percolation_maxcen]]

            self.ngood = good.size

            for i in range(self.ngood):
                Pcen0 = Pcen_basic[good[i]]
                Pcen_basic[good[i]] = 0.0
                Pcen_unnorm[good[i]] = Pcen0 * np.prod(1.0 - Pcen_basic[good])
                Pcen_basic[good[i]] = Pcen0

            Qmiss = np.prod(1.0 - Pcen_basic[good])

            KQ = 1./(Qmiss + np.sum(Pcen_unnorm))
            KP = 1./np.sum(Pcen_unnorm)

            Pcen = KP * Pcen_unnorm
            Qcen = KQ * Pcen_unnorm

            mod1 = np.sum(np.log(ucen[good] + (self.cluster.Lambda - 1) * usat[good] + bcounts[good]))
            mod2 = np.sum(np.log(self.cluster.Lambda * usat[good] + bcounts[good]))

            # A new statistic that doesn't quite work
            Qmiss = -2.0 * np.sum(np.log((ucen[good] + (self.cluster.Lambda - 1) * usat[good] + bcounts[good]) / (self.cluster.Lambda * usat[good] + bcounts[good])))

            maxind = use[good[0]]

        Pfg_basic = bcounts[good] / ((self.cluster.Lambda - 1.0) * usat[good] + bcounts[good])
        inf, = np.where(~np.isfinite(Pfg_basic))
        Pfg_basic[inf] = 0.0

        Pfg = (1.0 - Pcen[good]) * Pfg_basic

        Psat_basic = (self.cluster.Lambda - 1.0) * usat[good] / ((self.cluster.Lambda - 1.0) * usat[good] + bcounts[good])
        inf, = np.where(~np.isfinite(Psat_basic))
        Psat_basic[inf] = 0.0

        Psat = (1.0 - Pcen[good]) * Psat_basic

        self.ra[0: good.size] = self.cluster.neighbors.ra[use[good]]
        self.dec[0: good.size] = self.cluster.neighbors.dec[use[good]]
        self.maxind = use[good[0]]
        self.index[0: good.size] = use[good]
        self.p_cen[0: good.size] = Pcen[good]
        self.q_cen[0: good.size] = Qcen[good]
        self.p_fg[0: good.size] = Pfg
        self.p_sat[0: good.size] = Psat
        self.p_c[0: good.size] = Pcen_basic[good]

        return True

class CenteringRandom(Centering):
    """
    Centering class using the random-position algorithm.

    This is used for filter calibration.
    """

    def find_center(self):
        """
        Find the center using the CenteringRandom algorithm.

        This algorithm takes a random position within cluster r_lambda and
        calls it the center.  It is not a very good centering algorithm.

        Will set self.maxind (index of best center); self.ra, self.dec
        (position of best center); self.ngood (number of good candidates);
        self.index[:] (indices of all the candidates); self.p_cen[:] (pcen
        centering probabilities); self.q_cen[:] (qcen unused miss
        probabilities); self.p_sat[:] (p_sat satellite probabilities).

        Returns
        -------
        success: `bool`
           True when a center is successfully found. (Always True).
        """
        r = self.cluster.r_lambda * np.sqrt(np.random.random(size=1))
        phi = 2. * np.pi * np.random.random(size=1)

        x = r * np.cos(phi) / (self.cluster.mpc_scale)
        y = r * np.sin(phi) / (self.cluster.mpc_scale)

        ra_cen = self.cluster.ra + x / np.cos(np.radians(self.cluster.dec))
        dec_cen = self.cluster.dec + y

        self.ra[0] = ra_cen[0]
        self.dec[0] = dec_cen[0]
        self.ngood = 1
        self.index[0] = -1
        self.maxind = -1
        self.p_cen[0] = 1.0
        self.q_cen[0] = 1.0
        self.p_sat[0] = 0.0
        self.p_fg[0] = 0.0
        self.p_c[0] = 1.0

        return True


class CenteringRandomSatellite(Centering):
    """
    Centering class using the random-satellite algorithm.

    This is used for filter calibration.
    """

    def find_center(self):
        """
        Find the center using the CenteringRandomSatellite algorithm.

        This algorithm takes a random member (weighted by member pmem) and
        calls it the center.  It is not a very good centering algorithm (but
        better than pure random!)

        Will set self.maxind (index of best center); self.ra, self.dec
        (position of best center); self.ngood (number of good candidates);
        self.index[:] (indices of all the candidates); self.p_cen[:] (pcen
        centering probabilities); self.q_cen[:] (qcen unused miss
        probabilities); self.p_sat[:] (p_sat satellite probabilities).

        Returns
        -------
        success: `bool`
           True when a center is successfully found. (Always True).
        """
        st = np.argsort(self.cluster.neighbors.pmem)[::-1]

        pdf = self.cluster.neighbors.pmem[st]
        pdf /= np.sum(pdf)
        cdf = np.cumsum(pdf, dtype=np.float64)
        cdfi = (cdf * st.size).astype(np.int32)

        rand = (np.random.uniform(size=1) * st.size).astype(np.int32)
        ind = np.where(cdfi >= rand[0])[0][0]
        maxind = st[ind]

        ra_cen = self.cluster.neighbors.ra[maxind]
        dec_cen = self.cluster.neighbors.dec[maxind]

        self.ra[0] = ra_cen
        self.dec[0] = dec_cen
        self.index[0] = maxind
        self.maxind = maxind
        self.ngood = 1
        self.p_cen[0] = 1.0
        self.q_cen[0] = 1.0
        self.p_sat[0] = 0.0
        self.p_fg[0] = 0.0
        self.p_c[0] = 1.0

        return True

