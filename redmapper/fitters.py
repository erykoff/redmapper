from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
from scipy import special
import scipy.optimize

from .utilities import CubicSpline, interpol

class MedZFitter(object):
    """
    """
    def __init__(self, z_nodes, redshifts, values):
        self._z_nodes = z_nodes
        self._redshifts = redshifts
        self._values = values

    def fit(self, p0):
        """
        """
        # add bounds if we need it...

        pars = scipy.optimize.fmin(self, p0, disp=False)

        return pars

    def __call__(self, pars):
        """
        """
        spl = CubicSpline(self._z_nodes, pars)
        m = spl(self._redshifts)

        absdev = np.abs(self._values - m)
        t = np.sum(absdev.astype(np.float64))
        return t

class RedSequenceFitter(object):
    """
    """
    def __init__(self, mean_nodes,
                 redshifts, colors, errs, dmags=None, trunc=None,
                 slope_nodes=None, scatter_nodes=None, lupcorrs=None,
                 probs=None, bkgs=None):
        self._mean_nodes = np.atleast_1d(mean_nodes)
        if slope_nodes is None:
            self._slope_nodes = self._mean_nodes
        else:
            self._slope_nodes = np.atleast_1d(slope_nodes)
        if scatter_nodes is None:
            self._scatter_nodes = self._mean_nodes
        else:
            self._scatter_nodes = np.atleast_1d(scatter_nodes)

        self._redshifts = np.atleast_1d(redshifts)
        self._colors = np.atleast_1d(colors)
        self._err2s = np.atleast_1d(errs)**2.

        self._n_mean_nodes = self._mean_nodes.size
        self._n_slope_nodes = self._slope_nodes.size
        self._n_scatter_nodes = self._scatter_nodes.size

        if self._redshifts.size != self._colors.size:
            raise ValueError("Number of redshifts must be equal to colors")
        if self._redshifts.size != self._err2s.size:
            raise ValueError("Number of redshifts must be equal to errs")

        if trunc is not None:
            self._trunc = np.atleast_2d(trunc)
            if len(self._trunc.shape) != 2:
                raise ValueError("trunc must be a 2xn_redshifts array")
            if self._trunc.shape[0] != 2 or self._trunc.shape[1] != self._redshifts.size:
                raise ValueError("trunc must be a 2xn_redshifts array")
        else:
            self._trunc = None

        if dmags is not None:
            self._dmags = np.atleast_1d(dmags)
            if self._redshifts.size != self._dmags.size:
                raise ValueError("Number of redshifts must be equal to dmags")
            self._has_dmags = True
        else:
            self._dmags = np.zeros(self._redshifts.size)
            self._has_dmags = False

        if lupcorrs is not None:
            self._lupcorrs = np.atleast_1d(lupcorrs)
            if self._redshifts.size != self._lupcorrs.size:
                raise ValueError("Number of redshifts must be equal to lupcorrs")
            self._has_lupcorrs = True
        else:
            self._lupcorrs = np.zeros(self._redshifts.size)
            self._has_lupcorrs = False

        if probs is not None:
            self._probs = np.atleast_1d(probs)
            if self._redshifts.size != self._probs.size:
                raise ValueError("Number of redshifts must be equal to probs")
            self._has_probs = True
        else:
            self._has_probs = False

        if bkgs is not None:
            self._bkgs = np.atleast_1d(bkgs)
            if self._redshifts.size != self._bkgs.size:
                raise ValueError("Number of redshifts must be equal to bkgs")
            self._has_bkgs = True
        else:
            self._has_bkgs = False

        if self._has_probs and not self._has_bkgs:
            raise ValueError("If you supply probs you must also supply bkgs")

    def fit(self, p0_mean, p0_slope, p0_scatter,
            fit_mean=False, fit_slope=False, fit_scatter=False):
        """
        """
        self._fit_mean = fit_mean
        self._fit_slope = fit_slope
        self._fit_scatter = fit_scatter

        if not self._has_dmags and self._fit_slope:
            raise ValueError("Can only do fit_slope if dmags were supplied")

        ctr = 0
        p0 = np.array([])
        if self._fit_mean:
            self._mean_index = 0
            ctr += self._n_mean_nodes
            p0 = np.append(p0, p0_mean)
        if self._fit_slope:
            self._slope_index = ctr
            ctr += self._n_slope_nodes
            p0 = np.append(p0, p0_slope)
        if self._fit_scatter:
            self._scatter_index = ctr
            ctr += self._n_scatter_nodes
            p0 = np.append(p0, p0_scatter)

        if ctr == 0:
            raise ValueError("Must select at least one of fit_mean, fit_slope, fit_scatter")

        # Precompute...
        if not self._fit_mean:
            spl = CubicSpline(self._mean_nodes, p0_mean)
            self._gmean = spl(self._redshifts)
        if not self._fit_slope:
            spl = CubicSpline(self._slope_nodes, p0_slope)
            self._gslope = spl(self._redshifts)
        if not self._fit_scatter:
            spl = CubicSpline(self._scatter_nodes, p0_scatter)
            self._gsig = np.sqrt(np.clip(spl(self._redshifts), 0.001, None)**2. + self._err2s)

        if not self._fit_mean and not self._fit_scatter and self._trunc is not None:
            self._phi_b = 0.5 * (1. + special.erf((self._trunc[1, :] - self._gmean) / self._gsig))
            self._phi_a = 0.5 * (1. + special.erf((self._trunc[0, :] - self._gmean) / self._gsig))

        pars = scipy.optimize.fmin(self, p0, disp=False)

        retval = []
        if self._fit_mean:
            retval.append(pars[self._mean_index: self._mean_index + self._n_mean_nodes])
        if self._fit_slope:
            retval.append(pars[self._slope_index: self._slope_index + self._n_slope_nodes])
        if self._fit_scatter:
            retval.append(pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes])

        return retval

    def __call__(self, pars):
        """
        """
        if self._fit_mean:
            # We are fitting the mean
            spl = CubicSpline(self._mean_nodes, pars[self._mean_index: self._mean_index + self._n_mean_nodes])
            gmean = spl(self._redshifts)
        else:
            gmean = self._gmean

        if self._fit_slope:
            # We are fitting the slope
            spl = CubicSpline(self._slope_nodes, pars[self._slope_index: self._slope_index + self._n_slope_nodes])
            gslope = spl(self._redshifts)
        else:
            gslope = self._gslope

        if self._fit_scatter:
            # We are fitting the scatter
            spl = CubicSpline(self._scatter_nodes, pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes])
            self._gsig = np.sqrt(np.clip(spl(self._redshifts), 0.001, None)**2. + self._err2s)

        if (self._fit_mean or self._fit_scatter) and self._trunc is not None:
            phi_b = 0.5 * (1. + special.erf((self._trunc[1, :] - gmean) / self._gsig))
            phi_a = 0.5 * (1. + special.erf((self._trunc[0, :] - gmean) / self._gsig))
        elif self._trunc is not None:
            phi_b = self._phi_b
            phi_a = self._phi_a

        if self._has_dmags:
            model_color = gmean + gslope * self._dmags + self._lupcorrs
        else:
            model_color = gmean

        xi = (self._colors - model_color) / self._gsig

        phi = (1. / self._gsig) * (1. / np.sqrt(2. * np.pi)) * np.exp(-0.5 * xi**2.)

        if (self._trunc is not None):
            gci = phi / (phi_b - phi_a)
        else:
            gci = phi

        if self._has_probs:
            # Use probabilities and bkgs
            vals = np.log(self._probs * gci + (1.0 - self._probs) * self._bkgs)
        else:
            # No probabilities or bkgs
            vals = np.log(gci)

        t = -np.sum(vals)

        return t

class RedSequenceOffDiagonalFitter(object):
    """
    """
    def __init__(self, nodes, redshifts, d1, d2, s1, s2, probs, bkgs):
        self._nodes = np.atleast_1d(nodes)
        self._redshifts = np.atleast_1d(redshifts)
        self._d1 = np.atleast_1d(d1)
        self._d2 = np.atleast_1d(d2)
        self._s1 = np.atleast_1d(s1)
        self._s2 = np.atleast_1d(s2)

        self._c_int = np.zeros((self._redshifts.size, 2, 2))
        # self._full_covmats = np.zeros((self._nodes.size, ncol, ncol))

    def fit(self, p0):
        """
        """

        pars = scipy.optimize.fmin(self, p0, disp=False)

        return pars

    def __call__(self, pars):
        """
        """

        spl = CubicSpline(self._nodes, pars)
        r = np.clip(spl(self_redshifts), -0.9, 0.9)

        metrics = np.zeros((self._redshifts.size, 2, 2))
        self._c_int[:, 0, 1] = r * self._s1 * self._s2
        self._c_int[:, 1, 0] = self._c_int[:, 0, 1]

        #self._full_covmats[:, self._j, self._k] = p * np.sqrt(self._covmats[:, self._j, self._j]) * np.sqrt(self._covmats[:, self._k, self._k])
        #self._full_covmats[:, self._k, self._j] = self._covmats[:, self._j, self._k]

        covmats = self._c_int + self._c_noise

        dets = covmats[:, 0, 0] * covmats[:, 1, 1] - covmats[:, 0, 1] * covmats[:, 1, 0]

        metrics[:, 0, 0] = covmats[:, 1, 1] / dets
        metrics[:, 1, 1] = covmats[:, 0, 0] / dets
        metrics[:, 1, 0] = -covmats[:, 0, 1] / dets
        metrics[:, 0, 1] = -covmats[:, 1, 0] / dets

        exponents = -0.5 * (metrics[:, 0, 0] * self._d1 * self._d1 +
                            (metrics[:, 0, 1] + metrics[:, 1, 0]) * self._d1 * self._d2 +
                            metrics[:, 1, 1] * self._d2 * self._d2)

        gci = (dets**(-0.5) / (2. * np.pi)) * exp(exponents)

        vals = np.log(self._probs * gci + (1. - self._probs ) * self._bkgs)

        bad, = np.where(~np.isfinite(vals))
        vals[bad] = -100

        t=-(np.sum(vals) - np.sum(0.5 * (p / self._covmat_prior)**2.))

        if ~np.isfinite(t):
            t = 1e11
        else:
            wall = False
            if (p.max() > 0.9 or p.min() < -0.9) :
                wall = True

                # Check for negative eigenvalues
                

            if wall:
                t += 10000

class EcgmmFitter(object):
    """
    """
    def __init__(self, y, y_err):
        self._y = y
        self._y_err2 = y_err**2.

    def fit(self, wt0, mu, sigma, bounds=None, offset=0.0):
        """
        """

        p0 = np.concatenate([np.atleast_1d(wt0),
                             np.atleast_1d(mu) + offset,
                             np.atleast_1d(sigma)])

        self._y += offset

        if bounds is None:
            self._bounds = [(0.0, 1.0), # wt0
                            (-1.0 + offset, 1.0 + offset), # mu0
                            (-1.0 + offset, 1.0 + offset), # mu1
                            (0.0, 0.5), # sigma0
                            (0.0, 0.5)] # sigma1
        else:
            self._bounds = bounds

        pars = scipy.optimize.fmin(self, p0, disp=True, xtol=0.0001, ftol=0.0081)  ## FIXME

        wt = np.array([pars[0], 1.0 - pars[0]])
        mu = pars[1:3] - offset
        sigma = pars[3:5]

        # sort so that the red is the *second* one
        st = np.argsort(mu)

        self._y -= offset

        return wt[st], mu[st], sigma[st]

    def __call__(self, pars):
        """
        """
        wt0 = pars[0]
        mu0 = pars[1]
        mu1 = pars[2]
        sigma0 = pars[3]
        sigma1 = pars[4]

        wt1 = 1.0 - wt0

        if (wt0 < self._bounds[0][0] or wt0 > self._bounds[0][1] or
            mu0 < self._bounds[1][0] or mu0 > self._bounds[1][1] or
            mu1 < self._bounds[2][0] or mu1 > self._bounds[2][1] or
            sigma0 < self._bounds[3][0] or sigma0 > self._bounds[3][1] or
            sigma1 < self._bounds[4][0] or sigma1 > self._bounds[4][1]):
            return np.inf

        g = ((wt0 / np.sqrt(2. * np.pi * (sigma0**2. + self._y_err2)) * np.exp(-(self._y - mu0)**2. / (2. * (sigma0**2. + self._y_err2)))) +
             (wt1 / np.sqrt(2. * np.pi * (sigma1**2. + self._y_err2)) * np.exp(-(self._y - mu1)**2. / (2. * (sigma1**2. + self._y_err2)))))

        t = np.sum(np.log(g))

        return -t
