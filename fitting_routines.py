#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:33:06 2021

@author: shlomi
"""
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt


def fit_poly_model_xr(x, y, degree=1, plot='manual', ax=None, color='k',
                      return_just_p=False, logfit=False, fit_label=None):
    import numpy as np
    import matplotlib.pyplot as plt
    p, cov = np.polyfit(x, y, degree, cov=True)
    if ax is None:
        ax = plt.gca()
    if return_just_p:
        return p
    y_model = np.polyval(p, x)
    # Statistics
    n = y.size                                           # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    # used for CI and PI bands
    t = stats.t.ppf(0.975, n - m)

    # Estimates of Error in Data/Model
    resid = y - y_model
    # chi-squared; estimates error in data
    chi2 = np.sum((resid / y_model)**2)
    # reduced chi-squared; measures goodness of fit
    chi2_red = chi2 / dof
    # standard deviation of the error
    s_err = np.sqrt(np.sum(resid**2) / dof)
    # Fit
    if fit_label is None:
        fit_label = 'Fit: {:.2f}'.format(p[0])
        # fit_label = 'Fit: {:.2f} mm/km'.format(p[0] * -1000)
    if logfit:
        ax.plot(x, np.exp(y_model), ls="-", color=color,
                linewidth=1.5, alpha=0.8, label=fit_label)
    else:
        ax.plot(x, y_model, ls="-", color=color,
                linewidth=1.5, alpha=0.8, label=fit_label)

    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.polyval(p, x2)

    # Confidence Interval (select one)
    if plot is not None:
        if plot == 'manual':
            ax = plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax, logfit=logfit, color=color)
        elif plot == 'boot':
            ax = plot_ci_bootstrap(x, y, resid, ax=ax)

    # Prediction Interval
    # pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    # ax.fill_between(x2, y2 + pi, y2 - pi, color="k", linestyle="--")
    # ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    # ax.plot(x2, y2 + pi, "--", color="0.5")
    return p


def plot_ci_manual(t, s_err, n, x, x2, y2, logfit=False, ax=None, color='k'):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x)) **2 / np.sum((x - np.mean(x))**2))
    if logfit:
        ax.fill_between(x2, np.exp(y2 + ci), np.exp(y2 - ci), color="#b9cfe7", edgecolor=None, alpha=0.6)
    else:
        ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor=None, alpha=0.6)

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = np.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, np.polyval(pc, xs), "b-",
                linewidth=2, alpha=3.0 / float(nboot))

    return ax
