# /usr/bin/python

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from myfuncs.plots.scatter import scatter


def lm(
    x,
    y,
    inferencedata=None,
    hdi_prob=0.95,
    ax=None,
    bandalpha=0.6,
    scatter_kws={},
    scatter_color=None,
    line_color=None,
    xrange=None,
    sample_kwargs={},
    **kwargs,
):
    """Make a custom linear model plot with confidence bands.

    Args:
        x (array like): x values
        y (array like): y values
        trace (pymc3.MultiTrace, optional): GLM trace from PyMC3.
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to current axis.
        bandalpha (float, optional): Opacity level of confidence band.
        scatter_kws (dict, optional): Dictionary of keyword arguments passed onto `scatter`.
        **kwargs: Keyword arguments passed onto plot of regression line.

    Returns:
        tuple
            matplotlib.axis: Axis with the linear model plot.
            arviz.InferenceData
            pandas.DataFrame: The linear model pymc3.summary
    """
    if ax is None:
        ax = plt.gca()

    # Determine color (this is necessary so that the scatter and the line have the same color)
    if scatter_color is None and line_color is None:
        next_color = next(ax._get_lines.prop_cycler)["color"]
        scatter_color = next_color
        line_color = next_color
    elif scatter_color is None:
        scatter_color = line_color
    elif line_color is None:
        line_color = scatter_color

    # Scatter
    ax = scatter(x, y, color=scatter_color, ax=ax, **scatter_kws)

    # Run GLM in PyMC3
    if inferencedata is None:
        df = pd.DataFrame(dict(x=x, y=y))
        with pm.Model() as glm:
            pm.GLM.from_formula("y ~ x", data=df)
            inferencedata = pm.sample(return_inferencedata=True, **sample_kwargs)

    summary = az.summary(inferencedata, hdi_prob=hdi_prob)

    # Plot MAP regression line
    if xrange is None:
        xs = np.linspace(np.min(x), np.max(x), 100)
    else:
        xs = np.linspace(*xrange, 100)
    intercept = summary.loc["Intercept", "mean"]
    beta = summary.loc["x", "mean"]
    ax.plot(xs, intercept + beta * xs, color=line_color, zorder=0, **kwargs)

    # Plot posterior predictive credible region band
    intercept_samples = inferencedata.posterior["Intercept"].data.ravel()
    beta_samples = inferencedata.posterior["x"].data.ravel()
    ypred = intercept_samples + beta_samples * xs[:, None]
    ypred_lower = np.quantile(ypred, (1 - hdi_prob) / 2, axis=1)
    ypred_upper = np.quantile(ypred, 1 - (1 - hdi_prob) / 2, axis=1)
    ax.fill_between(
        xs,
        ypred_lower,
        ypred_upper,
        color=line_color,
        zorder=1,
        alpha=bandalpha,
        linewidth=0,
    )

    return ax, inferencedata, summary