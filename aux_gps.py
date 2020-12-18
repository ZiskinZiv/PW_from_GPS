#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:33:19 2019

@author: ziskin
"""
from PW_paths import work_yuval
# TODO: build curve fit tool with various function model: power, sum of sin ...
# TODO: no need to build it, use lmfit instead:
# TODO: check if lmfit accepts- datetimeindex, xarrays and NaNs.
# TODO: if not, build func to replace datetimeindex to numbers and vise versa


def replace_xarray_time_series_with_its_group(da, grp='month', time_dim='time'):
    """run the same func on each dim in da"""
    import xarray as xr
    dims = [x for x in da.dims if time_dim not in x]
    if len(dims) == 0:
        # no other dim except time:
        da = replace_time_series_with_its_group(da, grp=grp)
        return da
    dims_attrs = [da[x].attrs for x in dims]
    dims_attrs_dict = dict(zip(dims, dims_attrs))
    if len(dims) == 1:
        dim0_list = []
        for dim0 in da[dims[0]]:
            da0 = da.sel({dims[0]: dim0})
            da0 = replace_time_series_with_its_group(da0, grp=grp)
            dim0_list.append(da0)
        da_transformed = xr.concat(dim0_list, dims[0])
        da_transformed[dims[0]] = da[dims[0]]
        da_transformed.attrs[dims[0]] = dims_attrs_dict.get(dims[0])
    elif len(dims) == 2:
        dim0_list = []
        for dim0 in da[dims[0]]:
            dim1_list = []
            for dim1 in da[dims[1]]:
                da0 = da.sel({dims[0]: dim0, dims[1]: dim1})
                da0 = replace_time_series_with_its_group(da0, grp=grp)
                dim1_list.append(da0)
            dim0_list.append(xr.concat(dim1_list, dims[1]))
        da_transformed = xr.concat(dim0_list, dims[0])
        da_transformed[dims[0]] = da[dims[0]]
        da_transformed[dims[1]] = da[dims[1]]
        da_transformed.attrs[dims[0]] = dims_attrs_dict.get(dims[0])
        da_transformed.attrs[dims[1]] = dims_attrs_dict.get(dims[1])
    elif len(dims) == 3:
        dim0_list = []
        for dim0 in da[dims[0]]:
            dim1_list = []
            for dim1 in da[dims[1]]:
                dim2_list = []
                for dim2 in da[dims[2]]:
                    da0 = da.sel({dims[0]: dim0, dims[1]: dim1, dims[2]: dim2})
                    da0 = replace_time_series_with_its_group(da0, grp=grp)
                    dim2_list.append(da0)
                dim1_list.append(xr.concat(dim2_list, dims[2]))
            dim0_list.append(xr.concat(dim1_list, dims[1]))
        da_transformed = xr.concat(dim0_list, dims[0])
        da_transformed[dims[0]] = da[dims[0]]
        da_transformed[dims[1]] = da[dims[1]]
        da_transformed[dims[2]] = da[dims[2]]
        da_transformed.attrs[dims[0]] = dims_attrs_dict.get(dims[0])
        da_transformed.attrs[dims[1]] = dims_attrs_dict.get(dims[1])
        da_transformed.attrs[dims[2]] = dims_attrs_dict.get(dims[2])
    return da_transformed


def replace_time_series_with_its_group(da_ts, grp='month'):
    """ replace an xarray time series with its mean grouping e.g., time.month, 
    time.dayofyear, time.hour etc.., basiaclly implement .transform method 
    on 1D dataarray, index must be datetime"""
    import xarray as xr
    import pandas as pd
    da_ts = da_ts.reset_coords(drop=True)
    attrs = da_ts.attrs
    df = da_ts.to_dataframe(da_ts.name)
    if grp == 'month':
        grp_ind = df.index.month
    df = df.groupby(grp_ind).transform('mean')
    ds = df.to_xarray()
    da = ds[da_ts.name]
    da.attrs = attrs
    return da


def calculate_gradient(f, lat_dim='latitude', lon_dim='longitude',
                       level_dim='level', time_dim='time', savepath=None):
    from metpy.calc import lat_lon_grid_deltas
    from metpy.calc import gradient
    from aux_gps import save_ncfile
    import xarray as xr
    name = f.name
    dx, dy = lat_lon_grid_deltas(f[lon_dim], f[lat_dim])
#    f = f.transpose(..., lat_dim, lon_dim)
#    fy, fx = gradient(f, deltas=(dy, dx))
    if level_dim in f.dims and time_dim in f.dims:
        min_year = f[time_dim].dt.year.min().item()
        max_year = f[time_dim].dt.year.max().item()
        level_cnt = f[level_dim].size
        label = '{}_{}-{}.nc'.format(level_cnt, min_year, max_year)
        times = []
        for time in f[time_dim]:
            print('{}-{}'.format(time[time_dim].dt.month.item(), time[time_dim].dt.year.item()))
            levels = []
            for level in f[level_dim]:
                ftl = f.sel({time_dim: time, level_dim: level})
                fy, fx = gradient(ftl, deltas=(dy, dx))
                fx_da = xr.DataArray(fx.magnitude, dims=[lat_dim, lon_dim])
                fx_da.name = '{}x'.format(name)
                fy_da = xr.DataArray(fy.magnitude, dims=[lat_dim, lon_dim])
                fy_da.name = '{}y'.format(name)
                fx_da.attrs['units'] = fx.units.format_babel()
                fy_da.attrs['units'] = fy.units.format_babel()
                grad = xr.merge([fx_da, fy_da])
                levels.append(grad)
            times.append(xr.concat(levels, level_dim))
        ds = xr.concat(times, time_dim)
        ds[level_dim] = f[level_dim]
        ds[time_dim] = f[time_dim]
        ds[lat_dim] = f[lat_dim]
        ds[lon_dim] = f[lon_dim]
    else:
        if level_dim in f.dims:
            level_cnt = f[level_dim].size
            label = '{}.nc'.format(level_cnt)
            levels = []
            for level in f[level_dim]:
                fl = f.sel({level_dim: level})
                fy, fx = gradient(fl, deltas=(dy, dx))
                fx_da = xr.DataArray(fx.magnitude, dims=[lat_dim, lon_dim])
                fx_da.name = '{}x'.format(name)
                fy_da = xr.DataArray(fy.magnitude, dims=[lat_dim, lon_dim])
                fy_da.name = '{}y'.format(name)
                fx_da.attrs['units'] = fx.units.format_babel()
                fy_da.attrs['units'] = fy.units.format_babel()
                grad = xr.merge([fx_da, fy_da])
                levels.append(grad)
            da = xr.concat(levels, level_dim)
            da[level_dim] = f[level_dim]
        elif time_dim in f.dims:
            min_year = f[time_dim].dt.year.min().item()
            max_year = f[time_dim].dt.year.max().item()
            min_year = f[time_dim].dt.year.min().item()
            max_year = f[time_dim].dt.year.max().item()
            times = []
            for time in f[time_dim]:
                ft = f.sel({time_dim: time})
                fy, fx = gradient(ft, deltas=(dy, dx))
                fx_da = xr.DataArray(fx.magnitude, dims=[lat_dim, lon_dim])
                fx_da.name = '{}x'.format(name)
                fy_da = xr.DataArray(fy.magnitude, dims=[lat_dim, lon_dim])
                fy_da.name = '{}y'.format(name)
                fx_da.attrs['units'] = fx.units.format_babel()
                fy_da.attrs['units'] = fy.units.format_babel()
                grad = xr.merge([fx_da, fy_da])
                times.append(grad)
            ds = xr.concat(times, time_dim)
            ds[time_dim] = f[time_dim]
        ds[lat_dim] = f[lat_dim]
        ds[lon_dim] = f[lon_dim]
    if savepath is not None:
        filename = '{}_grad_{}'.format(f.name, label)
        save_ncfile(ds, savepath, filename)
    return ds


def calculate_divergence(u, v, lat_dim='latitude', lon_dim='longitude',
                         level_dim='level', time_dim='time', savepath=None):
    from metpy.calc import divergence
    from metpy.calc import lat_lon_grid_deltas
    from aux_gps import save_ncfile
    import xarray as xr
    dx, dy = lat_lon_grid_deltas(u[lon_dim], u[lat_dim])
    u = u.transpose(..., lat_dim, lon_dim)
    v = v.transpose(..., lat_dim, lon_dim)
    if level_dim in u.dims and time_dim in u.dims:
        min_year = u[time_dim].dt.year.min().item()
        max_year = u[time_dim].dt.year.max().item()
        level_cnt = u[level_dim].size
        label = '{}_{}-{}.nc'.format(level_cnt, min_year, max_year)
        times = []
        for time in u[time_dim]:
            print('{}-{}'.format(time[time_dim].dt.month.item(), time[time_dim].dt.year.item()))
            levels = []
            for level in u[level_dim]:
                utl = u.sel({time_dim: time, level_dim: level})
                vtl = v.sel({time_dim: time, level_dim: level})
                div = divergence(utl, vtl, dx=dx, dy=dy)
                div_da = xr.DataArray(div.magnitude, dims=[lat_dim, lon_dim])
                div_da.attrs['units'] = div.units.format_babel()
                levels.append(div_da)
            times.append(xr.concat(levels, level_dim))
        da = xr.concat(times, time_dim)
        da[level_dim] = u[level_dim]
        da[time_dim] = u[time_dim]
        da[lat_dim] = u[lat_dim]
        da[lon_dim] = u[lon_dim]
        da.name = '{}{}_div'.format(u.name, v.name)
    else:
        if level_dim in u.dims:
            level_cnt = u[level_dim].size
            label = '{}.nc'.format(level_cnt)
            levels = []
            for level in u[level_dim]:
                ul = u.sel({level_dim: level})
                vl = v.sel({level_dim: level})
                div = divergence(ul, vl, dx=dx, dy=dy)
                div_da = xr.DataArray(div.magnitude, dims=[lat_dim, lon_dim])
                div_da.attrs['units'] = div.units.format_babel()
                levels.append(div_da)
            da = xr.concat(levels, level_dim)
            da[level_dim] = u[level_dim]
        elif time_dim in u.dims:
            min_year = u[time_dim].dt.year.min().item()
            max_year = u[time_dim].dt.year.max().item()
            min_year = u[time_dim].dt.year.min().item()
            max_year = u[time_dim].dt.year.max().item()
            times = []
            for time in u[time_dim]:
                ut = u.sel({time_dim: time})
                vt = v.sel({time_dim: time})
                div = divergence(ut, vt, dx=dx, dy=dy)
                div_da = xr.DataArray(div.magnitude, dims=[lat_dim, lon_dim])
                div_da.attrs['units'] = div.units.format_babel()
                times.append(div_da)
            da = xr.concat(times, time_dim)
            da[time_dim] = u[time_dim]
        da[lat_dim] = u[lat_dim]
        da[lon_dim] = u[lon_dim]
        da.name = '{}{}_div'.format(u.name, v.name)
    if savepath is not None:
        filename = '{}{}_div_{}'.format(u.name, v.name, label)
        save_ncfile(da, savepath, filename)
    return da


def calculate_pressure_integral(da, pdim='level'):
    import numpy as np
    # first sort to decending levels:
    da = da.sortby(pdim, ascending=False)
    try:
        units = da[pdim].attrs['units']
    except KeyError:
        print('no units attrs found, assuming units are hPa')
        units = 'hPa'
    # transform to Pa:
    if units != 'Pa':
        print('{} units detected, converting to Pa!'.format(units))
        da[pdim] = da[pdim] * 100
    # P_{i+1} - P_i:
    plevel_diff = np.abs(da[pdim].diff(pdim, label='lower'))
    # var_i + var_{i+1}:
    da_sum = da.shift(level=-1) + da
    p_int = ((da_sum * plevel_diff) / 2.0).sum(pdim)
    return p_int


def linear_fit_using_scipy_da_ts(da_ts, model='TSEN', slope_factor=3650.25,
                                 plot=False, ax=None, units=None,
                                 method='simple', weights=None, not_time=False):
    """linear fit using scipy for dataarray time series,
    support for theilslopes(TSEN) and lingress(LR), produce 95% CI"""
    import xarray as xr
    from scipy.stats.mstats import theilslopes
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import numpy as np
    time_dim = list(set(da_ts.dims))[0]
    y = da_ts.dropna(time_dim).values
    if not_time:
        X = da_ts[time_dim].values.reshape(-1, 1)
        jul_no_nans = da_ts.dropna(time_dim)[time_dim].values
        jul = da_ts[time_dim].values
    else:
        jul, jul_no_nans = get_julian_dates_from_da(da_ts, subtract='median')
        X = jul_no_nans.reshape(-1, 1)
    if model == 'LR':
        if method == 'simple':
            coef, intercept, r_value, p_value, std_err = linregress(jul_no_nans, y)
            confidence_interval = 1.96 * std_err
            coef_lo = coef - confidence_interval
            coef_hi = coef + confidence_interval
        elif method == 'curve_fit':
            func = lambda x, a, b: a * x + b
            if weights is not None:
                sigma = weights.dropna(time_dim).values
            else:
                sigma = None
            best_fit_ab, covar = curve_fit(func, jul_no_nans, y,
                                           sigma = sigma,p0=[0, 0],
                                           absolute_sigma = False)
            sigma_ab = np.sqrt(np.diagonal(covar))
            coef = best_fit_ab[0]
            intercept = best_fit_ab[1]
            coef_lo = coef - sigma_ab[0]
            coef_hi = coef + sigma_ab[0]
    elif model == 'TSEN':
        coef, intercept, coef_lo, coef_hi = theilslopes(y, X)
    predict = jul * coef + intercept
    predict_lo = jul * coef_lo + intercept
    predict_hi = jul * coef_hi + intercept
    trend_hi = xr.DataArray(predict_hi, dims=[time_dim])
    trend_hi.name = 'trend_hi'
    trend_lo = xr.DataArray(predict_lo, dims=[time_dim])
    trend_lo.name = 'trend_lo'
    trend_hi[time_dim] = da_ts[time_dim]
    trend_lo[time_dim] = da_ts[time_dim]
    slope_in_factor_scale_lo = coef_lo * slope_factor
    slope_in_factor_scale_hi = coef_hi * slope_factor
    trend = xr.DataArray(predict, dims=[time_dim])
    trend.name = 'trend'
    trend[time_dim] = da_ts[time_dim]
    slope_in_factor_scale = coef * slope_factor
    if plot:
        labels =  ['{}'.format(da_ts.name)]
        if ax is None:
            fig, ax = plt.subplots()
        origln = da_ts.plot.line('k-', marker='o', ax=ax, linewidth=1.5, markersize=2.5)
        trendln = trend.plot(ax=ax, color='r', linewidth=2)
        trend_hi.plot.line('r--', ax=ax, linewidth=1.5)
        trend_lo.plot.line('r--', ax=ax, linewidth=1.5)
        trend_label = '{} model, slope={:.2f} ({:.2f}, {:.2f}) {}'.format(model, slope_in_factor_scale, slope_in_factor_scale_lo, slope_in_factor_scale_hi, units)
        handles = origln
        handles += trendln
        labels.append(trend_label)
        ax.legend(handles=handles, labels=labels, loc='upper left')
        ax.grid()
    trend_ds = xr.merge([trend, trend_hi, trend_lo])
    results_dict = {'slope_hi': slope_in_factor_scale_hi, 'slope_lo': slope_in_factor_scale_lo, 'slope': slope_in_factor_scale}
    results_dict['intercept'] = intercept
    return trend_ds, results_dict


def split_equal_da_ts_around_datetime(da_ts, dt='2014-05-01'):
    time_dim = list(set(da_ts.dims))[0]
    x1 = da_ts.dropna(time_dim).sel({time_dim: slice(None, dt)})
    x2 = da_ts.dropna(time_dim).sel({time_dim: slice(dt, None)})
    if x1.size == 0 or x2.size == 0:
        raise ValueError('one or two of the sub-series is 0 size.')
    if x1.size > x2.size:
        x1 = x1.isel({time_dim: slice(-x2.size , None)})
    elif x1.size < x2.size:
        x2 = x2.isel({time_dim: slice(0, x1.size)})
    return x1, x2


def wilcoxon_rank_test_xr(
        da_ts, alpha=0.05,
        cp_dt='2014-05-01',
        zero_method='wilcox',
        correction=False,
        alternative='two-sided',
        mode='auto'):
    import xarray as xr
    from scipy.stats import wilcoxon
    x, y = split_equal_da_ts_around_datetime(da_ts, dt=cp_dt)
    stat, pvalue = wilcoxon(x, y, zero_method=zero_method,
                            correction=correction, alternative=alternative
                            )
    if pvalue < alpha:
        # the two parts of the time series come from different distributions
        print('Two distributions!')
        normal = False
    else:
        # same distribution
        print('Same distribution')
        normal = True
    da = xr.DataArray([stat, pvalue, normal], dims=['result'])
    da['result'] = ['stat', 'pvalue', 'h']
    return da


def normality_test_xr(da_ts, sample=None, alpha=0.05, test='lili',
                      dropna=True, verbose=True):
    """normality tests on da_ts"""
    from statsmodels.stats.diagnostic import lilliefors
    from scipy.stats import shapiro
    from scipy.stats import normaltest
    import xarray as xr
    time_dim = list(set(da_ts.dims))[0]
    if sample is not None:
        da_ts = da_ts.resample({time_dim: sample}).mean()
    if dropna:
        da_ts = da_ts.dropna(time_dim)
    if test == 'shapiro':
        stat, pvalue = shapiro(da_ts)
    elif test == 'lili':
        stat, pvalue = lilliefors(da_ts, dist='norm', pvalmethod='table')
    elif test == 'normaltest':
        stat, pvalue = normaltest(da_ts)
    if pvalue < alpha:
        Not = 'NOT'
        normal = False
    else:
        Not = ''
        normal = True
    if verbose:
        print('Mean: {:.4f}, pvalue: {:.4f}'.format(stat, pvalue))
        print('Thus, the data is {} Normally distributed with alpha {}'.format(Not, alpha))
    da = xr.DataArray([stat, pvalue, normal], dims=['result'])
    da['result'] = ['stat', 'pvalue', 'h']
    return da


def homogeneity_test_xr(da_ts, hg_test_func, dropna=True, alpha=0.05,
                        sim=None, verbose=True):
    """False means data is homogenous, True means non-homogenous with significance alpha"""
    import xarray as xr
    import pandas as pd
    time_dim = list(set(da_ts.dims))[0] 
    if dropna:
        da_ts = da_ts.dropna(time_dim)
    h, cp, p, U, mu = hg_test_func(da_ts, alpha=alpha, sim=sim)
    result = hg_test_func(da_ts, alpha=alpha, sim=sim)
    name = type(result).__name__
    if verbose:
        print('running homogeneity {} with alpha {} and sim {}'.format(name, alpha, sim))

    cpl = pd.to_datetime(da_ts.isel({time_dim: result.cp})[time_dim].values)
    if 'U' in result._fields:
        stat = result.U
    elif 'T' in result._fields:
        stat = result.T
    elif 'Q' in result._fields:
        stat = result.Q
    elif 'R' in result._fields:
        stat = result.R
    elif 'V' in result._fields:
        stat = result.V
    da = xr.DataArray([name, result.h, cpl, result.p, stat, result.avg], dims=['results'])
    da['results'] = ['name', 'h', 'cp_dt', 'pvalue', 'stat', 'means']
    return da


def VN_ratio_trend_test_xr(da_ts, dropna=True, alpha=0.05, loadpath=work_yuval,
                        verbose=True, return_just_trend=False):
    """calculate the Von Nuemann ratio test statistic and test for trend."""
    import xarray as xr
    time_dim = list(set(da_ts.dims))[0]
    if dropna:
        da_ts = da_ts.dropna(time_dim)
    n = da_ts.dropna(time_dim).size
    d2 = (da_ts.diff(time_dim)**2.0).sum() / (n - 1)
    # s**2 is the variance:
    s2 = da_ts.var()
    eta = (d2 / s2).item()
    cv_da = xr.load_dataarray(loadpath / 'VN_critical_values.nc')
    cv = cv_da.sel(sample_size=n, pvalue=alpha, method='nearest').item()
    if eta < cv:
        if verbose:
            print('the hypothesis of stationary cannot be rejected at the level {}'.format(alpha))
        trend = True
    else:
        trend = False
    if return_just_trend:
        return trend
    else:
        da = xr.DataArray([eta, cv, trend, n], dims=['results'])
        da['results'] = ['eta', 'cv', 'trend', 'n']
        return da


def reduce_tail_xr(xarray, reduce='mean', time_dim='time', records=120,
                   return_df=False):
    import xarray as xr

    def reduce_tail_da(da, reduce=reduce, time_dim=time_dim, records=records):
        if reduce == 'mean':
            da = da.dropna(time_dim).tail(records).mean(time_dim)
        return da
    if isinstance(xarray, xr.DataArray):
        xarray = reduce_tail_da(xarray, reduce, time_dim, records)
    elif isinstance(xarray, xr.Dataset):
        xarray = xarray.map(reduce_tail_da, args=(reduce, time_dim, records))
        if return_df:
            df = xarray.to_array('dum').to_dataframe(reduce)
            df.index.name = ''
            return df
    return xarray


def decimal_year_to_datetime(decimalyear):
    from datetime import datetime, timedelta
    import pandas as pd
    year = int(decimalyear)
    rem = decimalyear - year
    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
    return pd.to_datetime(result)


def select_months(da_ts, months, remove=False, reindex=True):
    import xarray as xr
    from aux_gps import xr_reindex_with_date_range
    import pandas as pd
    import numpy as np
    time_dim = list(set(da_ts.dims))[0]
    attrs = da_ts.attrs
    try:
        name = da_ts.name
    except AttributeError:
        name = ''
    if remove:
        all_months = np.arange(1, 13)
        months = list(set(all_months).difference(set(months)))
    print('selecting months #{} from {}'.format(', #'.join([str(x) for x in months]), name))
    to_add = []
    for month in months:
        sliced = da_ts.sel({time_dim: da_ts['{}.month'.format(time_dim)] == int(month)})
        to_add.append(sliced)
    da = xr.concat(to_add, time_dim)
    da.attrs = attrs
    if reindex:
        freq = pd.infer_freq(da_ts[time_dim].values)
        da = xr_reindex_with_date_range(da, freq=freq)
    return da


def run_MLR_diurnal_harmonics(harmonic_dss, season=None, n_max=4, plot=True,
                              ax=None, legend_loc=None, ncol=1, legsize=8):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import explained_variance_score
    import matplotlib.pyplot as plt
    import numpy as np
    if n_max > harmonic_dss.cpd.max().values.item():
        n_max = harmonic_dss.cpd.max().values.item()
    try:
        field = harmonic_dss.attrs['field']
        if field == 'PW':
            field = 'PWV'
    except KeyError:
        field = 'no name'
    name = [x for x in harmonic_dss][0].split('_')[0]
    if season is None and 'season' not in harmonic_dss.dims:
        harmonic = harmonic_dss # .sel(season='ALL')
    elif season is None and 'season' in harmonic_dss.dims:
        harmonic = harmonic_dss.sel(season='ALL')
    elif season is not None:
        harmonic = harmonic_dss.sel(season=season)
    # pre-proccess:
    harmonic = harmonic.transpose('hour', 'cpd', ...)
    harmonic = harmonic.sel(cpd=slice(1, n_max))
    # X = harmonic[name + '_mean'].values
    y = harmonic[name].values.reshape(-1, 1)
    exp_list = []
    for cpd in harmonic['cpd'].values:
        X = harmonic[name + '_mean'].sel(cpd=cpd).values.reshape(-1, 1)
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, y)
        y_pred = lr.predict(X)
        ex_var = explained_variance_score(y, y_pred)
        exp_list.append(ex_var)
    explained = np.array(exp_list) * 100.0
    exp_dict = dict(zip([x for x in harmonic['cpd'].values], explained))
    exp_dict['total'] = np.cumsum(explained)
    exp_dict['season'] = season
    exp_dict['name'] = name
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        markers = ['s', 'x', '^', '>', '<', 'X']
        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green',
                  'tab:purple', 'tab:yellow']
        styles = ['-', '--', '-.', ':', 'None', ' ']
        for i, cpd in enumerate(harmonic['cpd'].values):
            harmonic[name + '_mean'].sel(cpd=cpd).plot(ax=ax, linestyle=styles[i], color=colors[i]) # marker=markers[i])
        harmonic[name + '_mean'].sum('cpd').plot(ax=ax, marker=None, color='k', alpha=0.7)
        harmonic[name].plot(ax=ax, marker='o', linewidth=0., color='k', alpha=0.7)
        S = ['S{}'.format(x) for x in harmonic['cpd'].values]
        S_total = ['+'.join(S)]
        S = ['S{} ({:.0f}%)'.format(x, exp_dict[int(x)]) for x in harmonic['cpd'].values]
        ax.legend(
            S + S_total + [field],
            prop={'size': legsize},
            framealpha=0.5,
            fancybox=True,
            loc=legend_loc, ncol=ncol, columnspacing=0.75, handlelength=1.0)
#        ax.grid()
        ax.set_xlabel('Time of day [UTC]')
        # ax.set_ylabel('{} anomalies [mm]'.format(field))
        if season is None:
            ax.set_title('Annual {} diurnal cycle for {} station'.format(field, name.upper()))
        else:
            ax.set_title('{} diurnal cycle for {} station in {}'.format(field, name.upper(), season))
        return ax
    else:
        return exp_dict


def harmonic_analysis_xr(da, n=6, normalize=False, anomalize=False, freq='D',
                         user_field_name=None):
    import xarray as xr
    from aux_gps import fit_da_to_model
    from aux_gps import normalize_xr
    from aux_gps import anomalize_xr
    try:
        field = da.attrs['channel_name']
    except KeyError:
        field = user_field_name
    if field is None:
        field = ''
    if normalize:
        da = normalize_xr(da, norm=1)
    time_dim = list(set(da.dims))[0]
    if anomalize:
        da = anomalize_xr(da, freq=freq)
    seasons = ['JJA', 'SON', 'DJF', 'MAM', 'ALL']
    print('station name: {}'.format(da.name))
    print('performing harmonic analysis with 1 to {} cycles per day.'.format(n))
    season_list = []
    for season in seasons:
        if season != 'ALL':
            print('analysing season {}.'.format(season))
            das = da.sel({time_dim: da['{}.season'.format(time_dim)] == season})
        else:
            print('analysing ALL seasons.')
            das = da
        ds = harmonic_da(das, n=n)
        season_list.append(ds)
    dss = xr.concat(season_list, 'season')
    dss['season'] = seasons
    dss.attrs['field'] = field
    return dss


def harmonic_da(da_ts, n=3, field=None, init=None):
    from aux_gps import fit_da_to_model
    import xarray as xr
    time_dim = list(set(da_ts.dims))[0]
    harmonics = [x + 1 for x in range(n)]
    if init is not None:
        init_amp = da_ts.groupby('{}.hour'.format(time_dim)).mean().mean('hour').values
    else:
        init_amp = 1.0
    init_values = [init_amp/float(x) for x in harmonics]
    params_list = []
    di_mean_list = []
    di_std_list = []
    for cpd, init_val in zip(harmonics, init_values):
        print('fitting harmonic #{}'.format(cpd))
        params = dict(
            sin_freq={
                'value': cpd}, sin_amp={
                'value': init_val}, sin_phase={
                'value': 0})
        res = fit_da_to_model(
            da_ts,
            modelname='sin',
            params=params,
            plot=False,
            verbose=False)
        name = da_ts.name.split('_')[0]
        params_da = xr.DataArray([x for x in res.attrs.values()],
                                  dims=['params', 'val_err'])
        params_da['params'] = [x for x in res.attrs.keys()]
        params_da['val_err'] = ['value', 'stderr']
        params_da.name = name + '_params'
        name = res.name.split('_')[0]
        diurnal_mean = res.groupby('{}.hour'.format(time_dim)).mean()
        diurnal_std = res.groupby('{}.hour'.format(time_dim)).std()
        # diurnal_mean.attrs.update(attrs)
        # diurnal_std.attrs.update(attrs)
        diurnal_mean.name = name + '_mean'
        diurnal_std.name = name + '_std'
        params_list.append(params_da)
        di_mean_list.append(diurnal_mean)
        di_std_list.append(diurnal_std)
    da_mean = xr.concat(di_mean_list, 'cpd')
    da_std = xr.concat(di_std_list, 'cpd')
    da_params = xr.concat(params_list, 'cpd')
    ds = da_mean.to_dataset(name=da_mean.name)
    ds[da_std.name] = da_std
    ds['cpd'] = harmonics
    ds[da_params.name] = da_params
    ds[da_ts.name] = da_ts.groupby('{}.hour'.format(time_dim)).mean()
    if field is not None:
        ds.attrs['field'] = field
    return ds


def anomalize_xr(da_ts, freq='D', time_dim=None, verbose=True):  # i.e., like deseason
    import xarray as xr
    if time_dim is None:
        time_dim = list(set(da_ts.dims))[0]
    attrs = da_ts.attrs
    if isinstance(da_ts, xr.Dataset):
        da_attrs = dict(zip([x for x in da_ts],[da_ts[x].attrs for x in da_ts]))
    try:
        name = da_ts.name
    except AttributeError:
        name = ''
    if isinstance(da_ts, xr.Dataset):
        name = [x for x in da_ts]
    if freq == 'D':
        if verbose:
            print('removing daily means from {}'.format(name))
        frq = 'daily'
        date = groupby_date_xr(da_ts)
        da_anoms = da_ts.groupby(date) - da_ts.groupby(date).mean()
    elif freq == 'H':
        if verbose:
            print('removing hourly means from {}'.format(name))
        frq = 'hourly'
        da_anoms = da_ts.groupby('{}.hour'.format(
            time_dim)) - da_ts.groupby('{}.hour'.format(time_dim)).mean()
    elif freq == 'MS':
        if verbose:
            print('removing monthly means from {}'.format(name))
        frq = 'monthly'
        da_anoms = da_ts.groupby('{}.month'.format(
            time_dim)) - da_ts.groupby('{}.month'.format(time_dim)).mean()
    da_anoms = da_anoms.reset_coords(drop=True)
    da_anoms.attrs.update(attrs)
    da_anoms.attrs.update(action='removed {} means'.format(frq))
    # if dataset, update attrs for each dataarray and add action='removed x means'
    if isinstance(da_ts, xr.Dataset):
        for x in da_ts:
            da_anoms[x].attrs.update(da_attrs.get(x))
            da_anoms[x].attrs.update(action='removed {} means'.format(frq))
    return da_anoms


def line_and_num_for_phrase_in_file(phrase='the dog barked', filename='file.txt'):
    with open(filename, 'r') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                return i, line
    return None, None


def grab_n_consecutive_epochs_from_ts(da_ts, sep='nan', n=10, time_dim=None,
                                      return_largest=False):
    """grabs n consecutive epochs from time series (xarray dataarrays)
    and return list of either dataarrays"""
    if time_dim is None:
        time_dim = list(set(da_ts.dims))[0]
    df = da_ts.to_dataframe()
    A = consecutive_runs(df, num='nan')
    A = A.sort_values('total_not-nan', ascending=False)
    max_n = len(A)
    if return_largest:
        start = A.iloc[0, 0]
        end = A.iloc[0, 1]
        da = da_ts.isel({time_dim:slice(start, end)})
        return da
    if n > max_n:
        print('{} epoches requested but only {} available'.format(n, max_n))
        n = max_n
    da_list = []
    for i in range(n):
        start = A.iloc[i, 0]
        end = A.iloc[i, 1]
        da = da_ts.isel({time_dim: slice(start, end)})
        da_list.append(da)
    return da_list


#def assemble_semi_period(reduced_da_ts):
#    import numpy as np
#    import xarray as xr
#    period = [x for x in reduced_da_ts.dims][0]
#    if period == 'month':
#        plength = reduced_da_ts[period].size
#        mnth_arr = np.arange(1, 13)
#        mnth_splt = np.array_split(mnth_arr, int(12/plength))
#        vals = reduced_da_ts.values
#        vals_list = []
#        vals_list.append(vals)
#        for i in range(len(mnth_splt)-1):
#            vals_list.append(vals)
#        modified_reduced = xr.DataArray(np.concatenate(vals_list), dims=['month'])
#        modified_reduced['month'] = mnth_arr
#        return modified_reduced
#    elif period == 'hour':
#        plength = reduced_da_ts[period].size
#        hr_arr = np.arange(0, 24)
#        hr_splt = np.array_split(hr_arr, int(24/plength))
#        vals = reduced_da_ts.values
#        vals_list = []
#        vals_list.append(vals)
#        for i in range(len(hr_splt)-1):
#            vals_list.append(vals)
#        modified_reduced = xr.DataArray(np.concatenate(vals_list), dims=['hour'])
#        modified_reduced['hour'] = hr_arr
#        return modified_reduced
#
#
#def groupby_semi_period(da_ts, period='6M'):
#    """return an xarray DataArray with the semi period of 1 to 11 months or
#    1 to 23 hours.
#    Input: period : string, first char is period length, second is frequency.
#    for now support is M for month and H for hour."""
#    import numpy as np
#    df = da_ts.to_dataframe()
#    plength = [x for x in period if x.isdigit()]
#    if len(plength) == 1:
#        plength = int(plength[0])
#    elif len(plength) == 2:
#        plength = int(''.join(plength))
#    freq = [x for x in period if x.isalpha()][0]
#    print(plength, freq)
#    if freq == 'M':
#        if np.mod(12, plength) != 0:
#            raise('pls choose integer amounts, e.g., 3M, 4M, 6M...')
#        mnth_arr = np.arange(1, 13)
#        mnth_splt = np.array_split(mnth_arr, int(12 / plength))
#        rpld = {}
#        for i in range(len(mnth_splt) - 1):
#            rpld.update(dict(zip(mnth_splt[i + 1], mnth_splt[0])))
#        df['month'] = df.index.month
#        df['month'] = df['month'].replace(rpld)
#        month = df['month'].to_xarray()
#        return month
#    if freq == 'H':
#        if np.mod(24, plength) != 0:
#            raise('pls choose integer amounts, e.g., 6H, 8H, 12H...')
#        hr_arr = np.arange(0, 24)
#        hr_splt = np.array_split(hr_arr, int(24 / plength))
#        rpld = {}
#        for i in range(len(hr_splt) - 1):
#            rpld.update(dict(zip(hr_splt[i + 1], hr_splt[0])))
#        df['hour'] = df.index.hour
#        df['hour'] = df['hour'].replace(rpld)
#        hour = df['hour'].to_xarray()
#        return hour


def groupby_half_hour_xr(da_ts, reduce='mean'):
    import pandas as pd
    import numpy as np
    df = da_ts.to_dataframe()
    native_freq = pd.infer_freq(df.index)
    if not native_freq:
        raise('Cannot infer frequency...')
    if reduce == 'mean':
        df = df.groupby([df.index.hour, df.index.minute]).mean()
    elif reduce == 'std':
        df = df.groupby([df.index.hour, df.index.minute]).std()
    time = pd.date_range(start='1900-01-01', periods=df.index.size,
                         freq=native_freq)
    df = df.set_index(time)
    df = df.resample('30T').mean()
    half_hours = np.arange(0, 24, 0.5)
    df.index = half_hours
    df.index.name = 'half_hour'
    ds = df.to_xarray()
    return ds


def groupby_date_xr(da_ts):
    df = da_ts.to_dataframe()
    df['date'] = df.index.date
    date = df['date'].to_xarray()
    return date


def loess_curve(da_ts, time_dim='time', season=None, plot=True):
    from skmisc.loess import loess
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    if season is not None:
        da_ts = da_ts.sel({time_dim: da_ts[time_dim + '.season'] == season})
    x = da_ts.dropna(time_dim)[time_dim].values
    y = da_ts.dropna(time_dim).values
    l_obj = loess(x, y)
    l_obj.fit()
    pred = l_obj.predict(x, stderror=True)
    conf = pred.confidence()
    lowess = np.copy(pred.values)
    ll = np.copy(conf.lower)
    ul = np.copy(conf.upper)
    da_lowess = xr.Dataset()
    da_lowess['mean'] = xr.DataArray(lowess, dims=[time_dim])
    da_lowess['upper'] = xr.DataArray(ul, dims=[time_dim])
    da_lowess['lower'] = xr.DataArray(ll, dims=[time_dim])
    da_lowess[time_dim] = x
    if plot:
        plt.plot(x, y, '+')
        plt.plot(x, lowess)
        plt.fill_between(x, ll, ul, alpha=.33)
        plt.show()
    return da_lowess


def autocorr_plot(da_ts, max_lag=40):
    import pandas as pd
    ser = pd.Series(da_ts)
    corrs = [ser.autocorr(lag=x) for x in range(0, max_lag)]
    lags = [x for x in range(0, max_lag)]
    lags_ser = pd.Series(corrs, index=lags)
    ax = lags_ser.plot(kind='bar', rot=0, figsize=(10, 5))
    return ax


def error_mean_rmse(y, y_pred):
    from sklearn.metrics import mean_squared_error
    import numpy as np
    mse = mean_squared_error(y.values, y_pred.values)
    rmse = np.sqrt(mse)
    mean = np.mean(y.values-y_pred.values)
    print('mean : {:.2f}, rmse : {:.2f}'.format(mean, rmse))
    return mean, rmse


def rename_data_vars(ds, suffix='_error', prefix=None, verbose=False):
    import xarray as xr
    if not isinstance(ds, xr.Dataset):
        raise ValueError('input must be an xarray dataset object!')
    vnames = [x for x in ds.data_vars]
#    if remove_suffix:
#        new_names = [x.replace(suffix, '') for x in ds.data_vars]
    if suffix is not None:
        new_names = [str(x) + suffix for x in ds.data_vars]
    if prefix is not None:
        new_names = [prefix + str(x) for x in ds.data_vars]
    name_dict = dict(zip(vnames, new_names))
    ds = ds.rename_vars(name_dict)
    if verbose:
        print('var names were added the suffix {}.'.format(suffix))
    return ds


def remove_duplicate_spaces_in_string(line):
    import re
    line_removed = " ".join(re.split("\s+", line, flags=re.UNICODE))
    return line_removed


def save_ncfile(xarray, savepath, filename='temp.nc', engine=None, dtype=None,
                fillvalue=None):
    import xarray as xr
    print('saving {} to {}'.format(filename, savepath))
    if dtype is None:
        comp = dict(zlib=True, complevel=9, _FillValue=fillvalue)  # best compression
    else:
        comp = dict(zlib=True, complevel=9, dtype=dtype, _FillValue=fillvalue)  # best compression
    if isinstance(xarray, xr.Dataset):
        encoding = {var: comp for var in xarray}
    elif isinstance(xarray, xr.DataArray):
        encoding = {var: comp for var in xarray.to_dataset()}
    xarray.to_netcdf(savepath / filename, 'w', encoding=encoding, engine=engine)
    print('File saved!')
    return


def weighted_long_term_monthly_means_da(da_ts, plot=True):
    """create a long term monthly means(climatology) from a dataarray time
    series with weights of items(mins,days etc..) per each month
    apperently, DataArray.groupby('time.month').mean('time') does exactely
    this... so this function is redundant"""
    import pandas as pd
    name = da_ts.name
    # first save attrs:
    attrs = da_ts.attrs
    try:
        df = da_ts.to_dataframe()
    except ValueError:
        name = 'name'
        df = da_ts.to_dataframe(name=name)
    df = df.dropna()
    df['month'] = df.index.month
    df['year'] = df.index.year
    cnt = df.groupby(['month', 'year']).count()[name].to_frame()
    cnt /= cnt.max()
    weights = pd.pivot_table(cnt, index='year', columns='month')
    dfmm = df.groupby(['month', 'year']).mean()[name].to_frame()
    dfmm = pd.pivot_table(dfmm, index='year', columns='month')
    # wrong:
#     weighted_monthly_means = dfmm * weights 
    # normalize weights:
    wtag = weights / weights.sum(axis=0)
    weighted_clim = (dfmm*wtag).sum(axis=0).unstack().squeeze()
    # convert back to time-series:
#    df_ts = weighted_monthly_means.stack().reset_index()
#    df_ts['dt'] = df_ts.year.astype(str) + '-' + df_ts.month.astype(str)
#    df_ts['dt'] = pd.to_datetime(df_ts['dt'])
#    df_ts = df_ts.set_index('dt')
#    df_ts = df_ts.drop(['year', 'month'], axis=1)
#    df_ts.index.name = 'time'
#    da = df_ts[name].to_xarray()
    da = weighted_clim.to_xarray()
    da.attrs = attrs
#    da = xr_reindex_with_date_range(da, drop=True, freq='MS')
    if plot:
        da.plot()
    return da


def create_monthly_index(dt_da, period=6, unit='month'):
    import numpy as np
    pdict = {6: 'H', 4: 'T', 3: 'Q'}
    dt = dt_da.to_dataframe()
    if unit == 'month':
        dt[unit] = getattr(dt.index, unit)
        months = np.arange(1, 13)
        month_groups = np.array_split(months, len(months) / period)
        for i, month_grp in enumerate(month_groups):
            dt.loc[(dt['month'] >=month_grp[0]) & (dt['month'] <=month_grp[-1]), 'grp_months'] = '{}{}'.format(pdict.get(period), i+1)
    da = dt['grp_months'].to_xarray()
    return da


def compute_consecutive_events_datetimes(da_ts, time_dim='time',
                                         minimum_epochs=10):
    """WARNING : for large xarrays it takes alot of time and memory!"""
    import pandas as pd
    import xarray as xr
    df = da_ts.notnull().to_dataframe()
    A = consecutive_runs(df, num=False)
    # filter out minimum consecutive epochs:
    if minimum_epochs is not None:
        A = A[A['total_True'] > minimum_epochs]
    dt_min = df.iloc[A['{}_True_start'.format(da_ts.name)]].index
    try:
        dt_max = df.iloc[A['{}_True_end'.format(da_ts.name)]].index
    except IndexError:
        dt_max = df.iloc[A['{}_True_end'.format(da_ts.name)][:-1]]
        end = pd.DataFrame(index=[df.index[-1]], data=[False],
                           columns=[da_ts.name])
        dt_max = dt_max.append(end)
        dt_max = dt_max.index
    events = []
    print('done part1')
    for i_min, i_max in zip(dt_min, dt_max):
        events.append(da_ts.sel({time_dim: slice(i_min, i_max)}))
    events_da = xr.concat(events, 'event')
    events_da['event'] = range(len(events))
    return events_da


def multi_time_coord_slice(min_time, max_time, freq='5T', time_dim='time',
                           name='general_group'):
    """returns a datetimeindex array of the multi-time-coords slice defined by
        min_time, max_time vectors and freq."""
    import pandas as pd
    import numpy as np
    assert len(min_time) == len(max_time)
    dates = [
            pd.date_range(
                    start=min_time[i],
                    end=max_time[i],
                    freq=freq) for i in range(
                            len(min_time))]
    dates = [pd.Series(np.ones(dates[i].shape, dtype=int) * i, index=dates[i]) for i in range(len(dates))]
    dates = pd.concat(dates)
    da = dates.to_xarray()
    da = da.rename({'index': time_dim})
    da.name = name
    return da


def calculate_g(lat):
    """calculate the gravitational acceleration with lat in degrees"""
    import numpy as np
    g0 = 9.780325
    nom = 1.0 + 0.00193185 * np.sin(np.deg2rad(lat)) ** 2.0
    denom = 1.0 - 0.00669435 * np.sin(np.deg2rad(lat)) ** 2.0
    g = g0 * (nom / denom)**0.5
    return g


def find_consecutive_vals_df(df, col='class', val=7):
    import numpy as np
    bool_vals = np.where(df[col] == val, 1, 0)
    con_df = consecutive_runs(bool_vals, num=0)
    return con_df


def lat_mean(xarray, method='cos', dim='lat', copy_attrs=True):
    import numpy as np
    import xarray as xr

    def mean_single_da(da, dim=dim, method=method):
        if dim not in da.dims:
            return da
        if method == 'cos':
            weights = np.cos(np.deg2rad(da[dim].values))
            da_mean = (weights * da).sum(dim) / sum(weights)
        if copy_attrs:
            da_mean.attrs = da.attrs
        return da_mean

    xarray = xarray.transpose(..., 'lat')
    if isinstance(xarray, xr.DataArray):
        xarray = mean_single_da(xarray)
    elif isinstance(xarray, xr.Dataset):
        xarray = xarray.map(mean_single_da, keep_attrs=copy_attrs)
    return xarray


def consecutive_runs(arr, num=False):
    import numpy as np
    import pandas as pd
    """get the index ranges (min, max) of the ~num condition.
    num can be 1 or 0 or True or False"""
    # Create an array that is 1 where a is num, and pad each end with an extra
    # 1.
    if isinstance(arr, pd.DataFrame):
        a = arr.squeeze().values
        name = arr.columns[0]
    elif isinstance(arr, np.ndarray):
        a = arr
    elif isinstance(arr, list):
        a = np.array(arr)
    if num == 'nan':
        isone = np.concatenate(([1], np.isnan(a).view(np.int8), [1]))
    else:
        isone = np.concatenate(([1], np.equal(a, num).view(np.int8), [1]))
    absdiff = np.abs(np.diff(isone))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    A = pd.DataFrame(ranges)
    A['2'] = A.iloc[:, 1] - A.iloc[:, 0]
    if isinstance(arr, pd.DataFrame):
        if isinstance(num, bool):
            notnum = not num
        elif isinstance(num, int):
            notnum = 'not-{}'.format(num)
        elif num == 'nan':
            notnum = 'not-nan'
        A.columns = [
            '{}_{}_start'.format(
                name, notnum), '{}_{}_end'.format(
                name, notnum), 'total_{}'.format(notnum)]
    return A


def get_all_possible_combinations_from_list(li, reduce_single_list=True):
    from itertools import combinations
    output = sum([list(map(list, combinations(li, i)))
                  for i in range(len(li) + 1)], [])
    output = output[1:]
    if reduce_single_list:
        output = [x[0] if len(x) == 1 else x for x in output]
    return output


def gantt_chart(ds, fw='bold', ax=None, pe_dict=None, fontsize=14, linewidth=10,
                title='RINEX files availability for the Israeli GNSS stations',
                time_dim='time', antialiased=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib.dates as mdates
    from matplotlib.ticker import AutoMinorLocator
    import matplotlib.patheffects as pe
    # TODO: fix the ticks/ticks labels
    sns.set_palette(sns.color_palette("tab10", len(ds)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    names = [x for x in ds]
    vals = range(1, len(ds) + 1)
    xmin = pd.to_datetime(ds[time_dim].min().values) - pd.Timedelta(1, unit='W')
    xmax = pd.to_datetime(ds[time_dim].max().values) + pd.Timedelta(1, unit='W')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#    dt_min_list = []
#    dt_max_list = []
    for i, da in enumerate(ds):
        print(da)
        df = ds[da].notnull().to_dataframe()
        A = consecutive_runs(df, num=False)
        dt_min = df.iloc[A['{}_True_start'.format(da)]].index
        try:
            dt_max = df.iloc[A['{}_True_end'.format(da)]].index
        except IndexError:
            dt_max = df.iloc[A['{}_True_end'.format(da)][:-1]]
            end = pd.DataFrame(index=[df.index[-1]], data=[False],columns=[da])
            dt_max = dt_max.append(end)
            dt_max = dt_max.index
        y = len(ds) + 1 - np.ones(dt_min.shape) * (i + 1)
#        y_list.append(y)
#        dt_min_list.append(dt_min)
#        dt_max_list.append(dt_max)
        # v = int(calc(i, max = len(ds)))
        if pe_dict is not None:
            ax.hlines(y, dt_min, dt_max, linewidth=linewidth, color=colors[i], path_effects=[pe.Stroke(linewidth=15, foreground='k'), pe.Normal()])
        else:
            ax.hlines(y, dt_min, dt_max, linewidth=linewidth, color=colors[i], antialiased=antialiased)
        #plt.show()
        # ds[da][~ds[da].isnull()] = i + 1
        # ds[da] = ds[da].fillna(0)
    # yticks and their labels:
    ax.set_yticks(vals)
    ax.set_yticklabels(names[::-1], fontweight=fw, fontsize=fontsize)
    [ax.get_yticklabels()[i].set_color(colors[::-1][i]) for i in range(len(colors))]
    ax.set_xlim(xmin, xmax)
    # handle x-axis (time):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major',
        direction='out',
        labeltop=False,
        labelbottom=True,
        top=False,
        bottom=True, left=True, labelsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(which='minor',
        direction='out',
        labeltop=False,
        labelbottom=True,
        top=False,
        bottom=True, left=False)
#     ax.xaxis.set_minor_locator(mdates.YearLocator())
#    ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='center',
             fontweight=fw, fontsize=fontsize)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=30, ha='center',
             fontweight=fw, fontsize=fontsize)
    # grid lines:
#    ax.grid(which='major', axis='x', linestyle='-', color='k')
#    ax.grid(which='minor', axis='x', linestyle='-', color='k')
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight=fw)
    # fig.tight_layout()
    return ax


def time_series_stack_with_window(ts_da, time_dim='time',
                                  window='1D'):
    """make it faster, much faster using isel and then cocant to dataset
    save also the datetimes"""
    import pandas as pd
    import xarray as xr
    
    window_dt = pd.Timedelta(window)
    freq = pd.infer_freq(ts_da[time_dim].values)
    if not any(i.isdigit() for i in freq):
        freq = '1' + freq
    freq_td = pd.Timedelta(freq)
    window_points = int(window_dt / freq_td)
    inds = []
    end_index = ts_da[time_dim].size - window_points
    index_to_run_over = range(0, end_index)
    for i in range(end_index):
        inds.append([i, i + window_points])
    arr_list = []
    arr_time_list = []
    ts_arr = ts_da.values
    ts_time_arr = ts_da[time_dim].values
    for ind in inds:
        arr_list.append(ts_arr[ind[0]: ind[1]])
        arr_time_list.append(ts_time_arr[ind[0]: ind[1]])
    ds = xr.Dataset()
    ds[ts_da.name] = xr.DataArray(arr_list, dims=['start_date', 'points'])
    ds[ts_da.name].attrs = ts_da.attrs
    ds[time_dim] = xr.DataArray(arr_time_list, dims=['start_date', 'points'])
    ds['start_date'] = ts_da.isel({time_dim: index_to_run_over})[time_dim].values
    ds['points'] = range(window_points)
    ds.attrs['freq'] = freq
    return ds


def get_RI_reg_combinations(dataset):
    """return n+1 sized dataset of full regressors and median value regressors"""
    import xarray as xr

    def replace_dta_with_median(dataset, dta):
        ds = dataset.copy()
        ds[dta] = dataset[dta] - dataset[dta] + dataset[dta].median('time')
        ds.attrs['median'] = dta
        return ds
    if type(dataset) != xr.Dataset:
        return print('Input is xarray dataset only')
    ds_list = []
    ds_list.append(dataset)
    dataset.attrs['median'] = 'full_set'
    for da in dataset.data_vars:
        ds_list.append(replace_dta_with_median(dataset, da))
    return ds_list


def annual_standertize(data, time_dim='time', std_nan=1.0):
    """just divide by the time.month std()"""
    attrs = data.attrs
    std_longterm = data.groupby('{}.month'.format(time_dim)).std(keep_attrs=True)
    if std_nan is not None:
        std_longterm = std_longterm.fillna(std_nan)
    data = data.groupby('{}.month'.format(time_dim)) / std_longterm
    data = data.reset_coords(drop=True)
    data.attrs.update(attrs)
    return data

    
def normalize_xr(data, time_dim='time', norm=1, down_bound=-1.,
                 upper_bound=1., verbose=True):
    attrs = data.attrs
    avg = data.mean(time_dim, keep_attrs=True)
    sd = data.std(time_dim, keep_attrs=True)
    if norm == 0:
        data = data
        norm_str = 'No'
    elif norm == 1:
        data = (data-avg)/sd
        norm_str = '(data-avg)/std'
    elif norm == 2:
        data = (data-avg)/avg
        norm_str = '(data-avg)/avg'
    elif norm == 3:
        data = data/avg
        norm_str = '(data/avg)'
    elif norm == 4:
        data = data/sd
        norm_str = '(data)/std'
    elif norm == 5:
        dh = data.max()
        dl = data.min()
        # print dl
        data = (((data-dl)*(upper_bound-down_bound))/(dh-dl))+down_bound
        norm_str = 'mapped between ' + str(down_bound) + ' and ' + str(upper_bound)
        # print data
        if verbose:
            print('Data is ' + norm_str)
    elif norm == 6:
        data = data-avg
        norm_str = 'data-avg'
    if verbose and norm != 5:
            print('Preforming ' + norm_str + ' Normalization')
    data.attrs = attrs
    data.attrs['Normalize'] = norm_str
    return data


def slice_task_date_range(files, date_range, task='non-specific'):
    from aux_gps import get_timedate_and_station_code_from_rinex
    import pandas as pd
    from pathlib import Path
    import logging
    """ return a slice files object (list of rfn Paths) with the correct
    within the desired date range"""
    logger = logging.getLogger('gipsyx')
    date_range = pd.to_datetime(date_range)
    logger.info(
        'performing {}  task within the dates: {} to {}'.format(task,
                                                                date_range[0].strftime(
                                                                    '%Y-%m-%d'),
                                                                date_range[1].strftime('%Y-%m-%d')))
    if not files:
        return files
    path = Path(files[0].as_posix().split('/')[0])
    rfns = [x.as_posix().split('/')[-1][0:12] for x in files]
    dts = get_timedate_and_station_code_from_rinex(rfns)
    rfn_series = pd.Series(rfns, index=dts)
    rfn_series = rfn_series.sort_index()
    mask = (rfn_series.index >= date_range[0]) & (
            rfn_series.index <= date_range[1])
    files = [path / x for x in rfn_series.loc[mask].values]
    return files


def geo_annotate(ax, lons, lats, labels, xytext=(3, 3), fmt=None, c='k',
                 fw='normal', fs=None, colorupdown=False):
    for x, y, label in zip(lons, lats, labels):
        if colorupdown:
            if float(label) >= 0.0:
                c = 'r'
            elif float(label) < 0.0:
                c = 'b'
        if fmt is not None:
            annot = ax.annotate(fmt.format(label), xy=(x, y), xytext=xytext,
                                textcoords="offset points", color=c,
                                fontweight=fw, fontsize=fs)
        else:
            annot = ax.annotate(label, xy=(x, y), xytext=xytext,
                                textcoords="offset points", color=c,
                                fontweight=fw, fontsize=fs)
    return annot


def piecewise_linear_fit(da, k=1, plot=True):
    """return dataarray with coords k "piece" indexing to k parts of
    datetime. k=None means get all datetime index"""
    import numpy as np
    import xarray as xr
    time_dim = list(set(da.dims))[0]
    time_no_nans = da.dropna(time_dim)[time_dim]
    time_pieces = np.array_split(time_no_nans.values, k)
    params = lmfit_params('line')
    best_values = []
    best_fits = []
    for piece in time_pieces:
        dap = da.sel({time_dim: piece})
        result = fit_da_to_model(dap, params, model_dict={'model_name': 'line'},
                                 method='leastsq', plot=False, verbose=False)
        best_values.append(result.best_values)
        best_fits.append(result.best_fit)
    bfs = np.concatenate(best_fits)
    tps = np.concatenate(time_pieces)
    da_final = xr.DataArray(bfs, dims=[time_dim])
    da_final[time_dim] = tps
    if plot:
        ax = plot_tmseries_xarray(da, points=True)
        for piece in time_pieces:
            da_final.sel({time_dim: piece}).plot(color='r', ax=ax)
    return da_final


def lmfit_params(model_name, k=None):
    from lmfit.parameter import Parameters
    sin_params = Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    amp = ['sin_amp', 50, True, None, None, None, None]
    phase = ['sin_phase', 0, True, None, None, None, None]
    # freq = ['sin_freq', 1/365.0, True, None, None, None]
    freq = ['sin_freq', 4, True, None, None, None]
    sin_params.add(*amp)
    sin_params.add(*phase)
    sin_params.add(*freq)
    line_params = Parameters()
    slope = ['line_slope', 1e-6, True, None, None, None, None]
    intercept = ['line_intercept', 58.6, True, None, None, None, None]
    line_params.add(*slope)
    line_params.add(*intercept)
    constant = Parameters()
    constant.add(*['constant', 40.0, True,None, None, None, None])
    if k is not None:
        sum_sin_params = Parameters()
        for mode in range(k):
            amp[0] = 'sin{}_amp'.format(mode)
            phase[0] = 'sin{}_phase'.format(mode)
            freq[0] = 'sin{}_freq'.format(mode)
            sum_sin_params.add(*amp)
            sum_sin_params.add(*phase)
            sum_sin_params.add(*freq)
    if model_name == 'sin_linear':
        return line_params + sin_params
    elif model_name == 'sin':
        return sin_params
    elif model_name == 'sin_constant':
        return sin_params + constant
    elif model_name == 'line':
        return line_params
    elif model_name == 'sum_sin' and k is not None:
        return sum_sin_params
    elif model_name == 'sum_sin_linear' and k is not None:
        return sum_sin_params + line_params


def fit_da_to_model(da, params=None, modelname='sin', method='leastsq', times=None, plot=True, verbose=True):
    """options for modelname:'sin', 'sin_linear', 'line', 'sin_constant', and
    'sum_sin'"""
    # for sum_sin or sum_sin_linear use model_dict={'model_name': 'sum_sin', k:3}
    # usage for params: you need to know the parameter names first:
    # modelname='sin', params=dict(sin_freq={'value':3},sin_amp={'value':0.3},sin_phase={'value':0})
    # fit_da_to_model(alon, modelname='sin', params=dict(sin_freq={'value':3},sin_amp={'value':0.3},sin_phase={'value':0}))
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    time_dim = list(set(da.dims))[0]
    if times is not None:
        da = da.sel({time_dim: slice(*times)})
    lm = lmfit_model_switcher()
    lm.pick_model(modelname)
    lm.generate_params(**params)
    params = lm.params
    model = lm.model
    if verbose:
        print(model)
        print(params)
    jul, jul_no_nans = get_julian_dates_from_da(da)
    y = da.dropna(time_dim).values
    result = model.fit(**params, data=y, time=jul_no_nans, method=method)
    if not result.success:
        raise ValueError('model not fitted properly...')
    fit_y = result.eval(**result.best_values, time=jul)
    fit = xr.DataArray(fit_y, dims=time_dim)
    fit[time_dim] = da[time_dim]
    fit.name = da.name + '_fit'
    p = {}
    for name, param in result.params.items():
        p[name] = [param.value, param.stderr]
    fit.attrs.update(**p)
    # return fit
    if verbose:
        print(result.best_values)
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        da.plot.line(marker='.', linewidth=0., color='b', ax=ax)
        dt = pd.to_datetime(da[time_dim].values)
        ax.plot(dt, fit_y, c='r')
        plt.legend(['data', 'fit'])
    return fit


def get_julian_dates_from_da(da, subtract='first'):
    """transform the time dim of a dataarray to julian dates(days since)"""
    import pandas as pd
    import numpy as np
    # get time dim:
    time_dim = list(set(da.dims))[0]
    # convert to days since 2000 (julian_date):
    jul = pd.to_datetime(da[time_dim].values).to_julian_date()
    # normalize all days to first entry:
    if subtract == 'first':
        first_day = jul[0]
        jul -= first_day
    elif subtract == 'median':
        med = np.median(jul)
        jul -= med
    # do the same but without nans:
    jul_no_nans = pd.to_datetime(
            da.dropna(time_dim)[time_dim].values).to_julian_date()
    if subtract == 'first':
        jul_no_nans -= first_day
    elif subtract == 'median':
        jul_no_nans -= med
    return jul.values, jul_no_nans.values


def lomb_scargle_xr(da_ts, units='cpy', user_freq='MS', plot=True, kwargs=None):
    from astropy.timeseries import LombScargle
    import pandas as pd
    import xarray as xr
    time_dim = list(set(da_ts.dims))[0]
    sp_str = pd.infer_freq(da_ts[time_dim].values)
    if not sp_str:
        print('using user-defined freq: {}'.format(user_freq))
        sp_str = user_freq
    if units == 'cpy':
        # cycles per year:
        freq_dict = {'MS': 12, 'D': 365.25, 'H': 8766}
        long_name = 'Cycles per Year'
    elif units == 'cpd':
        # cycles per day:
        freq_dict = {'H': 24}
        long_name = 'Cycles per Day'
    t = [x for x in range(da_ts[time_dim].size)]
    y = da_ts.values
    lomb_kwargs = {'samples_per_peak': 10, 'nyquist_factor': 2}
    if kwargs is not None:
        lomb_kwargs.update(kwargs)
    freq, power = LombScargle(t, y).autopower(**lomb_kwargs)
    unit_freq = freq_dict.get(sp_str)
    da = xr.DataArray(power, dims=['freq'])
    da['freq'] = freq * unit_freq
    da.attrs['long_name'] = 'Power from LombScargle'
    da.name = '{}_power'.format(da_ts.name)
    da['freq'].attrs['long_name'] = long_name
    if plot:
        da.plot()
    return da


def fft_xr(xarray, method='fft', units='cpy', nan_fill='mean', user_freq='MS',
           plot=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr
    from scipy import signal
#    import matplotlib
#    matplotlib.rcParams['text.usetex'] = True

    def fft_da(da, units, nan_fill, periods):
        time_dim = list(set(da.dims))[0]
        try:
            p_units = da.attrs['units']
        except KeyError:
            p_units = 'amp'
        if nan_fill == 'mean':
            x = da.fillna(da.mean(time_dim))
        elif nan_fill == 'zero':
            x = da.fillna(0)
        # infer freq of time series:
        sp_str = pd.infer_freq(x[time_dim].values)
        if user_freq is None:
            if not sp_str:
                raise Exception('Didnt find a frequency for {}, check for nans!'.format(da.name))
            if len(sp_str) > 1:
                mul = [char for char in sp_str if char.isdigit()]
                sp_str = ''.join([char for char in sp_str if char.isalpha()])
                if not mul:
                    mul = 1
                else:
                    if len(mul) > 1:
                        mul = int(''.join(mul))
                    else:
                        mul = int(mul[0])
                period = sp_str
            elif len(sp_str) == 1:
                mul = 1
                period = sp_str[0]
            p_name = periods[period][0]
            p_val = mul * periods[period][1]
            print('found {} {} frequency in {} time-series'.format(mul, p_name, da.name))
        else:
            p_name = periods[user_freq][0]
            # number of seconds in freq units in time-series:
            p_val = periods[user_freq][1]
            print('using user freq of {}'.format(user_freq))
        print('sample rate in seconds: {}'.format(p_val))
        if method == 'fft':
            # run fft:
            p = 20 * np.log10(np.abs(np.fft.rfft(x, n=None)))
            f = np.linspace(0, (1 / p_val) / 2, len(p))
        elif method == 'welch':
            f, p = signal.welch(x, 1e-6, 'hann', 1024, scaling='spectrum')
        if units == 'cpy':
            unit_freq = 1.0 / periods['Y'][1]  # in Hz
            print('unit_freq: cycles per year ({} seconds)'.format(periods['Y'][1]))
        elif units == 'cpd':
            unit_freq = 1.0 / periods['D'][1]  # in Hz
            print('unit_freq: cycles per day ({} seconds)'.format(periods['D'][1]))
            # unit_freq_in_time_series = unit_freq * p_val   # in Hz
        # f = np.linspace(0, unit_freq_in_time_series / 2, len(p))
        f_in_unit_freq = f / unit_freq
        p_units = '{}^2/{}'.format(p_units, units)
        power = xr.DataArray(p, dims=['freq'])
        power.name = da.name
        power['freq'] = f_in_unit_freq
        power['freq'].attrs['long_name'] = 'Frequency'
        power['freq'].attrs['units'] = units
        power.attrs['long_name'] = 'Power'
        power.attrs['units'] = p_units
        return power

    periods = {'N': ['nanoseconds', 1e-9],
               'U': ['microseconds', 1e-6],
               'us': ['microseconds', 1e-6],
               'L': ['milliseconds', 1e-3],
               'ms': ['milliseconds', 1e-3],
               'T': ['minutes', 60.0],
               'min': ['minutes', 60.0],
               'H': ['hours', 3600.0],
               'D': ['days', 86400.0],
               'W': ['weeks', 604800.0],
               'MS': ['months', 86400.0 * 30],
               'Y': ['years', 86400.0 * 365.25]
               }
    if isinstance(xarray, xr.DataArray):
        power = fft_da(xarray, units, nan_fill, periods)
        if plot:
            fig, ax = plt.subplots(figsize=(6, 8))
            power.plot.line(ax=ax, xscale='log', yscale='log')
            ax.grid()
        return power
    elif isinstance(xarray, xr.Dataset):
        p_list = []
        for da in xarray:
            p_list.append(fft_da(xarray[da], units, nan_fill, periods))
        ds = xr.merge(p_list)
        da_from_ds = ds.to_array(dim='station')
        try:
            ds.attrs['full_name'] = 'Power spectra for {}'.format(xarray.attrs['full_name'])
        except KeyError:
            pass
    elif isinstance(xarray, list):
        p_list = []
        for da in xarray:
            p_list.append(fft_da(da, units, nan_fill, periods))
        ds = xr.merge(p_list, compat='override')
        da_from_ds = ds.to_array(dim='epochs')
        try:
            ds.attrs['full_name'] = 'Power spectra for {}'.format(da.attrs['full_name'])
        except KeyError:
            pass
        if plot:
            da_mean = da_from_ds.mean('epochs')
            da_mean.attrs = da_from_ds.attrs
            # da_from_ds.plot.line(xscale='log', yscale='log', hue='station')
            fig, ax = plt.subplots(figsize=(8, 6))
            da_mean.plot.line(ax=ax, xscale='log', yscale='log')
            ax.grid()
        return ds
    return


def standard_error_slope(X, y):
    """ get the standard error of the slope of the linear regression,
    works in the case that X is a vector only"""
    import numpy as np
    ssxm, ssxym, ssyxm, ssym = np.cov(X, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
    n = len(X)
    df = n - 2
    sterrest = np.sqrt((1 - r**2) * ssym / ssxm / df)
    return sterrest


def tar_dir(files_with_path_to_tar, filename, savepath, compresslevel=9,
            with_dir_struct=False, verbose=False):
    import tarfile as tr
    """ compresses all glob_str_to_tar files (e.g., *.txt) in path_to_tar,
    and save it all to savepath with filename as filename. by default adds .tar
    suffix if not supplied by user. control compression level with
    compresslevel (i.e., None means no compression)."""
    def aname(file, arcname):
        if arcname is None:
            return None
        else:
            return file.as_posix().split('/')[-1]

    path_to_tar = files_with_path_to_tar[0].as_posix().split('/')[0]
    if len(filename.split('.')) < 2:
        filename += '.tar'
        if verbose:
            print('added .tar suffix to {}'.format(filename.split('.'[0])))
    else:
        filename = filename.split('.')[0]
        filename += '.tar'
        if verbose:
            print('changed suffix to tar')
    tarfile = savepath / filename
    if compresslevel is None:
        tar = tr.open(tarfile, "w")
    else:
        tar = tr.open(tarfile, "w:gz", compresslevel=compresslevel)
    if not with_dir_struct:
        arcname = True
        if verbose:
            print('files were archived without directory structure')
    else:
        arcname = None
        if verbose:
            print('files were archived with {} dir structure'.format(path_to_tar))
    total = len(files_with_path_to_tar)
    print('Found {} files to tar in dir {}'.format(total, path_to_tar))
    cnt = 0
    for file in files_with_path_to_tar:
        tar.add(file, arcname=aname(file, arcname=arcname))
        cnt += 1
#        if np.mod(cnt, 10) == 0:
#            print('.', end=" ")
    tar.close()
    print('Compressed all files in {} to {}'.format(
          path_to_tar, savepath / filename))
    return


def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    import sys
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def get_var(varname):
    """get a linux shell var (without the $)"""
    import subprocess
    CMD = 'echo $%s' % varname
    p = subprocess.Popen(
        CMD,
        stdout=subprocess.PIPE,
        shell=True,
        executable='/bin/bash')
    out = p.stdout.readlines()[0].strip().decode("utf-8")
    if len(out) == 0:
        return None
    else:
        return out


def plot_tmseries_xarray(ds, fields=None, points=False, error_suffix='_error',
                         errorbar_alpha=0.5, trend_suffix='_trend'):
    """plot time-series plot w/o errorbars of a xarray dataset"""
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    if points:
        ma = '.'  # marker
        lw = 0.  # linewidth
    else:
        ma = None  # marker
        lw = 1.0  # linewidth
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
#    if len(ds.dims) > 1:
#        raise ValueError('Number of dimensions in Dataset exceeds 1!')
    if isinstance(fields, str):
        fields = [fields]
    error_fields = [x for x in ds.data_vars if error_suffix in x]
    trend_fields = [x for x in ds.data_vars if trend_suffix in x]
    if fields is None and error_fields:
        all_fields = [x for x in ds.data_vars if error_suffix not in x]
    elif fields is None and trend_fields:
        all_fields = [x for x in ds.data_vars if trend_suffix not in x]
    elif fields is None and not error_fields:
        all_fields = [x for x in ds.data_vars]
    elif fields is None and not trend_fields:
        all_fields = [x for x in ds.data_vars]
    elif fields is not None and isinstance(fields, list):
        all_fields = sorted(fields)
    time_dim = list(set(ds[all_fields].dims))[0]
    if len(all_fields) == 1:
        da = ds[all_fields[0]]
        ax = da.plot(figsize=(20, 4), color='b', marker=ma, linewidth=lw)[0].axes
        ax.grid()
        if error_fields:
            print('adding errorbars fillbetween...')
            error = da.name + error_suffix
            ax.fill_between(da[time_dim].values, da.values - ds[error].values,
                            da.values + ds[error].values,
                            where=np.isfinite(da.values),
                            alpha=errorbar_alpha)
        if trend_fields:
            print('adding trends...')
            trend = da.name + trend_suffix
            da[trend].plot(ax=ax, color='r')
            trend_attr = [x for x in da[trend].attrs.keys()
                          if 'trend' in x][0]
            if trend_attr:
                trend_str = trend_attr.split('>')[-1]
                trend_val = da[trend].attrs[trend_attr]
                ax.text(0.1, 0.9, '{}: {:.2f}'.format(trend_str, trend_val),
                        horizontalalignment='center',
                        verticalalignment='top', color='green', fontsize=15,
                        transform=ax.transAxes)
        ax.grid(True)
        ax.set_title(da.name)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        return ax
    else:
        da = ds[all_fields].to_array('var')
        fg = da.plot(row='var', sharex=True, sharey=False, figsize=(20, 15),
                     hue='var', color='k', marker=ma, linewidth=lw)
        for i, (ax, field) in enumerate(zip(fg.axes.flatten(), all_fields)):
            ax.grid(True)
            if error_fields:
                print('adding errorbars fillbetween...')
                ax.fill_between(da[time_dim].values,
                                da.sel(var=field).values - ds[field + error_suffix].values,
                                da.sel(var=field).values + ds[field + error_suffix].values,
                                where=np.isfinite(da.sel(var=field).values),
                                alpha=errorbar_alpha)
            if trend_fields:
                print('adding trends...')
                ds[field + trend_suffix].plot(ax=ax, color='r')
                trend_attr = [x for x in ds[field + trend_suffix].attrs.keys()
                              if 'trend' in x][0]
                if trend_attr:
                    trend_str = trend_attr.split('>')[-1]
                    trend_val = ds[field + trend_suffix].attrs[trend_attr]
                    ax.text(0.1, 0.9, '{}: {:.2f}'.format(trend_str, trend_val),
                            horizontalalignment='center',
                            verticalalignment='top', color='green', fontsize=15,
                            transform=ax.transAxes)
            try:
                ax.set_ylabel('[' + ds[field].attrs['units'] + ']')
            except KeyError:
                pass
            ax.lines[0].set_color('C{}'.format(i))
            ax.grid(True)
        # fg.fig.suptitle()
        fg.fig.subplots_adjust(left=0.1, top=0.93)
    return fg


def flip_xy_axes(ax, ylim=None):
    if ylim is None:
        new_y_lim = ax.get_xlim()
    else:
        new_y_lim = ylim
    new_x_lim = ax.get_ylim()
    ylabel = ax.get_xlabel()
    xlabel = ax.get_ylabel()
    newx = ax.lines[0].get_ydata()
    newy = ax.lines[0].get_xdata()
    # set new x- and y- data for the line
    # ax.margins(y=0)
    ax.lines[0].set_xdata(newx)
    ax.lines[0].set_ydata(newy)
    ax.set_xlim(new_x_lim)
    ax.set_ylim(new_y_lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_yaxis()
    return ax


def choose_time_groupby_arg(da_ts, time_dim='time', grp='hour'):
    if grp != 'date':
        grp_arg = '{}.{}'.format(time_dim, grp)
    else:
        grp_arg = groupby_date_xr(da_ts)
    return grp_arg


def time_series_stack(time_da, time_dim='time', grp1='hour', grp2='month',
                      plot=True):
    """Takes a time-series xr.DataArray objects and reshapes it using
    grp1 and grp2. output is a xr.Dataset that includes the reshaped DataArray
    , its datetime-series and the grps. plots the mean also"""
    import xarray as xr
    import pandas as pd
    # try to infer the freq and put it into attrs for later reconstruction:
    freq = pd.infer_freq(time_da[time_dim].values)
    name = time_da.name
    time_da.attrs['freq'] = freq
    attrs = time_da.attrs
    # drop all NaNs:
    time_da = time_da.dropna(time_dim)
    # first grouping:
    grp1_arg = choose_time_groupby_arg(time_da, time_dim=time_dim, grp=grp1)
    grp_obj1 = time_da.groupby(grp1_arg)
    da_list = []
    t_list = []
    for grp1_name, grp1_inds in grp_obj1.groups.items():
        da = time_da.isel({time_dim: grp1_inds})
        if grp2 is not None:
            # second grouping:
            grp2_arg = choose_time_groupby_arg(time_da, time_dim=time_dim, grp=grp2)
            grp_obj2 = da.groupby(grp2_arg)
            for grp2_name, grp2_inds in grp_obj2.groups.items():
                da2 = da.isel({time_dim: grp2_inds})
                # extract datetimes and rewrite time coord to 'rest':
                times = da2[time_dim]
                times = times.rename({time_dim: 'rest'})
                times.coords['rest'] = range(len(times))
                t_list.append(times)
                da2 = da2.rename({time_dim: 'rest'})
                da2.coords['rest'] = range(len(da2))
                da_list.append(da2)
        else:
            times = da[time_dim]
            times = times.rename({time_dim: 'rest'})
            times.coords['rest'] = range(len(times))
            t_list.append(times)
            da = da.rename({time_dim: 'rest'})
            da.coords['rest'] = range(len(da))
            da_list.append(da)
    # get group keys:
    grps1 = [x for x in grp_obj1.groups.keys()]
    if grp2 is not None:
        grps2 = [x for x in grp_obj2.groups.keys()]
    # concat and convert to dataset:
    stacked_ds = xr.concat(da_list, dim='all').to_dataset(name=name)
    stacked_ds[time_dim] = xr.concat(t_list, 'all')
    if grp2 is not None:
        # create a multiindex for the groups:
        mindex = pd.MultiIndex.from_product([grps1, grps2], names=[grp1, grp2])
        stacked_ds.coords['all'] = mindex
    else:
        # create a multiindex for first group only:
        mindex = pd.MultiIndex.from_product([grps1], names=[grp1])
        stacked_ds.coords['all'] = mindex
    # unstack:
    ds = stacked_ds.unstack('all')
    ds.attrs = attrs
#    if plot:
#        plot_stacked_time_series(ds[name].mean('rest', keep_attrs=True))
    return ds


def plot_stacked_time_series(stacked_da):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    import numpy as np
    try:
        units = stacked_da.attrs['units']
    except KeyError:
        units = ''
    try:
        station = stacked_da.attrs['station']
    except KeyError:
        station = ''
    try:
        name = stacked_da.name
    except KeyError:
        name = ''
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # plt.rc('text', usetex=True)
    grp1_mean = stacked_da.mean(stacked_da.dims[0])
    grp2_mean = stacked_da.mean(stacked_da.dims[1])
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(
        2, 2, width_ratios=[
            1, 4], height_ratios=[
            5, 1], wspace=0, hspace=0)
    # grid = plt.GridSpec(2, 2, hspace=0.5, wspace=0.2)
#        ax_main = fig.add_subplot(grid[:-1, :-1])
#        ax_left = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
#        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
    ax_main = fig.add_subplot(grid[0, 1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_left.grid()
    ax_bottom = fig.add_subplot(grid[1, 1])
    ax_bottom.grid()
    pcl = stacked_da.T.plot.contourf(
        ax=ax_main, add_colorbar=False, cmap=plt.cm.get_cmap(
            'viridis', 41), levels=41)
    ax_main.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax_main.tick_params(
        direction='out',
        top='on',
        bottom='off',
        left='off',
        right='on',
        labelleft='off',
        labelbottom='off',
        labeltop='on',
        labelright='on',
        which='major')
    ax_main.tick_params(
        direction='out',
        top='on',
        bottom='off',
        left='off',
        right='on',
        which='minor')
    ax_main.grid(
        True,
        which='major',
        axis='both',
        linestyle='-',
        color='k',
        alpha=0.2)
    ax_main.grid(
        True,
        which='minor',
        axis='both',
        linestyle='--',
        color='k',
        alpha=0.2)
    ax_main.tick_params(
        top='on',
        bottom='off',
        left='off',
        right='on',
        labelleft='off',
        labelbottom='off',
        labeltop='on',
        labelright='on')
    bottom_limit = ax_main.get_xlim()
    left_limit = ax_main.get_ylim()
    grp1_mean.plot(ax=ax_left)
    grp2_mean.plot(ax=ax_bottom)
    ax_bottom.set_xlim(bottom_limit)
    ax_left = flip_xy_axes(ax_left, left_limit)
    ax_bottom.set_ylabel(r'${}$'.format(units), fontsize=12)
    ax_left.set_xlabel(r'${}$'.format(units), fontsize=12)
    fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(ax_main)
    # cax1 = divider.append_axes("right", size="5%", pad=0.2)
    # [left, bottom, width, height] of figure:
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.75])
    # fig.colorbar(pcl, orientation="vertical", pad=0.2, label=units)
    pcl_ticks = np.linspace(
        stacked_da.min().item(),
        stacked_da.max().item(),
        11)
    cbar = fig.colorbar(
        pcl,
        cax=cbar_ax,
        label=r'${}$'.format(units),
        ticks=pcl_ticks)
    cbar.set_ticklabels(['{:.1f}'.format(x) for x in pcl_ticks])
    title = ' '.join([name, station])
    fig.suptitle(title, fontweight='bold', fontsize=15)
    return fig


def time_series_stack_decrapeted(time_da, time_dim='time', grp1='hour', grp2='month'):
    """Takes a time-series xr.DataArray objects and reshapes it using
    grp1 and grp2. outout is a xr.Dataset that includes the reshaped DataArray
    , its datetime-series and the grps."""
    import xarray as xr
    import numpy as np
    import pandas as pd
    # try to infer the freq and put it into attrs for later reconstruction:
    freq = pd.infer_freq(time_da[time_dim].values)
    name = time_da.name
    time_da.attrs['freq'] = freq
    attrs = time_da.attrs
    # drop all NaNs:
    time_da = time_da.dropna(time_dim)
    # group grp1 and concat:
    grp_obj1 = time_da.groupby(time_dim + '.' + grp1)
    s_list = []
    for grp_name, grp_inds in grp_obj1.groups.items():
        da = time_da.isel({time_dim: grp_inds})
        s_list.append(da)
    grps1 = [x for x in grp_obj1.groups.keys()]
    stacked_da = xr.concat(s_list, dim=grp1)
    stacked_da[grp1] = grps1
    # group over the concatenated da and concat again:
    grp_obj2 = stacked_da.groupby(time_dim + '.' + grp2)
    s_list = []
    for grp_name, grp_inds in grp_obj2.groups.items():
        da = stacked_da.isel({time_dim: grp_inds})
        s_list.append(da)
    grps2 = [x for x in grp_obj2.groups.keys()]
    stacked_da = xr.concat(s_list, dim=grp2)
    stacked_da[grp2] = grps2
    # numpy part:
    # first, loop over both dims and drop NaNs, append values and datetimes:
    vals = []
    dts = []
    for grp1_val in stacked_da[grp1]:
        da = stacked_da.sel({grp1: grp1_val})
        for grp2_val in da[grp2]:
            val = da.sel({grp2: grp2_val}).dropna(time_dim)
            vals.append(val.values)
            dts.append(val[time_dim].values)
    # second, we get the max of the vals after the second groupby:
    max_size = max([len(x) for x in vals])
    # we fill NaNs and NaT for the remainder of them:
    concat_sizes = [max_size - len(x) for x in vals]
    concat_arrys = [np.empty((x)) * np.nan for x in concat_sizes]
    concat_vals = [np.concatenate(x) for x in list(zip(vals, concat_arrys))]
    # 1970-01-01 is the NaT for this time-series:
    concat_arrys = [np.zeros((x), dtype='datetime64[ns]')
                    for x in concat_sizes]
    concat_dts = [np.concatenate(x) for x in list(zip(dts, concat_arrys))]
    concat_vals = np.array(concat_vals)
    concat_dts = np.array(concat_dts)
    # finally , we reshape them:
    concat_vals = concat_vals.reshape((stacked_da[grp1].shape[0],
                                       stacked_da[grp2].shape[0],
                                       max_size))
    concat_dts = concat_dts.reshape((stacked_da[grp1].shape[0],
                                     stacked_da[grp2].shape[0],
                                     max_size))
    # create a Dataset and DataArrays for them:
    sda = xr.Dataset()
    sda.attrs = attrs
    sda[name] = xr.DataArray(concat_vals, dims=[grp1, grp2, 'rest'])
    sda[time_dim] = xr.DataArray(concat_dts, dims=[grp1, grp2, 'rest'])
    sda[grp1] = grps1
    sda[grp2] = grps2
    sda['rest'] = range(max_size)
    return sda


#def time_series_stack2(time_da, time_dim='time', grp1='hour', grp2='month',
#                       plot=True):
#    """produces a stacked plot with two groupings for a time-series"""
#    import xarray as xr
#    import matplotlib.pyplot as plt
#    import numpy as np
#    import matplotlib.ticker as tck
#    grp_obj1 = time_da.groupby(time_dim + '.' + grp1)
#    s_list = []
#    for grp_name, grp_inds in grp_obj1.groups.items():
#        da = time_da.isel({time_dim: grp_inds})
#        # da = da.rename({time_dim: grp + '_' + str(grp_name)})
#        # da.name += '_' + grp + '_' + str(grp_name)
#        s_list.append(da)
#    grps1 = [x for x in grp_obj1.groups.keys()]
#    stacked_da = xr.concat(s_list, dim=grp1)
#    stacked_da[grp1] = grps1
#    s_list = []
#    for grp_val in grps1:
#        da = stacked_da.sel({grp1: grp_val}).groupby(time_dim + '.' + grp2).mean()
#        s_list.append(da)
#    stacked_da2 = xr.concat(s_list, dim=grp1)
#    if plot:
#        try:
#            units = time_da.attrs['units']
#        except KeyError:
#            units = ''
#        try:
#            station = time_da.attrs['station']
#        except KeyError:
#            station = ''
#        try:
#            name = time_da.name
#        except KeyError:
#            name = ''
#        SMALL_SIZE = 12
#        MEDIUM_SIZE = 16
#        BIGGER_SIZE = 18
#        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#        grp1_mean = stacked_da2.mean(grp1)
#        grp2_mean = stacked_da2.mean(grp2)
#        fig = plt.figure(figsize=(16, 10), dpi=80)
#        grid = plt.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[5, 1], wspace=0, hspace=0)
#        # grid = plt.GridSpec(2, 2, hspace=0.5, wspace=0.2)
##        ax_main = fig.add_subplot(grid[:-1, :-1])
##        ax_left = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
##        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
#        ax_main = fig.add_subplot(grid[0, 1])
#        ax_left = fig.add_subplot(grid[0, 0])
#        ax_left.grid()
#        ax_bottom = fig.add_subplot(grid[1, 1])
#        ax_bottom.grid()
#        pcl = stacked_da2.T.plot.pcolormesh(ax = ax_main, add_colorbar=False, cmap=plt.cm.get_cmap('viridis', 19), snap=True)
#        ax_main.xaxis.set_minor_locator(tck.AutoMinorLocator())
#        ax_main.tick_params(direction='out', top='on', bottom='off', left='off', right='on', labelleft='off', labelbottom='off', labeltop='on', labelright='on', which='major')
#        ax_main.tick_params(direction='out', top='on', bottom='off', left='off', right='on', which='minor')
#        ax_main.grid(True, which='major', axis='both', linestyle='-', color='k', alpha=0.2)
#        ax_main.grid(True, which='minor', axis='both', linestyle='--', color='k', alpha=0.2)
#        ax_main.tick_params(top='on', bottom='off', left='off', right='on', labelleft='off', labelbottom='off', labeltop='on', labelright='on')
#        bottom_limit = ax_main.get_xlim()
#        left_limit = ax_main.get_ylim()
#        grp1_mean.plot(ax=ax_left)
#        grp2_mean.plot(ax=ax_bottom)
#        ax_bottom.set_xlim(bottom_limit)
#        ax_left = flip_xy_axes(ax_left, left_limit)
#        ax_bottom.set_ylabel(units)
#        ax_left.set_xlabel(units)
#        fig.subplots_adjust(right=0.8)
#        # divider = make_axes_locatable(ax_main)
#        # cax1 = divider.append_axes("right", size="5%", pad=0.2)
#        # [left, bottom, width, height] of figure:
#        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.75])
#        # fig.colorbar(pcl, orientation="vertical", pad=0.2, label=units)
#        pcl_ticks = np.linspace(stacked_da2.min().item(), stacked_da2.max().item(), 11)
#        cbar = fig.colorbar(pcl, cax=cbar_ax, label=units, ticks=pcl_ticks)
#        cbar.set_ticklabels(['{:.1f}'.format(x) for x in pcl_ticks])
#        title = ' '.join([name, station])
#        fig.suptitle(title, fontweight='bold', fontsize=15)
#        # fig.colorbar(pcl, ax=ax_main)
#        # plt.colorbar(pcl, cax=ax_main)
#    return stacked_da2


#def time_series_stack_decraped(time_da, time_dim='time', grp='hour', plot=True):
#    import xarray as xr
#    grp_obj = time_da.groupby(time_dim + '.' + grp)
#    s_list = []
#    for grp_name, grp_inds in grp_obj.groups.items():
#        da = time_da.isel({time_dim: grp_inds})
#        # da = da.rename({time_dim: grp + '_' + str(grp_name)})
#        # da.name += '_' + grp + '_' + str(grp_name)
#        s_list.append(da)
#    grps = [x for x in grp_obj.groups.keys()]
#    stacked_da = xr.concat(s_list, dim=grp)
#    stacked_da[grp] = grps
#    if 'year' in grp:
#        resample_span = '1Y'
#    elif grp == 'month':
#        resample_span = '1Y'
#    elif grp == 'day':
#        resample_span = '1MS'
#    elif grp == 'hour':
#        resample_span = '1D'
#    elif grp == 'minute':
#        resample_span = '1H'
#    stacked_da = stacked_da.resample({time_dim: resample_span}).mean(time_dim)
#    if plot:
#        stacked_da.T.plot.pcolormesh(figsize=(6, 8))
#    return stacked_da


def dt_to_np64(time_coord, unit='m', convert_back=False):
    """accepts time_coord and a required time unit and returns a dataarray
    of time_coord and unix POSIX continous float index"""
    import numpy as np
    import xarray as xr
    unix_epoch = np.datetime64(0, unit)
    one_time_unit = np.timedelta64(1, unit)
    time_unit_since_epoch = (time_coord.values - unix_epoch) / one_time_unit
    units = {'Y': 'years', 'M': 'months', 'W': 'weeks', 'D': 'days',
             'h': 'hours', 'm': 'minutes', 's': 'seconds'}
    new_time = xr.DataArray(time_unit_since_epoch, coords=[time_coord],
                            dims=[time_coord.name])
    new_time.attrs['units'] = units[unit] + ' since 1970-01-01 00:00:00'
    return new_time


def xr_reindex_with_date_range(ds, drop=True, time_dim=None, freq='5min',
                               dt_min=None, dt_max=None):
    """be careful when drop=True in datasets that have various nans in dataarrays"""
    import pandas as pd
    if time_dim is None:
        time_dim = list(set(ds.dims))[0]
    if drop:
        ds = ds.dropna(time_dim)
    if dt_min is not None:
        dt_min = pd.to_datetime(dt_min)
        start = pd.to_datetime(dt_min)
    else:
        start = pd.to_datetime(ds[time_dim].min().item())
    if dt_max is not None:
        dt_max = pd.to_datetime(dt_max)
        end = pd.to_datetime(dt_max)
    else:
        end = pd.to_datetime(ds[time_dim].max().item())
    new_time = pd.date_range(start, end, freq=freq)
    ds = ds.reindex({time_dim: new_time})
    return ds


def add_attr_to_xr(da, key, value, append=False):
    """add attr to da, if append=True, appends it, if key exists"""
    import xarray as xr
    if isinstance(da, xr.Dataset):
        raise TypeError('only xr.DataArray allowd!')
    if key in da.attrs and not append:
        raise ValueError('{} already exists in {}, use append=True'.format(key, da.name))
    elif key in da.attrs and append:
        da.attrs[key] += value
    else:
        da.attrs[key] = value
    return da


def filter_nan_errors(ds, error_str='_error', dim='time', meta='action'):
    """return the data in a dataarray only if its error is not NaN,
    assumes that ds is a xr.dataset and includes fields and their error
   like this: field, field+error_str"""
    import xarray as xr
    import numpy as np
    from aux_gps import add_attr_to_xr
    if isinstance(ds, xr.DataArray):
        raise TypeError('only xr.Dataset allowd!')
    fields = [x for x in ds.data_vars if error_str not in x]
    for field in fields:
        ds[field] = ds[field].where(np.isfinite(
            ds[field + error_str])).dropna(dim)
        if meta in ds[field].attrs:
            append = True
        add_attr_to_xr(
            ds[field],
            meta,
            ', filtered values with NaN errors',
            append)
    return ds


def smooth_xr(da, dim='time', weights=[0.25, 0.5, 0.25]):
    # fix to accept wither da or ds:
    import xarray as xr
    weight = xr.DataArray(weights, dims=['window'])
    if isinstance(da, xr.Dataset):
        attrs = dict(zip(da.data_vars, [da[x].attrs for x in da]))
        da_roll = da.to_array('dummy').rolling(
            {dim: len(weights)}, center=True).construct('window').dot(weight)
        da_roll = da_roll.to_dataset('dummy')
        for das, attr in attrs.items():
            da_roll[das].attrs = attr
            da_roll[das].attrs['action'] = 'weighted rolling mean with {} on {}'.format(
                weights, dim)
    else:
        da_roll = da.rolling({dim: len(weights)},
                             center=True).construct('window').dot(weight)
        da_roll.attrs['action'] = 'weighted rolling mean with {} on {}'.format(
            weights, dim)
    return da_roll


def keep_iqr(da, dim='time', qlow=0.25, qhigh=0.75, k=1.5, drop_with_freq=None,
             verbose=False):
    """return the data in a dataarray only in the k times the
    Interquartile Range (low, high), drop"""
    from aux_gps import add_attr_to_xr
    from aux_gps import xr_reindex_with_date_range
    try:
        quan = da.quantile([qlow, qhigh], dim).values
    except TypeError:
        # support for datetime64 dtypes:
        if da.dtype == '<M8[ns]':
            quan = da.astype(int).quantile(
                    [qlow, qhigh], dim).astype('datetime64[ns]').values
        # support for timedelta64 dtypes:
        elif da.dtype == '<m8[ns]':
            quan = da.astype(int).quantile(
                    [qlow, qhigh], dim).astype('timedelta64[ns]').values
    low = quan[0]
    high = quan[1]
    iqr = high - low
    lower = low - (iqr * k)
    higher = high + (iqr * k)
    before = da.size
    da = da.where((da < higher) & (da > lower)).dropna(dim)
    after = da.size
    if verbose:
        print('dropped {} outliers from {}.'.format(before-after, da.name))
    if 'action' in da.attrs:
        append = True
    else:
        append = False
    add_attr_to_xr(
        da, 'action', ', kept IQR ({}, {}, {})'.format(
            qlow, qhigh, k), append)
    if drop_with_freq is not None:
        da = xr_reindex_with_date_range(da, freq=drop_with_freq)
    return da


def transform_ds_to_lat_lon_alt(ds, coords_name=['X', 'Y', 'Z'],
                                error_str='_error', time_dim='time'):
    """appends to the data vars of ds(xr.dataset) the lat, lon, alt fields
    and their error where the geocent fields are X, Y, Z"""
    import xarray as xr
    from aux_gps import get_latlonalt_error_from_geocent_error
    geo_fields = [ds[x].values for x in coords_name]
    geo_errors = [ds[x + error_str].values for x in coords_name]
    latlong = get_latlonalt_error_from_geocent_error(*geo_fields, *geo_errors)
    new_fields = ['lon', 'lat', 'alt', 'lon_error', 'lat_error', 'alt_error']
    new_names = ['Longitude', 'Latitude', 'Altitude']
    new_units = ['Degrees', 'Degrees', 'm']
    for name, data in zip(new_fields, latlong):
        ds[name] = xr.DataArray(data, dims=[time_dim])
    for name, unit, full_name in zip(new_fields[0:3], new_units[0:3],
                                     new_names[0:3]):
        ds[name].attrs['full_name'] = full_name
        ds[name].attrs['units'] = unit
    return ds


def get_latlonalt_error_from_geocent_error(X, Y, Z, xe=None, ye=None, ze=None):
    """returns the value and error in lat(decimal degree), lon(decimal degree)
    and alt(meters) for X, Y, Z in geocent coords (in meters), all input is
    lists or np.arrays"""
    import pyproj
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=False)
    if (xe is not None) and (ye is not None) and (ze is not None):
        lon_pe, lat_pe, alt_pe = pyproj.transform(ecef, lla, X + xe, Y + ye,
                                                  Z + ze, radians=False)
        lon_me, lat_me, alt_me = pyproj.transform(ecef, lla, X - xe, Y - ye,
                                                  Z - ze, radians=False)
        lon_e = (lon_pe - lon_me) / 2.0
        lat_e = (lat_pe - lat_me) / 2.0
        alt_e = (alt_pe - alt_me) / 2.0
        return lon, lat, alt, lon_e, lat_e, alt_e
    else:
        return lon, lat, alt


def path_glob(path, glob_str='*.Z', return_empty_list=False):
    """returns all the files with full path(pathlib3 objs) if files exist in
    path, if not, returns FilenotFoundErro"""
    from pathlib import Path
#    if not isinstance(path, Path):
#        raise Exception('{} must be a pathlib object'.format(path))
    path = Path(path)
    files_with_path = [file for file in path.glob(glob_str) if file.is_file]
    if not files_with_path and not return_empty_list:
        raise FileNotFoundError('{} search in {} found no files.'.format(glob_str,
                                path))
    elif not files_with_path and return_empty_list:
        return files_with_path
    else:
        return files_with_path


def find_cross_points(df, cols=None):
    """find if col A is crossing col B in df and is higher (Up) or lower (Down)
    than col B (after crossing). cols=None means that the first two cols of
    df are used."""
    import numpy as np
    if cols is None:
        cols = df.columns.values[0:2]
    df['Diff'] = df[cols[0]] - df[cols[1]]
    df['Cross'] = np.select([((df.Diff < 0) & (df.Diff.shift() > 0)), ((
        df.Diff > 0) & (df.Diff.shift() < 0))], ['Up', 'Down'], None)
    return df


def get_rinex_filename_from_datetime(station, dt='2012-05-07'):
    """return rinex filename from datetime string"""
    import pandas as pd

    def filename_from_single_date(station, date):
        day = pd.to_datetime(date, format='%Y-%m-%d').dayofyear
        year = pd.to_datetime(date, format='%Y-%m-%d').year
        if len(str(day)) == 1:
            str_day = '00' + str(day) + '0'
        elif len(str(day)) == 2:
            str_day = '0' + str(day) + '0'
        elif len(str(day)) == 3:
            str_day = str(day) + '0'
        filename = station.lower() + str_day + '.' + str(year)[2:4] + 'd'
        return filename

    if isinstance(dt, list):
        filenames = []
        for date in dt:
            filename = filename_from_single_date(station, date)
            filenames.append(filename)
        return filenames
    else:
        filename = filename_from_single_date(station, dt)
        return filename


def get_timedate_and_station_code_from_rinex(rinex_str='tela0010.05d',
                                             just_dt=False):
    """return datetime from rinex2 format"""
    import pandas as pd
    import datetime

    def get_dt_from_single_rinex(rinex_str):
        station = rinex_str[0:4]
        days = int(rinex_str[4:7])
        year = rinex_str[-3:-1]
        Year = datetime.datetime.strptime(year, '%y').strftime('%Y')
        dt = datetime.datetime(int(Year), 1, 1) + datetime.timedelta(days - 1)
        dt = pd.to_datetime(dt)
        return dt, station.upper()

    if isinstance(rinex_str, list):
        dt_list = []
        for rstr in rinex_str:
            dt, station = get_dt_from_single_rinex(rstr)
            dt_list.append(dt)
        return dt_list
    else:
        dt, station = get_dt_from_single_rinex(rinex_str)
    if just_dt:
        return dt
    else:
        return dt, station


def configure_logger(name='general', filename=None):
    import logging
    import sys
    stdout_handler = logging.StreamHandler(sys.stdout)
    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        handlers = [file_handler, stdout_handler]
    else:
        handlers = [stdout_handler]

    logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            handlers=handlers
            )
    logger = logging.getLogger(name=name)
    return logger


def process_gridsearch_results(GridSearchCV):
    import xarray as xr
    import pandas as pd
    import numpy as np
    """takes GridSreachCV object with cv_results and xarray it into dataarray"""
    params = GridSearchCV.param_grid
    scoring = GridSearchCV.scoring
    names = [x for x in params.keys()]
    if len(params) > 1:
        # unpack param_grid vals to list of lists:
        pro = [[y for y in x] for x in params.values()]
        ind = pd.MultiIndex.from_product((pro), names=names)
        result_names = [x for x in GridSearchCV.cv_results_.keys() if
                        'time' not in x and 'param' not in x and
                        'rank' not in x]
        ds = xr.Dataset()
        for da_name in result_names:
            da = xr.DataArray(GridSearchCV.cv_results_[da_name])
            ds[da_name] = da
        ds = ds.assign(dim_0=ind).unstack('dim_0')
    elif len(params) == 1:
        result_names = [x for x in GridSearchCV.cv_results_.keys() if
                        'time' not in x and 'param' not in x and
                        'rank' not in x]
        ds = xr.Dataset()
        for da_name in result_names:
            da = xr.DataArray(GridSearchCV.cv_results_[da_name], dims={**params})
            ds[da_name] = da
        for k, v in params.items():
            ds[k] = v
    name = [x for x in ds.data_vars.keys() if 'split' in x and 'test' in x]
    split_test = xr.concat(ds[name].data_vars.values(), dim='kfolds')
    split_test.name = 'split_test'
    kfolds_num = len(name)
    name = [x for x in ds.data_vars.keys() if 'split' in x and 'train' in x]
    split_train = xr.concat(ds[name].data_vars.values(), dim='kfolds')
    split_train.name = 'split_train'
    name = [x for x in ds.data_vars.keys() if 'mean_test' in x]
    mean_test = xr.concat(ds[name].data_vars.values(), dim='scoring')
    mean_test.name = 'mean_test'
    name = [x for x in ds.data_vars.keys() if 'mean_train' in x]
    mean_train = xr.concat(ds[name].data_vars.values(), dim='scoring')
    mean_train.name = 'mean_train'
    name = [x for x in ds.data_vars.keys() if 'std_test' in x]
    std_test = xr.concat(ds[name].data_vars.values(), dim='scoring')
    std_test.name = 'std_test'
    name = [x for x in ds.data_vars.keys() if 'std_train' in x]
    std_train = xr.concat(ds[name].data_vars.values(), dim='scoring')
    std_train.name = 'std_train'
    ds = ds.drop(ds.data_vars.keys())
    ds['mean_test'] = mean_test
    ds['mean_train'] = mean_train
    ds['std_test'] = std_test
    ds['std_train'] = std_train
    ds['split_test'] = split_test
    ds['split_train'] = split_train
    mean_test_train = xr.concat(ds[['mean_train', 'mean_test']].data_vars.
                                values(), dim='train_test')
    std_test_train = xr.concat(ds[['std_train', 'std_test']].data_vars.
                               values(), dim='train_test')
    split_test_train = xr.concat(ds[['split_train', 'split_test']].data_vars.
                                 values(), dim='train_test')
    ds['train_test'] = ['train', 'test']
    ds = ds.drop(ds.data_vars.keys())
    ds['MEAN'] = mean_test_train
    ds['STD'] = std_test_train
    # CV = xr.Dataset(coords=GridSearchCV.param_grid)
    ds = xr.concat(ds[['MEAN', 'STD']].data_vars.values(), dim='MEAN_STD')
    ds['MEAN_STD'] = ['MEAN', 'STD']
    ds.name = 'CV_mean_results'
    ds.attrs['param_names'] = names
    if isinstance(scoring, str):
        ds.attrs['scoring'] = scoring
        ds = ds.squeeze(drop=True)
    else:
        ds['scoring'] = scoring
    ds = ds.to_dataset()
    ds['CV_full_results'] = split_test_train
    ds['kfolds'] = np.arange(kfolds_num)
    return ds


def calculate_std_error(arr, statistic='std'):
    from scipy.stats import moment
    import numpy as np
    # remove nans:
    arr = arr[np.logical_not(np.isnan(arr))]
    n = len(arr)
    if statistic == 'std':
        mu4 = moment(arr, moment=4)
        sig4 = np.var(arr)**2.0
        se = mu4 - sig4 * (n - 3) / (n - 1)
        se = (se / n)**0.25
    elif statistic == 'mean':
        std = np.std(arr)
        se = std / np.sqrt(n)
    return se


def calculate_distance_between_two_lat_lon_points(
        lat1,
        lon1,
        lat2,
        lon2,
        orig_epsg='4326',
        meter_epsg='2039',
        verbose=False):
    """calculate the distance between two points (lat,lon) with epsg of
    WGS84 and convert to meters with a local epsg. if lat1 is array then
    calculates the distance of many points."""
    import geopandas as gpd
    import pandas as pd
    try:
        df1 = pd.DataFrame(index=lat1.index)
    except AttributeError:
        try:
            len(lat1)
        except TypeError:
            lat1 = [lat1]
        df1 = pd.DataFrame(index=[x for x in range(len(lat1))])
    df1['lat'] = lat1
    df1['lon'] = lon1
    first_gdf = gpd.GeoDataFrame(
        df1, geometry=gpd.points_from_xy(
            df1['lon'], df1['lat']))
    first_gdf.crs = {'init': 'epsg:{}'.format(orig_epsg)}
    first_gdf.to_crs(epsg=int(meter_epsg), inplace=True)
    try:
        df2 = pd.DataFrame(index=lat2.index)
    except AttributeError:
        try:
            len(lat2)
        except TypeError:
            lat2 = [lat2]
        df2 = pd.DataFrame(index=[x for x in range(len(lat2))])
    df2['lat'] = lat2
    df2['lon'] = lon2
    second_gdf = gpd.GeoDataFrame(
        df2, geometry=gpd.points_from_xy(
            df2['lon'], df2['lat']))
    second_gdf.crs = {'init': 'epsg:{}'.format(orig_epsg)}
    second_gdf.to_crs(epsg=int(meter_epsg), inplace=True)
    ddf = first_gdf.geometry.distance(second_gdf.geometry)
    return ddf


def get_nearest_lat_lon_for_xy(lat_da, lon_da, points):
    """used to access UERRA reanalysis, where the variable has x,y as coords"""
    import numpy as np
    from scipy.spatial import cKDTree
    if isinstance(points, np.ndarray):
        points = list(points)
    combined_x_y_arrays = np.dstack(
        [lat_da.values.ravel(), lon_da.values.ravel()])[0]
    mytree = cKDTree(combined_x_y_arrays)
    points = np.atleast_2d(points)
    dist, inds = mytree.query(points)
    yx = []
    for ind in inds:
        y, x = np.unravel_index(ind, lat_da.shape)
        yx.append([y, x])
    return yx


def coarse_dem(data, dem_path=work_yuval / 'AW3D30'):
    """coarsen to data coords"""
    # data is lower resolution than awd
    import salem
    import xarray as xr
    # determine resulotion:
    try:
        lat_size = data.lat.size
        lon_size = data.lon.size
    except AttributeError:
        print('data needs to have lat and lon coords..')
        return
    # check for file exist:
    filename = 'israel_dem_' + str(lon_size) + '_' + str(lat_size) + '.nc'
    my_file = dem_path / filename
    if my_file.is_file():
        awds = xr.open_dataarray(my_file)
        print('{} is found and loaded...'.format(filename))
    else:
        awd = salem.open_xr_dataset(dem_path / 'israel_dem.tif')
        awds = data.salem.lookup_transform(awd)
        awds = awds['data']
        awds.to_netcdf(dem_path / filename)
        print('{} is saved to {}'.format(filename, dem_path))
    return awds


def invert_dict(d):
    """unvert dict"""
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = key
            else:
                inverse[item].append(key)
    return inverse


def concat_shp(path, shp_file_list, saved_filename):
    import geopandas as gpd
    import pandas as pd
    shapefiles = [path / x for x in shp_file_list]
    gdf = pd.concat([gpd.read_file(shp)
                     for shp in shapefiles]).pipe(gpd.GeoDataFrame)
    gdf.to_file(path / saved_filename)
    print('saved {} to {}'.format(saved_filename, path))
    return


def scale_xr(da, upper=1.0, lower=0.0, unscale=False):
    if not unscale:
        dh = da.max()
        dl = da.min()
        da_scaled = (((da-dl)*(upper-lower))/(dh-dl)) + lower
        da_scaled.attrs = da.attrs
        da_scaled.attrs['scaled'] = True
        da_scaled.attrs['lower'] = dl.item()
        da_scaled.attrs['upper'] = dh.item()
    if unscale and da.attrs['scaled']:
        dh = da.max()
        dl = da.min()
        upper = da.attrs['upper']
        lower = da.attrs['lower']
        da_scaled = (((da-dl)*(upper-lower))/(dh-dl)) + lower
    return da_scaled


def print_saved_file(name, path):
    print(name + ' was saved to ' + str(path))
    return


def dim_union(da_list, dim='time'):
    import pandas as pd
    setlist = [set(x[dim].values) for x in da_list]
    empty_list = [x for x in setlist if not x]
    if empty_list:
        print('NaN dim drop detected, check da...')
        return
    u = list(set.union(*setlist))
    # new_dim = list(set(a.dropna(dim)[dim].values).intersection(
    #     set(b.dropna(dim)[dim].values)))
    if dim == 'time':
        new_dim = sorted(pd.to_datetime(u))
    else:
        new_dim = sorted(u)
    return new_dim


def dim_intersection(da_list, dim='time', dropna=True, verbose=None):
    import pandas as pd
    if dropna:
        setlist = [set(x.dropna(dim)[dim].values) for x in da_list]
    else:
        setlist = [set(x[dim].values) for x in da_list]
    empty_list = [x for x in setlist if not x]
    if empty_list:
        if verbose == 0:
            print('NaN dim drop detected, check da...')
        return None
    u = list(set.intersection(*setlist))
    # new_dim = list(set(a.dropna(dim)[dim].values).intersection(
    #     set(b.dropna(dim)[dim].values)))
    if dim == 'time':
        new_dim = sorted(pd.to_datetime(u))
    else:
        new_dim = sorted(u)
    return new_dim


def get_unique_index(da, dim='time', verbose=False):
    import numpy as np
    before = da[dim].size
    _, index = np.unique(da[dim], return_index=True)
    da = da.isel({dim: index})
    after = da[dim].size
    if verbose:
        print('dropped {} duplicate coord entries.'.format(before-after))
    return da


def Zscore_xr(da, dim='time'):
    """input is a dattarray of data and output is a dattarray of Zscore
    for the dim"""
    attrs = da.attrs
    z = (da - da.mean(dim=dim)) / da.std(dim=dim)
    z.attrs = attrs
    if 'units' in attrs.keys():
        z.attrs['old_units'] = attrs['units']
    z.attrs['action'] = 'converted to Z-score'
    z.attrs['units'] = 'std'
    return z


def desc_nan(data, verbose=True):
    """count only NaNs in data and returns the thier amount and the non-NaNs"""
    import numpy as np
    import xarray as xr

    def nan_da(data):
        nans = np.count_nonzero(np.isnan(data.values))
        non_nans = np.count_nonzero(~np.isnan(data.values))
        if verbose:
            print(str(type(data)))
            print(data.name + ': non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
            print('Dimensions:')
        dim_nn_list = []
        for dim in data.dims:
            dim_len = data[dim].size
            dim_non_nans = np.int(data.dropna(dim)[dim].count())
            dim_nn_list.append(dim_non_nans)
            if verbose:
                print(dim + ': non-NaN labels: ' +
                      str(dim_non_nans) + ' of total ' + str(dim_len))
        return non_nans
    if isinstance(data, xr.DataArray):
        nn_dict = nan_da(data)
        return nn_dict
    elif isinstance(data, np.ndarray):
        nans = np.count_nonzero(np.isnan(data))
        non_nans = np.count_nonzero(~np.isnan(data))
        if verbose:
            print(str(type(data)))
            print('non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
    elif isinstance(data, xr.Dataset):
        for varname in data.data_vars.keys():
            non_nans = nan_da(data[varname])
    return non_nans


class lmfit_model_switcher(object):
    def pick_model(self, model_name, *args, **kwargs):
        """Dispatch method"""
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
        # Call the method as we return it
        self.model = method(*args, **kwargs)
        return self

    def pick_param(self, name, **kwargs):
        # **kwargs.keys() = value, vary, min, max, expr
        if not hasattr(self, 'model'):
            raise('pls pick model first!')
            return
        else:
            self.model.set_param_hint(name, **kwargs)
        return

    def generate_params(self, **kwargs):
        if not hasattr(self, 'model'):
            raise('pls pick model first!')
            return
        else:
            if kwargs is not None:
                for key, val in kwargs.items():
                    self.model.set_param_hint(key, **val)
                self.params = self.model.make_params()
            else:
                self.params = self.model.make_params()
        return

    def line(self, line_pre='line_'):
        from lmfit import Model

        def func(time, slope, intercept):
            f = slope * time + intercept
            return f
        return Model(func, independent_vars=['time'], prefix=line_pre)

    def sin(self, sin_pre='sin_'):
        from lmfit import Model

        def func(time, amp, freq, phase):
            import numpy as np
            f = amp * np.sin(2 * np.pi * freq * (time - phase))
            return f
        return Model(func, independent_vars=['time'], prefix=sin_pre)

    def sin_constant(self, sin_pre='sin_', con_pre='constant_'):
        from lmfit.models import ConstantModel

        constant = ConstantModel(prefix=con_pre)
        lmfit = lmfit_model_switcher()
        lmfit.pick_model('sin', sin_pre)
        return lmfit.model + constant

    def sin_linear(self, sin_pre='sin_', line_pre='line_'):
        lmfit = lmfit_model_switcher()
        sin = lmfit.pick_model('sin', sin_pre)
        line = lmfit.pick_model('line', line_pre)
        return sin + line

    def sum_sin(self, k):
        lmfit = lmfit_model_switcher()
        sin = lmfit.pick_model('sin', 'sin0_')
        for k in range(k-1):
            sin += lmfit.pick_model('sin', 'sin{}_'.format(k+1))
        return sin

    def sum_sin_constant(self, k, con_pre='constant_'):
        from lmfit.models import ConstantModel
        constant = ConstantModel(prefix=con_pre)
        lmfit = lmfit_model_switcher()
        sum_sin = lmfit.pick_model('sum_sin', k)
        return sum_sin + constant

    def sum_sin_linear(self, k, line_pre='line_'):
        lmfit = lmfit_model_switcher()
        sum_sin = lmfit.pick_model('sum_sin', k)
        line = lmfit.pick_model('line', line_pre)
        return sum_sin + line
