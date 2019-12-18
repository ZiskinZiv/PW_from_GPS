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


def time_series_stack_with_window(ts_da, time_dim='time',
                                  window='1D'):
    """make it faster, much faster using isel and then cocant to dataset
    save also the datetimes"""
    import pandas as pd
    import xarray as xr
    window_dt = pd.Timedelta(window)
    freq = pd.infer_freq(ts_da[time_dim].values)
    freq_td = pd.Timedelta('1' + freq, unit=freq)
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
    return ds


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
    freq = ['sin_freq', 1/365.0, True, None, None, None]
    sin_params.add(*amp)
    sin_params.add(*phase)
    sin_params.add(*freq)
    line_params = Parameters()
    slope = ['line_slope', 1e-6, True, None, None, None, None]
    intercept = ['line_intercept', 58.6, True, None, None, None, None]
    line_params.add(*slope)
    line_params.add(*intercept)
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
    elif model_name == 'line':
        return line_params
    elif model_name == 'sum_sin' and k is not None:
        return sum_sin_params
    elif model_name == 'sum_sin_linear' and k is not None:
        return sum_sin_params + line_params


def fit_da_to_model(da, params, model_dict={'model_name': 'sin'},
                    method='leastsq', times=None, plot=True, verbose=True):
    """options for modelname:'sin', 'sin_line', 'line', 'sin_constant', and
    'sum_sin'"""
    # for sum_sin or sum_sin_linear use model_dict={'model_name': 'sum_sin', k:3}
    import matplotlib.pyplot as plt
    import pandas as pd
    time_dim = list(set(da.dims))[0]
    if times is not None:
        da = da.sel({time_dim: slice(*times)})
    lm = lmfit_model_switcher()
    model = lm.pick_model(**model_dict)
    if verbose:
        print(model)
    jul, jul_no_nans = get_julian_dates_from_da(da)
    y = da.dropna(time_dim).values
    result = model.fit(**params, data=y, time=jul_no_nans, method=method)
    fit_y = result.eval(**result.best_values, time=jul)
    if verbose:
        print(result.best_values)
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        da.plot.line(marker='.', linewidth=0., color='b', ax=ax)
        dt = pd.to_datetime(da[time_dim].values)
        ax.plot(dt, fit_y, c='r')
        plt.legend(['data', 'fit'])
    return result


def get_julian_dates_from_da(da):
    """transform the time dim of a dataarray to julian dates(days since)"""
    import pandas as pd
    # get time dim:
    time_dim = list(set(da.dims))[0]
    # convert to days since 2000 (julian_date):
    jul = pd.to_datetime(da[time_dim].values).to_julian_date()
    # normalize all days to first entry:
    first_day = jul[0]
    jul -= first_day
    # do the same but without nans:
    jul_no_nans = pd.to_datetime(
            da.dropna(time_dim)[time_dim].values).to_julian_date()
    jul_no_nans -= first_day
    return jul.values, jul_no_nans.values


def fft_xr(xarray, units='cpy', nan_fill='mean', plot=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr
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
        # infer freq of time series:
        sp_str = pd.infer_freq(x[time_dim].values)
        if not sp_str:
            raise Exception('Didnt find a frequency for {}, check for nans!'.format(da.name))
        if len(sp_str) > 1:
            sp_str = [char for char in sp_str]
            mul = int(sp_str[0])
            period = sp_str[1]
        elif len(sp_str) == 1:
            mul = 1
            period = sp_str[0]
        p_name = periods[period][0]
        # number of seconds in freq units in time-series:
        p_val = mul * periods[period][1]
        print('found {} {} frequency in {} time-series'.format(mul, p_name, da.name))
        # run fft:
        p = 20 * np.log10(np.abs(np.fft.rfft(x)))
        if units == 'cpy':
            unit_freq = 1.0 / periods['Y'][1]  # in Hz
            # unit_freq_in_time_series = unit_freq * p_val   # in Hz
        # f = np.linspace(0, unit_freq_in_time_series / 2, len(p))
        f = np.linspace(0, (1 / p_val) / 2, len(p))
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
        if plot:
            da_mean = da_from_ds.mean('station')
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
    grp_obj1 = time_da.groupby(time_dim + '.' + grp1)
    da_list = []
    t_list = []
    for grp1_name, grp1_inds in grp_obj1.groups.items():
        da = time_da.isel({time_dim: grp1_inds})
        # second grouping:
        grp_obj2 = da.groupby(time_dim + '.' + grp2)
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
    # get group keys:
    grps1 = [x for x in grp_obj1.groups.keys()]
    grps2 = [x for x in grp_obj2.groups.keys()]
    # concat and convert to dataset:
    stacked_ds = xr.concat(da_list, dim='all').to_dataset(name=name)
    stacked_ds[time_dim] = xr.concat(t_list, 'all')
    # create a multiindex for the groups:
    mindex = pd.MultiIndex.from_product([grps1, grps2], names=[grp1, grp2])
    stacked_ds.coords['all'] = mindex
    # unstack:
    ds = stacked_ds.unstack('all')
    ds.attrs = attrs
    if plot:
        plot_stacked_time_series(ds[name].mean('rest', keep_attrs=True))
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
    grp1_mean = stacked_da.mean(stacked_da.dims[0])
    grp2_mean = stacked_da.mean(stacked_da.dims[1])
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[5, 1], wspace=0, hspace=0)
    # grid = plt.GridSpec(2, 2, hspace=0.5, wspace=0.2)
#        ax_main = fig.add_subplot(grid[:-1, :-1])
#        ax_left = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
#        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
    ax_main = fig.add_subplot(grid[0, 1])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_left.grid()
    ax_bottom = fig.add_subplot(grid[1, 1])
    ax_bottom.grid()
    pcl = stacked_da.T.plot.pcolormesh(ax = ax_main, add_colorbar=False, cmap=plt.cm.get_cmap('viridis', 19), snap=True)
    ax_main.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax_main.tick_params(direction='out', top='on', bottom='off', left='off', right='on', labelleft='off', labelbottom='off', labeltop='on', labelright='on', which='major')
    ax_main.tick_params(direction='out', top='on', bottom='off', left='off', right='on', which='minor')
    ax_main.grid(True, which='major', axis='both', linestyle='-', color='k', alpha=0.2)
    ax_main.grid(True, which='minor', axis='both', linestyle='--', color='k', alpha=0.2)
    ax_main.tick_params(top='on', bottom='off', left='off', right='on', labelleft='off', labelbottom='off', labeltop='on', labelright='on')
    bottom_limit = ax_main.get_xlim()
    left_limit = ax_main.get_ylim()
    grp1_mean.plot(ax=ax_left)
    grp2_mean.plot(ax=ax_bottom)
    ax_bottom.set_xlim(bottom_limit)
    ax_left = flip_xy_axes(ax_left, left_limit)
    ax_bottom.set_ylabel(units)
    ax_left.set_xlabel(units)
    fig.subplots_adjust(right=0.8)
    # divider = make_axes_locatable(ax_main)
    # cax1 = divider.append_axes("right", size="5%", pad=0.2)
    # [left, bottom, width, height] of figure:
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.75])
    # fig.colorbar(pcl, orientation="vertical", pad=0.2, label=units)
    pcl_ticks = np.linspace(stacked_da.min().item(), stacked_da.max().item(), 11)
    cbar = fig.colorbar(pcl, cax=cbar_ax, label=units, ticks=pcl_ticks)
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


def xr_reindex_with_date_range(ds, drop=True, freq='5min'):
    import pandas as pd
    time_dim = list(set(ds.dims))[0]
    if drop:
        ds = ds.dropna(time_dim)
    start = pd.to_datetime(ds[time_dim].min().item())
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


def keep_iqr(ds, dim='time', qlow=0.25, qhigh=0.75, k=1.5):
    """return the data in a dataset or dataarray only in the
    Interquartile Range (low, high)"""
    import xarray as xr

    def keep_iqr_da(da, dim, qlow, qhigh, meta='action'):
        from aux_gps import add_attr_to_xr
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
        da = da.where((da < higher) & (da > lower)).dropna(dim)
        if meta in da.attrs:
            append = True
        else:
            append = False
        add_attr_to_xr(
            da, meta, ', kept IQR ({}, {}, {})'.format(
                qlow, qhigh, k), append)
        return da
    if isinstance(ds, xr.DataArray):
        filtered_da = keep_iqr_da(ds, dim, qlow, qhigh)
    elif isinstance(ds, xr.Dataset):
        da_list = []
        for name in ds.data_vars:
            da = keep_iqr_da(ds[name], dim, qlow, qhigh)
            da_list.append(da)
        filtered_da = xr.merge(da_list)
    return filtered_da


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


def get_unique_index(da, dim='time'):
    import numpy as np
    _, index = np.unique(da[dim], return_index=True)
    da = da.isel({dim: index})
    return da


def Zscore_xr(da, dim='time'):
    """input is a dattarray of data and output is a dattarray of Zscore
    for the dim"""
    z = (da - da.mean(dim=dim)) / da.std(dim=dim)
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
        return method(*args, **kwargs)

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
        sin = lmfit.pick_model('sin', sin_pre)
        return sin + constant

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
