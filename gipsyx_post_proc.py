#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:24:01 2019

@author: shlomi
"""

import pandas as pd
from PW_paths import work_yuval
from PW_paths import work_path
from PW_paths import geo_path
from PW_paths import cwd

garner_path = work_yuval / 'garner'
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
sound_path = work_yuval / 'sounding'
rinex_on_geo = geo_path / 'Work_Files/PW_yuval/rinex'
PW_stations_path = work_yuval / '1minute'
stations = pd.read_csv('All_gps_stations.txt', header=0, delim_whitespace=True,
                       index_col='name')


def plot_gipsy_field(ds, fields='WetZ'):
    import matplotlib.pyplot as plt
    import numpy as np
    if isinstance(fields, list) and len(fields) == 1:
        fields = fields[0]
    if fields is None:
        all_fields = sorted(list(set([x.split('_')[0] for x in ds.data_vars])))
        desc = [ds[x].attrs['description'] for x in ds[all_fields]]
        units = [ds[x].attrs['units'] for x in ds[all_fields]]
        fig, axes = plt.subplots(
            len(all_fields), 1, figsize=(
                20, 15), sharex=True)
        df = ds.to_dataframe()
        for i, (ax, field, name, unit) in enumerate(
                zip(axes.flatten(), all_fields, desc, units)):
            df[field].plot(
                ax=ax,
                style='.',
                linewidth=0.,
                color="C{}".format(i))
            ax.fill_between(df.index,
                            df[field].values - df[field + '_error'].values,
                            df[field].values + df[field + '_error'].values,
                            where=np.isfinite(df['WetZ'].values),
                            alpha=0.5)
            ax.grid()
            ax.set_title(name)
            ax.set_ylabel(unit)
        fig.tight_layout()
    elif fields is not None and isinstance(fields, str):
        fig, ax = plt.subplots(figsize=(16, 5))
        ds[fields].plot.line(marker='.', linewidth=0., ax=ax, color='b')
        ax.fill_between(ds.time.values,
                        ds[fields].values - ds[fields + '_error'].values,
                        ds[fields].values + ds[fields + '_error'].values,
                        where=np.isfinite(ds[fields]),
                        alpha=0.5)
        ax.grid()
        fig.tight_layout()
    elif fields is not None and isinstance(fields, list):
        fig, axes = plt.subplots(len(fields), 1, figsize=(20, 15), sharex=True)
        desc = [ds[x].attrs['description'] for x in ds[fields]]
        units = [ds[x].attrs['units'] for x in ds[fields]]
        df = ds.to_dataframe()
        for i, (ax, field, name, unit) in enumerate(
                zip(axes.flatten(), fields, desc, units)):
            df[field].plot(
                ax=ax,
                style='.',
                linewidth=0.,
                color="C{}".format(i))
            ax.fill_between(df.index,
                            df[field].values - df[field + '_error'].values,
                            df[field].values + df[field + '_error'].values,
                            where=np.isfinite(df['WetZ'].values),
                            alpha=0.5)
            ax.grid()
            ax.set_title(name)
            ax.set_ylabel(unit)
        fig.tight_layout()
    return


def read_gipsyx_all_yearly_files(load_save_path, plot=False):
    from aux_gps import path_glob
    import xarray as xr
    files = sorted(path_glob(load_save_path, '*.nc'))
    ds_list = []
    for file in files:
        filename = file.as_posix().split('/')[-1]
        station = file.as_posix().split('/')[-1].split('_')[0]
        if 'ppp_post' not in filename:
            continue
        print('concatanating {}'.format(filename))
        dss = xr.open_dataset(file)
        ds_list.append(dss)
    ds = xr.concat(ds_list, 'time')
    for name, var in dss.data_vars.items():
        ds[name].attrs = var.attrs
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    ymin = ds.time.min().dt.year.item()
    ymax = ds.time.max().dt.year.item()
    new_filename = '{}_PPP_{}-{}.nc'.format(station, ymin, ymax)
    ds.to_netcdf(load_save_path / new_filename, 'w', encoding=encoding)
    print('{} was saved to {}'.format(new_filename, load_save_path))
    return ds


def post_procces_gipsyx_all_years(load_save_path, plot=False):
    from aux_gps import path_glob
    files = sorted(path_glob(load_save_path, '*.nc'))
    for file in files:
        filename = file.as_posix().split('/')[-1]
        station = file.as_posix().split('/')[-1].split('_')[0]
        year = file.as_posix().split('/')[-1].split('_')[-1].split('.')[0]
        if 'raw' not in filename:
            continue
        new_filename = '{}_ppp_post_{}.nc'.format(station, year)
        if (load_save_path / new_filename).is_file():
            print('{} already exists in {}, skipping...'.format(new_filename,
                                                                load_save_path))
            continue
        ds = post_procces_gipsyx_yearly_file(file, savepath=load_save_path,
                                             plot=False)
    return


def post_procces_gipsyx_yearly_file(path_file, savepath=None, plot=False):
    import xarray as xr
    from aux_gps import get_unique_index
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
#    from scipy import stats
#    import pandas as pd
#    import seaborn as sns
    station = path_file.as_posix().split('/')[-1].split('_')[0]
    year = path_file.as_posix().split('/')[-1].split('_')[-1].split('.')[0]
    print('proccessing {} station in year: {}'.format(station, year))
    dss = xr.load_dataset(path_file)
    da_fs = []
    meta = dss.attrs
    vars_list = list(set([x.split('-')[0] for x in dss.data_vars.keys()]))
    for field in vars_list:
        da_field = analyse_results_ds_one_station(dss, field, verbose=0)
        da_year = replace_fields_in_ds(dss, da_field, field, verbose=0)
        da_fs.append(da_year)
    ds = xr.merge(da_fs)
    df = get_unique_index(ds, 'time').to_dataframe()
    st = df.index.min()
    ed = df.index.max()
    new_time = pd.date_range(st, ed, freq='5min')
    df = df.reindex(new_time)
    df.index.name = 'time'
    ds = df.to_xarray()
    # filter outlies (zscore>3):
    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    # df = df[df > 0]
    # ds = df.to_xarray()
    ds.attrs = meta
    desc = ['Zenith Wet Delay', 'North Gradient of Zenith Wet Delay',
            'East Gradient of Zenith Wet Delay', 'Longitude', 'Latitude',
            'Altitude', 'WGS84(geocentric) X coordinate',
            'WGS84(geocentric) Y coordinate', 'WGS84(geocentric) Z coordinate']
    desc_error = [x + ' Error' for x in desc]
    units = ['cm', 'cm/m', 'cm/m', 'Degrees', 'Degrees', 'm',
             'm', 'm', 'm']
    fields = ['WetZ', 'GradNorth', 'GradEast', 'lon', 'lat', 'alt', 'X', 'Y',
              'Z']
    fields_error = [x + '_error' for x in fields]
    units_dict = dict(zip(fields, units))
    desc_dict = dict(zip(fields, desc))
    desc_er_dict = dict(zip(fields_error, desc_error))
    units_er_dict = dict(zip(fields_error, units))
    for field in fields:
        ds[field].attrs['units'] = units_dict[field]
        ds[field].attrs['description'] = desc_dict[field]
    for er_field in fields_error:
        ds[er_field].attrs['units'] = units_er_dict[er_field]
        ds[er_field].attrs['description'] = desc_er_dict[er_field]
    ds = get_unique_index(ds, 'time')
    if plot:
        fig, axes = plt.subplots(6, 1, figsize=(20, 15), sharex=True)
        df = ds.to_dataframe()
        for ax, field, name, unit in zip(axes.flatten(), fields, desc, units):
            df[field].plot(ax=ax, style='.', linewidth=0., color='k')
            ax.fill_between(df.index,
                            df[field].values - df[field + '_error'].values,
                            df[field].values + df[field + '_error'].values,
                            where=np.isfinite(df['WetZ'].values),
                            alpha=0.5)
            ax.grid()
            ax.set_title(name)
            ax.set_ylabel(unit)
        fig.tight_layout()
    if savepath is not None:
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in ds.data_vars}
#        ymin = ds.time.min().dt.year.item()
#        ymax = ds.time.max().dt.year.item()
#        filename = '{}_PPP_{}-{}.nc'.format(station, ymin, ymax)
#        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
#        print('{} was saved to {}'.format(filename, savepath))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        new_filename = '{}_ppp_post_{}.nc'.format(station, year)
        ds.to_netcdf(savepath / new_filename, 'w', encoding=encoding)
        print('{} was saved to {}'.format(new_filename, savepath))
    return ds


def replace_fields_in_ds(dss, da_repl, field='WetZ', verbose=None):
    """replaces dss overlapping field(and then some) with the stiched signal
    fron da_repl. be carful with the choices for field"""
    from aux_gps import get_unique_index
    import xarray as xr
    if verbose == 0:
        print('replacing {} field.'.format(field))
    # choose the field from the bigger dss:
    nums = sorted(list(set([int(x.split('-')[1]) for x in dss])))
    ds = dss[['{}-{}'.format(field, i) for i in nums]]
    da_list = []
    for i, _ in enumerate(ds):
        if i == len(ds) - 1:
            break
        first = ds['{}-{}'.format(field, i)]
        second = ds['{}-{}'.format(field, i+1)]
        min_time = first.dropna('time').time.min()
        max_time = second.dropna('time').time.max()
        da = da_repl.sel(time=slice(min_time, max_time))
        if verbose == 1:
            print('proccesing {} and {}'.format(first.name, second.name))
        # utime = dim_union([first, second], 'time')
        first_time = set(first.dropna('time').time.values).difference(set(da.time.values))
        second_time = set(second.dropna('time').time.values).difference(set(da.time.values))
        first = first.sel(time=list(first_time))
        second = second.sel(time=list(second_time))
        da_list.append(xr.concat([first, da, second], 'time'))
    da_final = xr.concat(da_list, 'time')
    da_final = da_final.sortby('time')
    da_final.name = field
    da_final = get_unique_index(da_final, 'time')
    return da_final


def analyse_results_ds_one_station(dss, field='WetZ', verbose=None,
                                   plot=False):
    """analyse and find an overlapping signal to fields 'WetZ' or 'WetZ_error'
    in dss"""
    # algorithm for zwd stitching of 30hrs gipsyx runs:
    # just take the mean of the two overlapping signals
    # and then smooth is with savgol_filter using 3 hours more data in each
    # direction...
    import matplotlib.pyplot as plt
    import pandas as pd

    def stitch_two_cols(df, window=25, order=3, method='smooth_mean',
                        cols=None):
        """Use smoothing with savgol filter on the mean of two overlapping
        signals. assume that df columns are : A, B, if cols=None
        means take A, B to be the first two cols of df"""
        from scipy.signal import savgol_filter
        import numpy as np
        if cols is None:
            cols = df.columns.values[0:2]
        if method == 'smooth_mean':
            df['Mean'] = df[cols].mean(axis=1)
            sav = savgol_filter(df.Mean.values, window, order)
            df['stitched_signal'] = sav
        elif method == 'error':
            df['stitched_signal'] = np.sqrt(df[cols[0]].fillna(0)**2 +
                                            df[cols[1]].fillna(0)**2)
        elif method == 'just_mean':
            df['stitched_signal'] = df[cols].mean(axis=1)
        return df

    def select_two_ds_from_gipsyx_results(ds, names=['WetZ_0', 'WetZ_1'],
                                          hours_offset=None):
        """selects two dataarrays from the raw gipsyx results dataset"""
        from aux_gps import dim_intersection
        import xarray as xr
        time = dim_intersection([ds[names[0]], ds[names[1]]], dim='time')
        if not time:
            return None
        if hours_offset is not None:
            # freq = pd.infer_freq(time)
            start = time[0] - pd.DateOffset(hours=hours_offset)
            end = time[-1] + pd.DateOffset(hours=hours_offset)
            # time = pd.date_range(start, end, freq=freq)
            first = ds[names[0]].sel(time=slice(start, end))
            second = ds[names[1]].sel(time=slice(start, end))
        else:
            first = ds[names[0]].sel(time=time)
            second = ds[names[1]].sel(time=time)
        two = xr.Dataset()
        two[first.name] = first
        two[second.name] = second
        df = two.to_dataframe()
        return df
    if verbose == 0:
        print('analysing {} field.'.format(field))
    # first, group different vars for different stitching schemes:
    to_smooth = ['GradEast', 'GradNorth', 'WetZ']
    to_just_mean = ['X', 'Y', 'Z', 'lat', 'lon', 'alt']
    to_error = [x + '_error' for x in to_smooth] + [x + '_error' for x in
                                                    to_just_mean]
    # second, select the field to work on:
    nums = sorted(list(set([int(x.split('-')[1]) for x in dss])))
    ds = dss[['{}-{}'.format(field, i) for i in nums]]
    df_list = []
    for i, _ in enumerate(ds):
        if i == len(ds) - 1:
            break
        first = ds['{}-{}'.format(field, i)]
        second = ds['{}-{}'.format(field, i+1)]
        if verbose == 1:
            print('proccesing {} and {}'.format(first.name, second.name))
        # 3 hours addition to each side:
        df = select_two_ds_from_gipsyx_results(ds, [first.name, second.name],
                                               3)
        if df is not None:
            if field in to_smooth:
                stitched = stitch_two_cols(df, method='smooth_mean')
            elif field in to_just_mean:
                stitched = stitch_two_cols(df, method='just_mean')
            elif field in to_error:
                stitched = stitch_two_cols(df, method='error')
            df_list.append(stitched)
            # df_list.append(find_cross_points(df, None))
        elif df is None:
            if verbose:
                print('skipping {} and {}...'.format(first.name, second.name))
    da = pd.concat([x['stitched_signal'] for x in df_list]).to_xarray()

    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        da.plot.line(marker='.', linewidth=0., ax=ax, color='k')
        for i, ppp in enumerate(ds):
            ds['{}-{}'.format(field, i)].plot(ax=ax)
#    dfs = []
#    for df in df_list:
#        # check if there is an offset:
#        A = df.columns.values[0]
#        B = df.columns.values[1]
#        if all([x is None for x in df.Cross]):
#            offset = df.Diff.median()
#            df['{}_new'.format(B)] = df[B] + offset
#            dfs.append(df)
    return da


#def gipsyx_rnxedit_errors(df1, savepath=None):
#    """get the df output of gipsyx_runs_error_analysis and map out the reciever
#    error analysis using regex and print the output"""
#    df = df1.copy()
#    error_col = df.columns.values.item()
#    df['receiver'] = df[error_col].str.findall(r"'(.*?)'")
#    df['receiver'] = [x[0] if x is not None else None for x in df['receiver']]
#    text = [df.loc[i, error_col]
#            for i in df.index if df.loc[i, error_col] is not None][0]
#    station = error_col.split('_')[0]
#    if savepath is not None:
#        filename = station + '_rnxEdit_errors.txt'
#        with open(savepath / filename, 'a') as f:
#            f.write("%s\n" % text)
#            f.write("dataframe: \n")
#            df['receiver'].to_csv(f)
#            print('{} was saved to {}'.format(filename, savepath))
#    return df


#def gipsyx_runs_error_analysis(path):
#    from collections import Counter
#    from aux_gps import get_timedate_and_station_code_from_rinex
#
#    def further_filter(counter):
#        return c
#
#    def find_errors(content_list, name):
#        if len(content_list) <= 1:
#            return None
#        elif len(content_list) > 1:
#            keys = [x for x in content_list if 'KeyError' in x]
#            vals = [x for x in content_list if 'ValueError' in x]
#            excpt = [x for x in content_list if 'Exception' in x]
#            err = [x for x in content_list if 'Error' in x]
#            errors = keys + vals + excpt + err
#        if not errors:
#            dt, _ = get_timedate_and_station_code_from_rinex(name)
#            print('found new error on {} ({})'.format(name,  dt.strftime('%Y-%m-%d')))
#        return errors
#    edict = {}
#    good = 0
#    bad = 0
#    for file in path.glob('*.err'):
#        filename = file.as_posix().split('/')[-1][0:12]
#        if good == 0 and bad == 0:
#            print('running error analysis for station {}'.format(filename[0:4]))
#        with open(file) as f:
#            content = f.readlines()
#            # you may also want to remove whitespace characters like `\n` at
#            # the end of each line
#            content = [x.strip() for x in content]
#            errors = find_errors(content, filename)
#            if errors is not None:
#                edict[filename] = list(set(errors))
#                bad += 1
#            else:
#                good += 1
#    g = [get_timedate_and_station_code_from_rinex(x) for x in edict.keys()]
#    dts = [x[0] for x in g]
#    station = [x[1] for x in g][0]
#    df = pd.DataFrame(data=edict.values(), index=dts)
#    df = df.sort_index()
#    len_er = len(df.columns)
#    df.columns = [station + '_errors_' + str(i) for i in range(len_er)]
#    flat_list = [item for sublist in edict.values() for item in sublist]
#    counted_errors = Counter(flat_list)
#    print(
#        'total files: {}, good runs: {}, bad runs: {}'.format(
#            good +
#            bad,
#            good,
#            bad))
#    errors_sorted = sorted(counted_errors.items(), key=lambda x: x[1],
#                           reverse=True)
#    return errors_sorted, df


def gipsyx_runs_error_analysis(path, glob_str='*.tdp'):
    from collections import Counter
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    import pandas as pd
    
    def find_errors(content_list, name):
        keys = [x for x in content_list if 'KeyError' in x]
        vals = [x for x in content_list if 'ValueError' in x]
        excpt = [x for x in content_list if 'Exception' in x]
        err = [x for x in content_list if 'Error' in x]
        trouble = [x for x in content_list if 'Trouble' in x]
        problem = [x for x in content_list if 'Problem' in x]
        fatal = [x for x in content_list if 'FATAL' in x]
        timed = [x for x in content_list if 'Timed' in x]
        errors = keys + vals + excpt + err + trouble + problem + fatal + timed
        if not errors:
            dt, _ = get_timedate_and_station_code_from_rinex(name)
            print('found new error on {} ({})'.format(name,  dt.strftime('%Y-%m-%d')))
        return errors

    rfns = []
    files = path_glob(path, glob_str, True)
    for file in files:
        # first get all the rinex filenames that gipsyx ran successfuly:
        rfn = file.as_posix().split('/')[-1][0:12]
        rfns.append(rfn)
    if files:
        print('running error analysis for station {}'.format(rfn[0:4].upper()))
    all_errors = []
    errors = []
    dates = []
    rinex = []
    files = path_glob(path, '*.err')
    for file in files:
        rfn = file.as_posix().split('/')[-1][0:12]
        # now, filter the error files that were copyed but there is tdp file
        # i.e., the gipsyx run was successful:
        if rfn in rfns:
            continue
        else:
            dt, _ = get_timedate_and_station_code_from_rinex(rfn)
            dates.append(dt)
            rinex.append(rfn)
            with open(file) as f:
                content = f.readlines()
                # you may also want to remove whitespace characters like `\n` at
                # the end of each line
                content = [x.strip() for x in content]
                all_errors.append(content)
                errors.append(find_errors(content, rfn))
    er = [','.join(x) for x in all_errors]
    df = pd.DataFrame(data=rinex, index=dates, columns=['rinex'])
    df['error'] = er
    df = df.sort_index()
    total = len(rfns) + len(df)
    good = len(rfns)
    bad = len(df)
    print('total files: {}, successful runs: {}, errornous runs: {}'.format(
            total, good, bad))
    print('success percent: {0:.1f}%'.format(100.0 * good / total))
    print('error percent: {0:.1f}%'.format(100.0 * bad / total))
    # now count the similar errors and sort:
    flat_list = [item for sublist in errors for item in sublist]
    counted_errors = Counter(flat_list)
    errors_sorted = sorted(counted_errors.items(), key=lambda x: x[1],
                           reverse=True)
    return errors_sorted, df


def save_yearly_gipsyx_results(path=work_yuval, savepath=work_yuval):
    """call read one station for each year and save the results, then
    concat and save to a bigger raw file, can add postproccess function"""
    from aux_gps import path_glob
    from aux_gps import get_timedate_and_station_code_from_rinex
    files = path_glob(path, '*.tdp')
    rfns = [x.as_posix().split('/')[-1][0:12] for x in files]
    dts = [get_timedate_and_station_code_from_rinex(rfn, just_dt=True) for
           rfn in rfns]
    _, station = get_timedate_and_station_code_from_rinex(rfns[0])
    years = list(set([dt.year for dt in dts]))
    for year in sorted(years):
        filename = '{}_ppp_raw_{}.nc'.format(station, year)
        if (savepath / filename).is_file():
            print('{} already in {}, skipping...'.format(filename, savepath))
            continue
        ds, _ = read_one_station_gipsyx_results(path, savepath, year)
    return


def read_one_station_gipsyx_results(path=work_yuval, savepath=None,
                                    year=None):
    """read one station (all years) consisting of many tdp files"""
#     from scipy import stats
#     import numpy as np
    import xarray as xr
#     import pandas as pd
    from aux_gps import get_timedate_and_station_code_from_rinex
    from aux_gps import path_glob
    if year is not None:
        year = int(year)
        print('getting tdp files from year {}'.format(year))
#    if times is not None:
#        dt_range = pd.date_range(times[0], times[1], freq='1D')
#        print('getting tdp files from {} to {}'.format(times[0], times[1]))
    df_list = []
    errors = []
    dts = []
    print('reading folder:{}'.format(path))
    files = path_glob(path, '*.tdp')
    for tdp_file in files:
        rfn = tdp_file.as_posix().split('/')[-1][0:12]
        dt, station = get_timedate_and_station_code_from_rinex(rfn)
        if year is not None:
            if dt.year != year:
                continue
            else:
                print(rfn)
                try:
                    df, meta = process_one_day_gipsyx_output(tdp_file)
                    dts.append(df.index[0])
                except TypeError:
                    print('problem reading {}, appending to errors...'.format(rfn))
                    errors.append(rfn)
                    continue
                df_list.append(df)
        elif year is None:
            try:
                df, meta = process_one_day_gipsyx_output(tdp_file)
                dts.append(df.index[0])
            except TypeError:
                print('problem reading {}, appending to errors...'.format(rfn))
                errors.append(rfn)
                continue
            df_list.append(df)
    # sort by first dates of each df:
    df_dict = dict(zip(dts, df_list))
    df_list = []
    for key in sorted(df_dict):
        df_list.append(df_dict[key])
    dss = [df.to_xarray() for df in df_list]
    dss_new = []
    for i, ds in enumerate(dss):
        keys_to_rename = [x for x in ds.data_vars.keys()]
        values_to_rename = [x + '-{}'.format(i) for x in keys_to_rename]
        dict_to_rename = dict(zip(keys_to_rename, values_to_rename))
        dss_new.append(ds.rename(dict_to_rename))
    ds = xr.merge(dss_new)
#    # concat and sort:
#    df_all = pd.concat(df_list)
#    df_all = df_all.sort_index()
#    df_all.index.name = 'time'
#    # filter out negative values:
#    df_all = df_all[df_all > 0]
#    # filter outlies (zscore>3):
#    df_all = df_all[(np.abs(stats.zscore(df_all)) < 3).all(axis=1)]
#    # filter out constant values:
#    df_all['value_grp'] = df_all.zwd.diff(1)
#    df_all = df_all[np.abs(df_all['value_grp']) > 1e-7]
#    ds = df_all.to_xarray()
#    ds = ds.drop('value_grp')
    ds.attrs['station'] = station
#    ds.attrs['lat'] = meta['lat']
#    ds.attrs['lon'] = meta['lon']
#    ds.attrs['alt'] = meta['alt']
#     ds.attrs['units'] = 'cm'
#    if plot:
#        ax = df_all['zwd'].plot(legend=True, figsize=(12, 7), color='k')
#        ax.fill_between(df_all.index, df_all['zwd'] - df_all['error'],
#                        df_all['zwd'] + df_all['error'], alpha=0.5)
#        ax.grid()
#        ax.set_title('Zenith Wet Delay')
#        ax.set_ylabel('[cm]')
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        # ymin = ds.time.min().dt.year.item()
        # ymax = ds.time.max().dt.year.item()
        filename = '{}_ppp_raw_{}.nc'.format(station, year)
        ds.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('{} was saved to {}'.format(filename, savepath))
    return ds, errors


def process_one_day_gipsyx_output(path_and_file=work_yuval / 'smoothFinal.tdp',
                                  plot=False):
    import pandas as pd
    # import pyproj
    import matplotlib.pyplot as plt
    from aux_gps import get_latlonalt_error_from_geocent_error
    df_raw = pd.read_fwf(path_and_file, header=None)
    # get all the vars from smoothFinal.tdp file and put it in a df_list:
    keys = ['WetZ', 'GradNorth', 'GradEast', 'Pos.X', 'Pos.Y', 'Pos.Z']
    df_list = [df_raw[df_raw.iloc[:, -1].str.contains(x)] for x in keys]
    # make sure that all keys in df have the same length:
    assert len(set([len(x) for x in df_list])) == 1
    # translate the seconds col to datetime:
    seconds = df_list[0].iloc[:, 0]
    dt = pd.to_datetime('2000-01-01T12:00:00')
    time = dt + pd.to_timedelta(seconds, unit='sec')
    # build a new df that contains all the vars(from keys):
    ppp = pd.DataFrame(index=time)
    ppp.index.name = 'time'
    # time.set_index(time, inplace=True)
    # df_zwd.index.name = 'time'
    for i, df in enumerate(df_list):
        df.columns = ['seconds', 'to_drop', keys[i], keys[i] + '_error',
                      'meta']
        ppp[keys[i]] = df[keys[i]].values
        ppp[keys[i] + '_error'] = df[keys[i] + '_error'].values
    # rename all the Pos. to nothing:
    ppp.columns = ppp.columns.str.replace('Pos.', '')
    lon, lat, alt, lon_error, lat_error, alt_error = get_latlonalt_error_from_geocent_error(
        ppp.X.values, ppp.Y.values, ppp.Z.values, ppp.X_error.values, ppp.Y_error.values, ppp.Z_error.values,)
    vals = [lon, lat, alt]
    vals_error = [lon_error, lat_error, alt_error]
    for i, key in enumerate(['lon', 'lat', 'alt']):
        ppp[key] = vals[i]
        ppp[key + '_error'] = vals_error[i]
    # get initial lat, lon, alt for meta data purpose:
#    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
#    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
#    lon, lat, alt = pyproj.transform(ecef, lla, ppp.X[0], ppp.Y[0],
#                                     ppp.Z[0], radians=False)
    desc = ['Zenith Wet Delay', 'North Gradient of Zenith Wet Delay',
            'East Gradient of Zenith Wet Delay', 'Longitude', 'Latitude',
            'Altitude']
    units = ['[cm]', '[cm / m]', '[cm / m]', 'Degrees', 'Degrees', '[m]']
    fields = ['WetZ', 'GradNorth', 'GradEast', 'lon', 'lat', 'alt']
    units_dict = dict(zip(fields, units))
    desc_dict = dict(zip(fields, desc))
    meta = {'units': units_dict, 'desc': desc_dict}
    # convert tropospheric products to cm, rest stay in meters:
    trop_cols = ppp.columns.values[0:6]
    ppp[trop_cols] = ppp[trop_cols].mul(100.0)
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

        for ax, field, name, unit in zip(axes.flatten(), fields, desc, units):
            ppp[field].plot(ax=ax, legend=True, color='k')
            ax.fill_between(ppp.index, ppp[field] - ppp[field + '_error'],
                            ppp[field] + ppp[field + '_error'], alpha=0.5)
            ax.grid()
            ax.set_title(name)
            ax.set_ylabel(unit)
    return ppp, meta
