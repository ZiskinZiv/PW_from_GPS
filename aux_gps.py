#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:33:19 2019

@author: ziskin
"""
from PW_paths import work_yuval


def xr_reindex_with_date_range(ds, time_dim='time', freq='5min'):
    import pandas as pd
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
        ds[field] = ds[field].where(np.isfinite(ds[field+error_str])).dropna(dim)
        if meta in ds[field].attrs:
            append = True
        add_attr_to_xr(ds[field], meta, ', filtered values with NaN errors', append)
    return ds


def keep_iqr(ds, dim='time', qlow=0.25, qhigh=0.75, k=1.5):
    """return the data in a dataset or dataarray only in the
    Interquartile Range (low, high)"""
    import xarray as xr

    def keep_iqr_da(da, dim, qlow, qhigh, meta='action'):
        from aux_gps import add_attr_to_xr
        quan = da.quantile([qlow, qhigh], dim).values
        low = quan[0]
        high = quan[1]
        iqr = high - low
        lower = low - (iqr * k)
        higher = high + (iqr * k)
        da = da.where((da < higher) & (da > lower)).dropna(dim)
        if meta in da.attrs:
            append = True
        add_attr_to_xr(da, meta, ', kept IQR ({}, {}, {})'.format(qlow, qhigh, k), append)
        return da
    if isinstance(ds, xr.DataArray):
        iqr = keep_iqr_da(ds, dim, qlow, qhigh)
    elif isinstance(ds, xr.Dataset):
        da_list = []
        for name in ds.data_vars:
            da = keep_iqr_da(ds[name], dim, qlow, qhigh)
            da_list.append(da)
        iqr = xr.merge(da_list)
    return iqr


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


def get_latlonalt_error_from_geocent_error(X, Y, Z, xe, ye, ze):
    """returns the value and error in lat(decimal degree), lon(decimal degree)
    and alt(meters) for X, Y, Z in geocent coords (in meters), all input is
    lists or np.arrays"""
    import pyproj
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=False)
    lon_pe, lat_pe, alt_pe = pyproj.transform(ecef, lla, X + xe, Y + ye,
                                              Z + ze, radians=False)
    lon_me, lat_me, alt_me = pyproj.transform(ecef, lla, X - xe, Y - ye,
                                              Z - ze, radians=False)
    lon_e = (lon_pe - lon_me) / 2.0
    lat_e = (lat_pe - lat_me) / 2.0
    alt_e = (alt_pe - alt_me) / 2.0
    return lon, lat, alt, lon_e, lat_e, alt_e


def path_glob(path, glob_str='*.Z', return_empty_list=False):
    """returns all the files with full path(pathlib3 objs) if files exist in
    path, if not, returns FilenotFoundErro"""
    from pathlib import Path
    if not isinstance(path, Path):
        raise Exception('{} must be a pathlib object'.format(path))
    files_with_path = [file for file in path.glob(glob_str) if file.is_file]
    if not files_with_path and not return_empty_list:
        raise Exception('{} search in {} found no files.'.format(glob_str,
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


def datetime_to_rinex_filename(station='tela', dt='2012-05-07'):
    """return rinex filename from datetime string"""
    import pandas as pd
    day = pd.to_datetime(dt, format='%Y-%m-%d').dayofyear
    year = pd.to_datetime(dt, format='%Y-%m-%d').year
    if len(str(day)) == 1:
        str_day = '00' + str(day) + '0'
    elif len(str(day)) == 2:
        str_day = '0' + str(day) + '0'
    elif len(str(day)) == 3:
        str_day = str(day) + '0'
    filename = station.lower() + str_day + '.' + str(year)[2:4] + 'd'
    return filename


def get_timedate_and_station_code_from_rinex(rinex_str='tela0010.05d',
                                             just_dt=False):
    """return datetime from rinex2 format"""
    import pandas as pd
    import datetime
    station = rinex_str[0:4]
    days = int(rinex_str[4:7])
    year = rinex_str[-3:-1]
    Year = datetime.datetime.strptime(year, '%y').strftime('%Y')
    dt = datetime.datetime(int(Year), 1, 1) + datetime.timedelta(days - 1)
    if just_dt:
        return pd.to_datetime(dt)
    else:
        return pd.to_datetime(dt), station.upper()


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
