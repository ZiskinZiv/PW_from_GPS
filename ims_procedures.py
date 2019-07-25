#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:22:51 2019

@author: ziskin
"""
from PW_startup import *
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
ims_10mins_path = ims_path / '10mins'


def parse_cv_results(grid_search_cv):
    from aux_gps import process_gridsearch_results
    """parse cv_results from GridsearchCV object"""
    # only supports neg-abs-mean-error with leaveoneout
    from sklearn.model_selection import LeaveOneOut
    if (isinstance(grid_search_cv.cv, LeaveOneOut)
            and grid_search_cv.scoring == 'neg_mean_absolute_error'):

        cds = process_gridsearch_results(grid_search_cv)
        cds = - cds
    return cds


def Interpolating_models_ims(time='2013-10-19T22:00:00', var='TD', plot=True,
                             gis_path=gis_path, method='kriging',
                             dem_path=work_yuval / 'AW3D30', lapse_rate=5.,
                             cv=None, rms=None, gridsearch=False):
    """main 2d_interpolation from stations to map"""
    # cv usage is {'kfold': 5} or {'rkfold': [2, 3]}
    # TODO: try 1d modeling first, like T=f(lat)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from pykrige.rk import Krige
    import numpy as np
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from scipy.spatial import Delaunay
    from scipy.interpolate import griddata
    from sklearn.metrics import mean_squared_error
    from aux_gps import coarse_dem
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pyproj
    from sklearn.utils.estimator_checks import check_estimator
    from pykrige.compat import GridSearchCV
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')

    def parse_cv(cv):
        from sklearn.model_selection import KFold
        from sklearn.model_selection import RepeatedKFold
        from sklearn.model_selection import LeaveOneOut
        """input:cv number or string"""
        # check for integer:
        if 'kfold' in cv.keys():
            n_splits = cv['kfold']
            print('CV is KFold with n_splits={}'.format(n_splits))
            return KFold(n_splits=n_splits)
        if 'rkfold' in cv.keys():
            n_splits = cv['rkfold'][0]
            n_repeats = cv['rkfold'][1]
            print('CV is ReapetedKFold with n_splits={},'.format(n_splits) +
                  ' n_repeates={}'.format(n_repeats))
            return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=42)
        if 'loo' in cv.keys():
            return LeaveOneOut()
    # from aux_gps import scale_xr
    da = create_lat_lon_mesh(points_per_degree=500)
    awd = coarse_dem(da)
    awd = awd.values
    geo_snap = geo_pandas_time_snapshot(var=var, datetime=time, plot=False)
    if var == 'TD':
        [a, b] = np.polyfit(geo_snap['alt'].values, geo_snap['TD'].values, 1)
        if lapse_rate == 'auto':
            lapse_rate = np.abs(a) * 1000
        fig, ax_lapse = plt.subplots(figsize=(10, 6))
        sns.regplot(data=geo_snap, x='alt', y='TD', color='r',
                    scatter_kws={'color': 'b'}, ax=ax_lapse)
        suptitle = time.replace('T', ' ')
        ax_lapse.set_xlabel('Altitude [m]')
        ax_lapse.set_ylabel('Temperature [degC]')
        ax_lapse.text(0.5, 0.95, 'Lapse_rate: {:.2f} degC/km'.format(lapse_rate),
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_lapse.transAxes, fontsize=12, color='k',
                      fontweight='bold')
        ax_lapse.grid()
        ax_lapse.set_title(suptitle, fontsize=14, fontweight='bold')
#     fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    alts = []
    for i, row in geo_snap.iterrows():
        lat = da.sel(lat=row['lat'], method='nearest').lat.values
        lon = da.sel(lon=row['lon'], method='nearest').lon.values
        alt = row['alt']
        if lapse_rate is not None and var == 'TD':
            da.loc[{'lat': lat, 'lon': lon}] = row[var] + \
                lapse_rate * alt / 1000.0
            alts.append(alt)
        elif lapse_rate is None or var != 'TD':
            da.loc[{'lat': lat, 'lon': lon}] = row[var]
            alts.append(alt)
    # da_scaled = scale_xr(da)
    c = np.linspace(min(da.lat.values), max(da.lat.values), da.shape[0])
    r = np.linspace(min(da.lon.values), max(da.lon.values), da.shape[1])
    rr, cc = np.meshgrid(r, c)
    vals = ~np.isnan(da.values)
    if lapse_rate is None:
        Xrr, Ycc, Z = pyproj.transform(
                lla, ecef, rr[vals], cc[vals], np.array(alts), radians=False)
        X = np.column_stack([Xrr, Ycc, Z])
        XX, YY, ZZ = pyproj.transform(lla, ecef, rr, cc, awd.values,
                                      radians=False)
        rr_cc_as_cols = np.column_stack([XX.flatten(), YY.flatten(),
                                         ZZ.flatten()])
    else:
        X = np.column_stack([rr[vals], cc[vals]])
        rr_cc_as_cols = np.column_stack([rr.flatten(), cc.flatten()])
    # y = da_scaled.values[vals]
    y = da.values[vals]
    if method == 'gp-rbf':
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.gaussian_process.kernels import WhiteKernel
        kernel = 1.0 * RBF(length_scale=0.25, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
#        kernel = None
        model = GaussianProcessRegressor(alpha=0.0, kernel=kernel,
                                         n_restarts_optimizer=5,
                                         random_state=42, normalize_y=True)

    elif method == 'gp-qr':
        from sklearn.gaussian_process.kernels import RationalQuadratic
        from sklearn.gaussian_process.kernels import WhiteKernel
        kernel = RationalQuadratic(length_scale=100.0) \
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e+1))
        model = GaussianProcessRegressor(alpha=0.0, kernel=kernel,
                                         n_restarts_optimizer=5,
                                         random_state=42, normalize_y=True)
    elif method == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    elif method == 'svr':
        model = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                    gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                    shrinking=True, tol=0.001, verbose=False)
    elif method == 'okrig':
        model = Krige(method='ordinary', variogram_model='spherical',
                      verbose=True)
    elif method == 'ukrig':
        model = Krige(method='universal', variogram_model='linear',
                      verbose=True)
#    elif method == 'okrig3d':
#        # don't bother - MemoryError...
#        model = OrdinaryKriging3D(rr[vals], cc[vals], np.array(alts),
#                                  da.values[vals], variogram_model='linear',
#                                  verbose=True)
#        awd = coarse_dem(da)
#        interpolated, ss = model.execute('grid', r, c, awd['data'].values)
#    elif method == 'rkrig':
#        # est = LinearRegression()
#        est = RandomForestRegressor()
#        model = RegressionKriging(regression_model=est, n_closest_points=5,
#                                  verbose=True)
#        p = np.array(alts).reshape(-1, 1)
#        model.fit(p, X, y)
#        P = awd.flatten().reshape(-1, 1)
#        interpolated = model.predict(P, rr_cc_as_cols).reshape(da.values.shape)
#    try:
#        u = check_estimator(model)
#    except TypeError:
#        u = False
#        pass
    if cv is not None and not gridsearch:  # and u is None):
        # from sklearn.model_selection import cross_validate
        from sklearn import metrics
        cv = parse_cv(cv)
        ytests = []
        ypreds = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]  # requires arrays
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # there is only one y-test and y-pred per iteration over the loo.split,
            # so to get a proper graph, we append them to respective lists.
            ytests += list(y_test)
            ypreds += list(y_pred)
        true_vals = np.array(ytests)
        predicted = np.array(ypreds)
        r2 = metrics.r2_score(ytests, ypreds)
        ms_error = metrics.mean_squared_error(ytests, ypreds)
        print("R^2: {:.5f}%, MSE: {:.5f}".format(r2*100, ms_error))
    if gridsearch:
        cv = parse_cv(cv)
        param_dict = {"method": ["ordinary", "universal"],
                      "variogram_model": ["linear", "power", "gaussian",
                                          "spherical"],
                      # "nlags": [4, 6, 8],
                      # "weight": [True, False]
                      }
        estimator = GridSearchCV(Krige(), param_dict, verbose=True, cv=cv,
                                 scoring='neg_mean_absolute_error',
                                 return_train_score=True, n_jobs=1)
        estimator.fit(X, y)
        if hasattr(estimator, 'best_score_'):
            print('best_score = {:.3f}'.format(estimator.best_score_))
            print('best_params = ', estimator.best_params_)
        
        return estimator
#    if (cv is not None and not u):
#        from sklearn import metrics
#        cv = parse_cv(cv)
#        ytests = []
#        ypreds = []
#        for train_idx, test_idx in cv.split(X):
#            X_train, X_test = X[train_idx], X[test_idx]  # requires arrays
#            y_train, y_test = y[train_idx], y[test_idx]
##            model = UniversalKriging(X_train[:, 0], X_train[:, 1], y_train,
##                                     variogram_model='linear', verbose=False,
##                                     enable_plotting=False)
#            model.X_ORIG = X_train[:, 0]
#            model.X_ADJUSTED = model.X_ORIG
#            model.Y_ORIG = X_train[:, 1]
#            model.Y_ADJUSTED = model.Y_ORIG
#            model.Z = y_train
#            y_pred, ss = model.execute('points', X_test[0, 0],
#                                             X_test[0, 1])
#            # there is only one y-test and y-pred per iteration over the loo.split,
#            # so to get a proper graph, we append them to respective lists.
#            ytests += list(y_test)
#            ypreds += list(y_pred)
#        true_vals = np.array(ytests)
#        predicted = np.array(ypreds)
#        r2 = metrics.r2_score(ytests, ypreds)
#        ms_error = metrics.mean_squared_error(ytests, ypreds)
#        print("R^2: {:.5f}%, MSE: {:.5f}".format(r2*100, ms_error))
#        cv_results = cross_validate(gp, X, y, cv=cv, scoring='mean_squared_error',
#                                    return_train_score=True, n_jobs=-1)
#        test = xr.DataArray(cv_results['test_score'], dims=['kfold'])
#        train = xr.DataArray(cv_results['train_score'], dims=['kfold'])
#        train.name = 'train'
#        cds = test.to_dataset(name='test')
#        cds['train'] = train
#        cds['kfold'] = np.arange(len(cv_results['test_score'])) + 1
#        cds['mean_train'] = cds.train.mean('kfold')
#        cds['mean_test'] = cds.test.mean('kfold')

    # interpolated=griddata(X, y, (rr, cc), method='nearest')
    model.fit(X, y)
    interpolated = model.predict(rr_cc_as_cols).reshape(da.values.shape)
    da_inter = da.copy(data=interpolated)
    if lapse_rate is not None and var == 'TD':
        da_inter -= lapse_rate * awd / 1000.0
    if (rms is not None and cv is None):  # or (rms is not None and not u):
        predicted = []
        true_vals = []
        for i, row in geo_snap.iterrows():
            lat = da.sel(lat=row['lat'], method='nearest').lat.values
            lon = da.sel(lon=row['lon'], method='nearest').lon.values
            pred = da_inter.loc[{'lat': lat, 'lon': lon}].values.item()
            true = row[var]
            predicted.append(pred)
            true_vals.append(true)
        predicted = np.array(predicted)
        true_vals = np.array(true_vals)
        ms_error = mean_squared_error(true_vals, predicted)
        print("MSE: {:.5f}".format(ms_error))
    if plot:
        import salem
        from salem import DataLevels, Map
        import cartopy.crs as ccrs
        # import cartopy.io.shapereader as shpreader
        import matplotlib.pyplot as plt
        # fname = gis_path / 'ne_10m_admin_0_sovereignty.shp'
        # fname = gis_path / 'gadm36_ISR_0.shp'
        # ax = plt.axes(projection=ccrs.PlateCarree())
        f, ax = plt.subplots(figsize=(6, 10))
        # shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
        shdf = salem.read_shapefile(gis_path / 'Israel_and_Yosh.shp')
        # shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Israel']  # remove other countries
        shdf.crs = {'init': 'epsg:4326'}
        dsr = da_inter.salem.roi(shape=shdf)
        grid = dsr.salem.grid
        grid = da_inter.salem.grid
        sm = Map(grid)
        # sm.set_shapefile(gis_path / 'Israel_and_Yosh.shp')
        # sm = dsr.salem.quick_map(ax=ax)
#        sm2 = salem.Map(grid, factor=1)
#        sm2.set_shapefile(gis_path/'gis_osm_water_a_free_1.shp',
#                          edgecolor='k')
        sm.set_data(dsr)
        # sm.set_nlevels(7)
        # sm.visualize(ax=ax, title='Israel {} interpolated temperature from IMS'.format(method),
        #             cbar_title='degC')
        sm.set_shapefile(gis_path/'gis_osm_water_a_free_1.shp',
                         edgecolor='k')  # , facecolor='aqua')
        # sm.set_topography(awd.values, crs=awd.crs)
        # sm.set_rgb(crs=shdf.crs, natural_earth='hr')  # ad
        # lakes = salem.read_shapefile(gis_path/'gis_osm_water_a_free_1.shp')
        sm.set_cmap(cm='rainbow')
        sm.visualize(ax=ax, title='Israel {} interpolated temperature from IMS'.format(method),
                     cbar_title='degC')
        dl = DataLevels(geo_snap[var], levels=sm.levels)
        dl.set_cmap(sm.cmap)
        x, y = sm.grid.transform(geo_snap.lon.values, geo_snap.lat.values)
        ax.scatter(x, y, color=dl.to_rgb(), s=20, edgecolors='k', linewidths=0.5)
        suptitle = time.replace('T', ' ')
        f.suptitle(suptitle, fontsize=14, fontweight='bold')
        if (rms is not None or cv is not None) and (not gridsearch):
            import seaborn as sns
            f, ax = plt.subplots(1, 2, figsize=(12, 6))
            sns.scatterplot(x=true_vals, y=predicted, ax=ax[0], marker='.',
                            s=100)
            resid = predicted - true_vals
            sns.distplot(resid, bins=5, color='c', label='residuals',
                         ax=ax[1])
            rmean = np.mean(resid)
            rstd = np.std(resid)
            rmedian = np.median(resid)
            rmse = np.sqrt(mean_squared_error(true_vals, predicted))
            plt.axvline(rmean, color='r', linestyle='dashed', linewidth=1)
            _, max_ = plt.ylim()
            plt.text(rmean + rmean / 10, max_ - max_ / 10,
                     'Mean: {:.2f}, RMSE: {:.2f}'.format(rmean, rmse))
            f.tight_layout()
        # lakes.plot(ax=ax, color='b', edgecolor='k')
        # lake_borders = gpd.overlay(countries, capitals, how='difference')
        # adm1_shapes = list(shpreader.Reader(fname).geometries())
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # ax.coastlines(resolution='10m')
        # ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
        #                  edgecolor='black', facecolor='gray', alpha=0.5)
        # da_inter.plot.pcolormesh('lon', 'lat', ax=ax)
        #geo_snap.plot(ax=ax, column=var, cmap='viridis', edgecolor='black',
        #              legend=False)
    return da_inter


def create_lat_lon_mesh(lats=[29.5, 33.5], lons=[34, 36],
                        points_per_degree=1000):
    import xarray as xr
    import numpy as np
    lat = np.arange(lats[0], lats[1], 1.0 / points_per_degree)
    lon = np.arange(lons[0], lons[1], 1.0 / points_per_degree)
    nans = np.nan * np.ones((len(lat), len(lon)))
    da = xr.DataArray(nans, dims=['lat', 'lon'])
    da['lat'] = lat
    da['lon'] = lon
    return da


def read_save_ims_10mins(path=ims_10mins_path, var='TD'):
    import xarray as xr
    search_str = '*' + var + '_10mins.nc'
    da_list = []
    for file_and_path in path.glob(search_str):
        da = xr.open_dataarray(file_and_path)
        print('reading ims 10mins {} data for {} station'.format(var, da.name))
        da_list.append(da)
    print('merging...')
    ds = xr.merge(da_list)
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    filename = 'ims_' + var + '_10mins.nc'
    print('saving...')
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('{} was saved to {}.'.format(filename, path))
    return ds


def analyse_10mins_ims_field(path=ims_10mins_path, var='TD',
                             gis_path=gis_path, dem_path=work_yuval/ 'AW3D30'):
    import xarray as xr
    import collections
    import numpy as np
    # TODO: make 2d histogram of stations by altitude and time...
    awd = xr.open_rasterio(dem_path / 'israel_dem.tif')
    awd = awd.squeeze(drop=True)
    filename = 'ims_' + var + '_10mins.nc'
    ds = xr.open_dataset(path / filename)
    meta = read_ims_metadata_from_files(path=gis_path,
                                        filename='IMS_10mins_meta_data.xlsx')
    meta.index = meta.ID.astype('int')
    meta.drop('ID', axis=1, inplace=True)
    meta.sort_index(inplace=True)
    # there are some stations with the same altitude, i'm mapping them:
    duplicate_alts = [item for item, count in collections.Counter(
                        meta['alt']).items() if count > 1]
    print(duplicate_alts)
    # then replacing them with a 1-meter seperations:
    for dup in duplicate_alts:
        dup_size = len(meta.loc[meta['alt'] == dup, 'alt'])
        start_value = meta.loc[meta['alt'] == dup, 'alt'].values[0]
        replace_values = np.arange(start_value, start_value + dup_size)
        print(
                'duplicate {} has {} values, replacing with {}'.format(
                        dup,
                        dup_size,
                        replace_values))
        meta.loc[meta['alt'] == dup, 'alt'] = replace_values
    for da in ds.data_vars.keys():
        id_ = ds[da].attrs['station_id']
        try:
            lat = meta.loc[id_, 'lat']
            lon = meta.loc[id_, 'lon']
            alt = meta.loc[id_, 'alt']
        except KeyError:
            lat = ds[da].attrs['station_lat']
            lon = ds[da].attrs['station_lon']
            print('station {} keyerror.'.format(da))
            alt = 'None'
        try:
            alt = awd.sel(x=float(lon), y=float(lat), method='nearest').values.item()
        except ValueError:
            print('station {} has not known lat or lon...'.format(ds[da].attrs['station_name']))
        ds[da].attrs['station_lat'] = lat
        ds[da].attrs['station_lon'] = lon
        ds[da].attrs['station_alt'] = alt
    return ds


def geo_pandas_time_snapshot(path=ims_10mins_path, var='TD',
                             datetime='2013-10-19T10:00:00',
                             gis_path=gis_path, plot=True):
    import xarray as xr
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    # TODO: add simple df support
    # first, read ims_10mins data for choice var:
    filename = 'ims_' + var + '_10mins.nc'
    ds = xr.open_dataset(path / filename)
    ds = ds.sel(time=datetime)
    meta = read_ims_metadata_from_files(path=gis_path,
                                        filename='IMS_10mins_meta_data.xlsx')
    meta.index = meta.ID.astype('int')
    meta.drop('ID', axis=1, inplace=True)
    meta.sort_index(inplace=True)
    cols_list = []
    for dvar in ds.data_vars.values():
        value = dvar.values.item()
        id_ = dvar.attrs['station_id']
        try:
            lat = meta.loc[id_, 'lat']
            lon = meta.loc[id_, 'lon']
            alt = meta.loc[id_, 'alt']
        except KeyError:
            lat = dvar.attrs['station_lat']
            lon = dvar.attrs['station_lon']
            alt = None
        name = dvar.name
        var_ = dvar.attrs['channel_name']
        cols = [pd.to_datetime(datetime), name, id_, lat, lon, alt,
                var_, value]
        cols_list.append(cols)
    df = pd.DataFrame(cols_list)
    df.columns = ['time', 'name', 'id', 'lat', 'lon', 'alt', 'var_name', var_]
    df.dropna(inplace=True)
    df = df.astype({'lat': 'float64', 'lon': 'float64'})
    # geopandas part:
    isr = gpd.read_file(gis_path / 'Israel_demog_yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    geo_snap = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                                df.lat),
                                crs=isr.crs)
    if plot:
        ax = isr.plot()
        geo_snap.plot(ax=ax, column=var_, cmap='viridis', edgecolor='black',
                      legend=True)
        plt.title(var_ + ' in ' + datetime)
    return geo_snap


def get_meta_data_hourly_ims_climate_database(ds):
    import pandas as pd
    name_list = []
    for name, da in ds.data_vars.items():
        data = [name.split('_')[0], da.attrs['station_id'], da.attrs['lat'],
                da.attrs['lon'], da.attrs['height']]
        name_list.append(data)
    df = pd.DataFrame(name_list)
    df.columns = ['name', 'id', 'lat', 'lon', 'height']
    return df


def proccess_hourly_ims_climate_database(path=ims_path, var='tas',
                                         times=('1996', '2019')):
    import xarray as xr
    import numpy as np
    ds = xr.open_dataset(path / 'hourly_ims.nc')
    if var is not None:
        ds = ds.sel({'var': var})
        print('selecting {} variables'.format(var))
        if times is not None:
            print('selecting times from {} to {}'.format(times[0], times[1]))
            ds = ds.sel(time=slice(times[0], times[1]))
            to_drop_list = []
            for name, da in ds.data_vars.items():
                if (np.isnan(da) == True).all().item():
                    to_drop_list.append(name)
            ds = ds.drop(to_drop_list)
    return ds


def read_hourly_ims_climate_database(path=ims_path / 'ground',
                                     savepath=None):
    """downloaded from tau...ds is a dataset of all stations,
    times is a time period"""
    import pandas as pd
    import xarray as xr
    from aux_gps import print_saved_file
    da_list = []
    for file in sorted(path.glob('*.csv')):
        name = file.as_posix().split('/')[-1].split('_')[0]
        sid = file.as_posix().split('/')[-1].split('_')[1]
        array_name = '_'.join([name, sid])
        print('reading {} station...'.format(array_name))
        df = pd.read_csv(file, index_col='time')
        df.index = pd.to_datetime(df.index)
        df.drop(labels=['Unnamed: 0', 'name'], axis=1, inplace=True)
        lat = df.loc[:, 'lat'][0]
        lon = df.loc[:, 'lon'][0]
        height = df.loc[:, 'height'][0]
        df.drop(labels=['lat', 'lon', 'height'], axis=1, inplace=True)
        da = df.to_xarray().to_array(dim='var')
        da.name = array_name
        da.attrs['station_id'] = sid
        da.attrs['lat'] = lat
        da.attrs['lon'] = lon
        da.attrs['height'] = height
        da_list.append(da)
    ds = xr.merge(da_list)
    print('Done!')
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(savepath / 'hourly_ims.nc', 'w', encoding=encoding)
        print_saved_file('hourly_ims.nc', savepath)
    return ds


def read_ims_metadata_from_files(path=gis_path,
                                 filename='IMS_10mins_meta_data.xlsx'):
    # for longer climate archive data use filename = IMS_climate_archive_meta_data.xls
    import pandas as pd
    """parse ims stations meta-data"""
    if '10mins' in filename:
        ims = pd.read_excel(path / filename,
                            sheet_name='מטה-דטה', skiprows=1)
        # drop two last cols and two last rows:
        ims = ims.drop(ims.columns[[-1, -2]], axis=1)
        ims = ims.drop(ims.tail(2).index)
        cols = ['#', 'ID', 'name_hebrew', 'name_english', 'east', 'west',
                'lon', 'lat', 'alt', 'starting_date', 'variables', 'model',
                'eq_position', 'wind_meter_height', 'notes']
        ims.columns = cols
        ims.index = ims['#'].astype(int)
        ims = ims.drop('#', axis=1)
        # fix lat, lon cols:
        ims['lat'] = ims['lat'].str.replace(u'\xba', '').astype(float)
        ims['lon'] = ims['lon'].str.replace(u'\xba', '').astype(float)
        # fix alt col:
        ims['alt'] = ims['alt'].replace('~', '', regex=True).astype(float)
        # fix starting date col:
        ims['starting_date'] = pd.to_datetime(ims['starting_date'])
    else:
        ims = pd.read_excel(path + filename,
                            sheet_name='תחנות אקלים', skiprows=1)
        cols = ['ID', 'name_hebrew', 'name_english', 'station_type', 'east',
                'west', 'lon', 'lat', 'alt', 'starting_date', 'closing_date',
                'date_range']
        ims.columns = cols
        # ims.index = ims['ID'].astype(int)
        # ims = ims.drop('ID', axis=1)
        # fix lat, lon cols:
        ims['lat'] = ims['lat'].str.replace(u'\xba', '').astype(float)
        ims['lon'] = ims['lon'].str.replace(u'\xba', '').astype(float)
        # fix alt col:
        ims['alt'] = ims['alt'].replace('~', '', regex=True).astype(float)
        # fix starting date, closing_date col:
        ims['starting_date'] = pd.to_datetime(ims['starting_date'])
        ims['closing_date'] = pd.to_datetime(ims['closing_date'])
    return ims


def produce_geo_ims(path, filename='IMS_10mins_meta_data.xlsx',
                    closed_stations=False, plot=True):
    import geopandas as gpd
    import numpy as np
    isr = gpd.read_file(path / 'israel_demog2012.shp')
    isr.crs = {'init': 'epsg:4326'}
    ims = read_ims_metadata_from_files(path, filename)
    if closed_stations:
        ims = ims[np.isnat(ims.closing_date)]
    geo_ims = gpd.GeoDataFrame(ims, geometry=gpd.points_from_xy(ims.lon,
                                                                ims.lat),
                               crs=isr.crs)
    if plot:
        ax = isr.plot()
        geo_ims.plot(ax=ax, column='alt', cmap='Reds', edgecolor='black',
                     legend=True)
    return geo_ims


def ims_api_get_meta(active_only=True, channel_name='TD'):
    import requests
    import pandas as pd
    """get meta data on 10mins ims stations"""
    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_10mins = pd.DataFrame(r.json())
    # filter inactive stations:
    if active_only:
        stations_10mins = stations_10mins[stations_10mins.active]
    # arrange lat lon nicely and add channel num for dry temp:
    lat_ = []
    lon_ = []
    channelId_list = []
    for index, row in stations_10mins.iterrows():
        lat_.append(row['location']['latitude'])
        lon_.append(row['location']['longitude'])
        channel = [x['channelId'] for x in row.monitors if x['name'] ==
                   channel_name]
        if channel:
            channelId_list.append(channel[0])
        else:
            channelId_list.append(None)
    stations_10mins['lat'] = lat_
    stations_10mins['lon'] = lon_
    stations_10mins[channel_name + '_channel'] = channelId_list
    stations_10mins.drop(['location', 'StationTarget', 'stationsTag'],
                         axis=1, inplace=True)
    return stations_10mins


def download_ims_single_station(stationid, savepath=ims_path,
                                channel_name='TD'):
    """download single station with channel_name from earliest to latest.
    if chanel_name is None, download all channels"""
    import requests
    import pandas as pd
    # TODO: add all channels download...

    def parse_ims_to_df(raw_data, ch_name):
        """gets ims station raw data, i.e., r.json()['data'] and returns
        a pandas dataframe"""
        import pandas as pd
        if ch_name is not None:
            datetimes = [x['datetime'] for x in raw_data]
            data = [x['channels'][0] for x in raw_data]
            df = pd.DataFrame.from_records(data,
                                           index=pd.to_datetime(datetimes,
                                                                utc=True))
            df.drop(['alias', 'description'], axis=1, inplace=True)
            cols = [ch_name + '_' + x for x in df.columns]
            df.columns = cols
        elif ch_name is None:
            # add all channels d/l here:
            datetimes = [x['datetime'] for x in raw_data]
            names = [x['name'] for x in data['channels']]
            keys = [*data['channels'][0].keys()]
        return df

    def to_dataarray(df, meta):
        # add all channels d/l here:
        import pandas as pd
        ds = df.to_xarray()
        ds['time'] = pd.to_datetime(ds.time)
        channel_name = [*ds.data_vars.keys()][0].split('_')[0]
        channel_id = ds[channel_name + '_id'].isel(time=0).values.item()
        to_drop = [x for x in ds.data_vars.keys() if 'value' not in x]
        ds = ds.drop(to_drop)
        da = ds[channel_name + '_value'].reset_coords(drop=True)
        da.name = meta['name']
        da.attrs['channel_id'] = int(channel_id)
        da.attrs['channel_name'] = channel_name
        da.attrs['station_name'] = meta['name']
        da.attrs['station_id'] = meta['id']
        da.attrs['active'] = meta['active']
        da.attrs['station_lat'] = str(meta['loc']['latitude'])
        da.attrs['station_lon'] = str(meta['loc']['longitude'])
        for key, value in da.attrs.items():
            print(key, value)
        return da

    def get_dates_list(start_date, end_date):
        """divide the date span into full 1 years and a remainder, tolist"""
        import numpy as np
        import pandas as pd
        end_date = pd.to_datetime(end_date)
        start_date = pd.to_datetime(start_date)
        s_year = start_date.year
        e_year = end_date.year
        years = np.arange(s_year, e_year + 1)
        dates = [start_date.replace(year=x) for x in years]
        if (end_date - dates[-1]).days > 0:
            dates.append(end_date)
        return dates

    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
    headers = {'Authorization': 'ApiToken ' + myToken}
    r = requests.get('https://api.ims.gov.il/v1/envista/stations/',
                     headers=headers)
    stations_10mins = pd.DataFrame(r.json())
    meta = {}
    st_name = stations_10mins['name'].where(
            stations_10mins['stationId'] == stationid).dropna().item()
    location = stations_10mins['location'].where(
            stations_10mins['stationId'] == stationid).dropna().item()
    active = stations_10mins['active'].where(
            stations_10mins['stationId'] == stationid).dropna().item()
    meta['name'] = '-'.join(st_name.split(' '))
    meta['id'] = stationid
    meta['loc'] = location
    meta['active'] = active
    r_early = requests.get('https://api.ims.gov.il/v1/envista/stations/' +
                           str(stationid) + '/data/earliest', headers=headers)
    r_late = requests.get('https://api.ims.gov.il/v1/envista/stations/' +
                          str(stationid) + '/data/latest', headers=headers)
    data = r_early.json()['data'][0]
    earliest = pd.to_datetime(data['datetime']).strftime('%Y-%m-%d')
    data = r_late.json()['data'][0]
    latest = pd.to_datetime(data['datetime']).strftime('%Y-%m-%d')
    print(
         'Downloading station {} with id: {}, from {} to {}'.format(
                 st_name,
                 stationid,
                 earliest,
                 latest))
    # one channel download:
    if channel_name is not None:
        channel_id = [x['id'] for x in data['channels']
                      if x['name'] == channel_name]
        if channel_id:
            print('getting just {} channel with id: {}'.format(channel_name,
                                                               channel_id[0]))
            ch_id = channel_id[0]
            dates = get_dates_list(earliest, latest)
            df_list = []
            for i in range(len(dates) - 1):
                first_date = dates[i].strftime('%Y/%m/%d')
                last_date = dates[i + 1].strftime('%Y/%m/%d')
                print('proccesing dates: {} to {}'.format(first_date,
                                                          last_date))
                dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                              str(stationid) + '/data/' + str(ch_id) +
                              '?from=' + first_date + '&to=' + last_date)
                r = requests.get(dl_command, headers=headers)
                if r.status_code == 204:  # i.e., no content:
                    print('no content for this search, skipping...')
                    continue
                print('parsing to dataframe...')
                df_list.append(parse_ims_to_df(r.json()['data'], channel_name))
            print('concatanating df and transforming to xarray...')
            df_all = pd.concat(df_list)
            # only valid results:
            # df_valid = df_all[df_all['valid']]
            df_all.index.name = 'time'
            # remove duplicated index values:
            df_all = df_all[~df_all.index.duplicated()]
            first = df_all.index[0]
            last = df_all.index[-1]
            new_index = pd.date_range(first, last, freq='10min')
            df_all = df_all.reindex(new_index)
            valid_name = channel_name + '_valid'
            value_name = channel_name + '_value'
            df_all[valid_name].fillna(False, inplace=True)
            # replace non valid measurments with nans
            new_vals = df_all[value_name].where(df_all[valid_name])
            df_all[value_name] = new_vals
            df_all.index.name = 'time'
            da = to_dataarray(df_all, meta)
            filename = '_'.join([meta['name'], str(meta['id']), channel_name,
                                 '10mins']) + '.nc'
            comp = dict(zlib=True, complevel=9)  # best compression
            encoding = {var: comp for var in da.to_dataset().data_vars}
            print('saving to {} to {}'.format(filename, savepath))
            da.to_netcdf(savepath / filename, 'w', encoding=encoding)
            print('done!')
    # all channels download add support here:
    elif channel_name is None:
        print('getting all channels...')
        dates = get_dates_list(earliest, latest)
        df_list = []
        for i in range(len(dates) - 1):
            first_date = dates[i].strftime('%Y/%m/%d')
            last_date = dates[i + 1].strftime('%Y/%m/%d')
            print('proccesing dates: {} to {}'.format(first_date,
                                                      last_date))
            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
                          str(stationid) + '/data?from=' + first_date +
                          '&to=' + last_date)
            r = requests.get(dl_command, headers=headers)
            if r.status_code == 204:  # i.e., no content:
                print('no content for this search, skipping...')
                break
            print('parsing to dataframe...')
            df_list.append(parse_ims_to_df(r.json()['data'], None))
    return

#def download_ims_data(geo_df, path, end_date='2019-04-15'):
#    import requests
#    import glob
#    import pandas as pd
#
#    def to_dataarray(df, index, row):
#        import pandas as pd
#        ds = df.to_xarray()
#        ds['time'] = pd.to_datetime(ds.time)
#        channel_name = ds.name.isel(time=0).values
#        channel_id = ds.id.isel(time=0).values
#        ds = ds.drop(['id', 'name'])
#        da = ds.to_array(dim='TD', name=str(index))
#        da.attrs['channel_id'] = channel_id.item()
#        da.attrs['channel_name'] = channel_name.item()
#        da.attrs['station_name'] = row.name_english
#        da.attrs['station_id'] = row.ID
#        da.attrs['station_lat'] = row.lat
#        da.attrs['station_lon'] = row.lon
#        da.attrs['station_alt'] = row.alt
#        return da
#
#    def get_dates_list(starting_date, end_date):
#        """divide the date span into full 1 years and a remainder, tolist"""
#        import numpy as np
#        import pandas as pd
#        end_date = pd.to_datetime(end_date)
#        s_year = starting_date.year
#        e_year = end_date.year
#        years = np.arange(s_year, e_year + 1)
#        dates = [starting_date.replace(year=x) for x in years]
#        if (end_date - dates[-1]).days > 0:
#            dates.append(end_date)
#        return dates
#
#    myToken = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
#    headers = {'Authorization': 'ApiToken ' + myToken}
#    already_dl = []
#    for paths in glob.glob(path+'*_TD.nc'):
#        already_dl.append(paths.split('/')[-1].split('.')[0].split('_')[0])
#        to_download = list(set(geo_df.index.values.tolist()
#                               ).difference(set(already_dl)))
#    if to_download:
#        geo_df = geo_df.loc[to_download]
#    for index, row in geo_df.iterrows():
#        # get a list of dates to download: (1 year old parts)
#        dates = get_dates_list(row.starting_date, end_date)
#        # get station id and channel id(only dry temperature):
#        name = row.name_english
#        station_id = row.ID
#        channel_id = row.channel_id
#        # if tempertue is not measuered in station , skip:
#        if channel_id == 0:
#            continue
#        print(
#            'Getting IMS data for {} station(ID={}) from channel {}'.format(
#                name,
#                station_id,
#                channel_id))
#        # loop over one year time span and download:
#        df_list = []
#        for i in range(len(dates) - 1):
#            first_date = dates[i].strftime('%Y/%m/%d')
#            last_date = dates[i + 1].strftime('%Y/%m/%d')
#            print('proccesing dates: {} to {}'.format(first_date, last_date))
#            dl_command = ('https://api.ims.gov.il/v1/envista/stations/' +
#                          str(station_id) + '/data/' + str(channel_id) +
#                          '?from=' + first_date + '&to=' + last_date)
#            r = requests.get(dl_command, headers=headers)
#            if r.status_code == 204:  # i.e., no content:
#                print('no content for this search, skipping...')
#                break
#            print('parsing to dataframe...')
#            df_list.append(parse_ims_to_df(r.json()['data']))
#        print('concatanating df and transforming to xarray...')
#        df_all = pd.concat(df_list)
#        # only valid results:
#        # df_valid = df_all[df_all['valid']]
#        df_all.index.name = 'time'
#        da = to_dataarray(df_all, index, row)
#        filename = index + '_TD.nc'
#        comp = dict(zlib=True, complevel=9)  # best compression
#        encoding = {var: comp for var in da.to_dataset().data_vars}
#        print('saving to {} to {}'.format(filename, path))
#        da.to_netcdf(path + filename, 'w', encoding=encoding)
#        print('done!')
#    #    return df_list
#    # pick station and time span
#    # download
#    # call parse_ims_to_df
#    # concatanate and save to nc
#    return


def produce_T_dataset(path, save=True, unique_index=True,
                      clim_period='dayofyear'):
    import xarray as xr
    da_list = []
    for file_and_path in path.glob('*TD.nc'):
        da = xr.open_dataarray(file_and_path)
        print('post-proccessing temperature data for {} station'.format(da.name))
        da_list.append(fill_missing_single_ims_station(da, unique_index,
                                                       clim_period))
    ds = xr.merge(da_list)
    if save:
        filename = 'IMS_TD_israeli_for_gps.nc'
        print('saving {} to {}'.format(filename, path))
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path / filename, 'w', encoding=encoding)
        print('Done!')
    return ds


def fill_missing_single_ims_station(da, unique_index=True,
                                    clim_period='dayofyear'):
    """fill in the missing time data for the ims station of any variable with
    clim_period is the fine tuning of the data replaced, options are:
        month, weekofyear, dayofyear. return a dataset with original and filled
        dataarray"""
    # da should be dattaarray and not dataset!
    import pandas as pd
    import numpy as np
    import xarray as xr
    from aux_gps import get_unique_index
    if unique_index:
        ind_diff = da.size - get_unique_index(da).size
        da = get_unique_index(da)
        print('dropped {} non-unique datetime index.'.format(ind_diff))
    # make sure no coords are in xarray:
    da = da.reset_coords(drop=True)
    # make sure nans are dropped:
    nans_diff = da.size - da.dropna('time').size
    print('dropped {} nans.'.format(nans_diff))
    da_no_nans = da.dropna('time')
    if clim_period == 'month':
        grpby = 'time.month'
        print('long term monthly mean data replacment selected')
    elif clim_period == 'weekofyear':
        print('long term weekly mean data replacment selected')
        grpby = 'time.weekofyear'
    elif clim_period == 'dayofyear':
        print('long term daily mean data replacment selected')
        grpby = 'time.dayofyear'
    # first compute the climatology and the anomalies:
    print('computing anomalies:')
    climatology = da_no_nans.groupby(grpby).mean('time')
    anom = da_no_nans.groupby(grpby) - climatology
    # then comupte the diurnal cycle:
    print('computing diurnal change:')
    diurnal = anom.groupby('time.hour').mean('time')
    # assemble old and new time and comupte the difference:
    print('assembeling missing data:')
    old_time = pd.to_datetime(da_no_nans.time.values)
    freq = pd.infer_freq(da.time.values)
    new_time = pd.date_range(da_no_nans.time.min().item(),
                             da_no_nans.time.max().item(), freq=freq)
    missing_time = pd.to_datetime(
        sorted(
            set(new_time).difference(
                set(old_time))))
    missing_data = np.empty((missing_time.shape))
    print('proccessing missing data...')
    for i in range(len(missing_data)):
        # replace data as to monthly long term mean and diurnal hour:
        # missing_data[i] = (climatology.sel(month=missing_time[i].month) +
        missing_data[i] = (climatology.sel({clim_period: getattr(missing_time[i],
                                                                 clim_period)}) +
                           diurnal.sel(hour=missing_time[i].hour))
    series = pd.Series(data=missing_data, index=missing_time)
    series.index.name = 'time'
    mda = series.to_xarray()
    mda.name = da.name
    new_data = xr.concat([mda, da_no_nans], 'time')
    new_data = new_data.sortby('time')
    # copy attrs:
    new_data.attrs = da.attrs
    new_data.attrs['description'] = 'missing data was '\
                                    'replaced by using ' + clim_period \
                                    + ' mean and hourly signal.'
    # put new_data and missing data into a dataset:
    dataset = new_data.to_dataset(name=new_data.name)
    dataset[new_data.name + '_original'] = da_no_nans
    print('done!')
    return dataset

#    # resample to 5min with resample_method: (interpolate is very slow)
#    print('resampling to 5 mins using {}'.format(resample_method))
#    # don't resample the missing data:
#    dataset = dataset.resample(time='5min').ffill()