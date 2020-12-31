#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:12:42 2020

@author: shlomi
"""
from PW_paths import work_yuval
ims_path = work_yuval / 'IMS_T'
gis_path = work_yuval / 'gis'
awd_path = work_yuval/'AW3D30'


def slice_var_ds_at_dt_and_convert_to_dataframe(var_ds, df, dt='2018-04-15T22:00:00'):
    """
    slice the var dataset (PWV) with specific datetime and add lat, lon and alt from df

    Parameters
    ----------
    var_ds : Xarray Dataset
        containing variable such as PWV vs. time.
    df : Pandas DataFrame
        containing lat, lon and alt cols, indexed by var_ds data_vars.
    dt : datetime string, optional
        DESCRIPTION. The default is '2018-04-15T22:00:00'.

    Returns
    -------
    hdf : pandas dataframe
        sliced var indexed by alt.

    """
    time_dim = list(set(var_ds.dims))[0]
    var_dt = var_ds.sel({time_dim: dt}).expand_dims(time_dim)
    hdf = var_dt.to_dataframe().T
    hdf = hdf.join(df[['lat', 'lon', 'alt']])
    hdf = hdf.set_index('alt')
    hdf = hdf.sort_index().dropna()
    return hdf


def get_pressure_lapse_rate(path=ims_path, model='LR', plot=False):
    from aux_gps import linear_fit_using_scipy_da_ts
    import matplotlib.pyplot as plt
    import xarray as xr
    from aux_gps import keep_iqr
    bp = xr.load_dataset(ims_path / 'IMS_BP_israeli_10mins.nc')
    bps = [keep_iqr(bp[x]) for x in bp]
    bp = xr.merge(bps)
    mean_p = bp.mean('time').to_array('alt')
    mean_p.name = 'mean_pressure'
    alts = [bp[x].attrs['station_alt'] for x in bp.data_vars]
    mean_p['alt'] = alts
    _, results = linear_fit_using_scipy_da_ts(mean_p, model=model, slope_factor=1, not_time=True)
    slope = results['slope']
    inter = results['intercept']
    modeled_var = slope * mean_p['alt'] + inter
    if plot:
        fig, ax = plt.subplots()
        modeled_var.plot(ax=ax, color='r')
        mean_p.plot.line(linewidth=0., marker='o', ax=ax, color='b')
        # lr = 1000 * abs(slope)
        textstr = 'Pressure lapse rate: {:.1f} hPa/km'.format(1000 * slope)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        ax.set_xlabel('Height a.s.l [m]')
        ax.set_ylabel('Mean Pressure [hPa]')
    return results


def get_var_lapse_rate(hdf, model='LR', plot=False):
    from aux_gps import linear_fit_using_scipy_da_ts
    import matplotlib.pyplot as plt
    import numpy as np
    hda = hdf.iloc[:, 0].to_xarray()
    dt = hda.name.strftime('%Y-%m-%d %H:%M')
    hda.name = ''
    log_hda = np.log(hda)
    # assume pwv = pwv0*exp(-h/H)
    # H is the water vapor scale height
    _, results = linear_fit_using_scipy_da_ts(log_hda, model=model, slope_factor=1, not_time=True)
    H = -1.0 / results['slope']
    a0 = np.exp(results['intercept'])
    modeled_var = a0 * np.exp(-hda['alt'] / H)
    if plot:
        fig, ax = plt.subplots()
        modeled_var.plot(ax=ax, color='r')
        hda.plot.line(linewidth=0., marker='o', ax=ax, color='b')
        # lr = 1000 * abs(slope)
        ax.set_title(dt)
        textstr = 'WV scale height: {:.1f} m'.format(H)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        ax.set_xlabel('Height a.s.l [m]')
        ax.set_ylabel('PWV [mm]')
    return H


def apply_lapse_rate_change(hdf, H):
    import numpy as np
    # make sure lapse rate is negative:
    assert H > 0
    new_hdf = hdf.copy()
    new_hdf.iloc[:, 0] = hdf.iloc[:, 0] * np.exp(hdf.index / H)
    return new_hdf


def interpolate_at_one_dt(new_hdf, H, dem_path=awd_path, ppd=50):
    from aux_gps import coarse_dem
    import numpy as np
    from pykrige.rk import Krige
    # create mesh and load DEM:
    da = create_lat_lon_mesh(points_per_degree=ppd)  # 500?
    # populate the empty mesh grid with stations data:
    for i, row in new_hdf.iterrows():
        lat = da.sel(lat=row['lat'], method='nearest').lat.values
        lon = da.sel(lon=row['lon'], method='nearest').lon.values
        da.loc[{'lat': lat, 'lon': lon}] = row.iloc[0]

    c = np.linspace(min(da.lat.values), max(da.lat.values), da.shape[0])
    r = np.linspace(min(da.lon.values), max(da.lon.values), da.shape[1])
    rr, cc = np.meshgrid(r, c)
    vals = ~np.isnan(da.values)
    X = np.column_stack([rr[vals], cc[vals]])
    rr_cc_as_cols = np.column_stack([rr.flatten(), cc.flatten()])
    # y = da_scaled.values[vals]
    y = da.values[vals]
    model = Krige(method='ordinary', variogram_model='spherical',
                  verbose=True)
    model.fit(X, y)
    interpolated = model.predict(rr_cc_as_cols).reshape(da.values.shape)
    da_inter = da.copy(data=interpolated)
    awd = coarse_dem(da, dem_path=dem_path)
    assert H > 0
    da_inter *= np.exp(-1.0 * awd / H)
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

def Interpolating_models_ims(time='2013-10-19T22:00:00', var='TD', plot=True,
                             gis_path=gis_path, method='okrig',
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
    da = create_lat_lon_mesh(points_per_degree=250)  # 500?
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
        rr_cc_as_cols = np.column_stack([XX.flatten(), YY.flatten(), ZZ.flatten()])
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
#            ytests += list(y_test)        cmap = plt.get_cmap('spring', 10)
        Q = ax.quiver(isr['X'], isr['Y'], isr['U'], isr['V'],
                      isr['cm_per_year'], cmap=cmap)
        fig.colorbar(Q, extend='max')

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
