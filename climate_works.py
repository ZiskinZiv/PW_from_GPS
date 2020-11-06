#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:21:02 2020

@author: shlomi
"""
from sklearn_xarray import RegressorWrapper
from PW_paths import work_yuval
climate_path = work_yuval / 'climate'


def prepare_ORAS5_download_script(path=work_yuval, var='sossheig'):
    from aux_gps import path_glob
    files = path_glob(path, 'wget_oras5*.sh')
    for file in files:
        filename = file.as_posix().split('/')[-1].split('.')[0]
        print('reading file {} file'.format(filename))
        with open(file) as f:
            content = f.readlines()
            var_content = [x for x in content if var in x]
            new_filename = filename + '_{}.sh'.format(var)
            with open(path / new_filename, 'w') as fi:
                for item in var_content:
                    fi.write("%s\n" % item)
    return


def create_index_from_synoptics(path=climate_path, syn_cat='normal'):
    """create a long term index from synoptics"""
    from aux_gps import anomalize_xr
    from aux_gps import annual_standertize
    from synoptic_procedures import agg_month_count_syn_class
    da = agg_month_count_syn_class(path=path, syn_category=syn_cat,
                                   freq=False)
    ds = da.to_dataset('syn_cls')
    ds = anomalize_xr(ds, 'MS')
    ds = annual_standertize(ds)
    ds = ds.fillna(0)
    return ds


def read_ea_index(path=climate_path):
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import pandas as pd
    file = path_glob(path, 'ea_index.txt')[0]
    df = pd.read_csv(file, names=['year', 'month', 'ea'], delim_whitespace=True, header=9)
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.sort_index()
    df = df.drop(['year', 'month'], axis=1)
    da = df.to_xarray()
    save_ncfile(da, path, 'ea_index.nc')
    return da


def read_west_moi(path=climate_path):
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import pandas as pd
    file = path_glob(path, 'Western_MOI.txt')[0]
    df = pd.read_csv(file, delim_whitespace=True)
    df['year']=df.index
    df = pd.melt(df, id_vars='year', var_name='month', value_name='wemoi')
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.sort_index()
    df = df.drop(['year', 'month'], axis=1)
    da = df.to_xarray()
    save_ncfile(da, path, 'wemo_index.nc')
    return da


def read_mo_indicies(path=climate_path, moi=1, resample_to_mm=True):
    from aux_gps import path_glob
    from aux_gps import save_ncfile
    import pandas as pd
    file = path_glob(path, 'moi{}.dat'.format(moi))[0]
    df = pd.read_fwf(file,
        names=['year', 'date', 'moi{}'.format(moi)],
        widths=[4, 8, 5])
    df['date'] = df['date'].str.strip('.')
    df['date'] = df['date'].str.strip(' ')
    df['date'] = df['date'].str.replace(' ', '0')
    df['date'] = df['date'].str.replace('.', '-')
    df['time'] = df['year'].astype(str) + '-' + df['date'].astype(str)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
    df = df.set_index('time')
    df = df.drop(['date', 'year'], axis=1)
    da = df.to_xarray()
    if resample_to_mm:
        da = da.resample(time='MS').mean()
    save_ncfile(da, path, 'moi{}_index.nc'.format(moi))
    return da


def produce_interannual_df(climate_path=climate_path, work_path=work_yuval,
                           lags=1, corr_thresh=0.2, smooth=False):
    import xarray as xr
    from aux_gps import smooth_xr
    from aux_gps import anomalize_xr
    from aux_gps import annual_standertize
    from synoptic_procedures import agg_month_count_syn_class    # load pwv:
    pw = xr.load_dataset(
        work_path /
        'GNSS_PW_monthly_anoms_thresh_50_homogenized.nc')
    pw_mean = pw.to_array('station').mean('station')
    if smooth:
        pw_mean = smooth_xr(pw_mean)
    df_pw = pw_mean.to_dataframe(name='pwv')
    # load other large circulation indicies:
    ds = load_all_indicies(path=climate_path)
    if smooth:
        ds = smooth_xr(ds)
    df = ds.to_dataframe()
    # add lags:
    for ind in df.columns:
        df['{}+1'.format(ind)] = df[ind].shift(lags)
        df['{}-1'.format(ind)] = df[ind].shift(-lags)
    # load synoptics:
    ds = agg_month_count_syn_class(path=climate_path, syn_category='upper', freq=False).to_dataset('syn_cls')
    ds = anomalize_xr(ds, 'MS')
    ds = annual_standertize(ds)
    ds = ds.fillna(0)
    ds_cls = agg_month_count_syn_class(path=climate_path, syn_category='normal', freq=False).to_dataset('syn_cls')
    ds_cls = anomalize_xr(ds_cls, 'MS')
    ds_cls = annual_standertize(ds_cls)
    ds_cls = ds_cls.fillna(0)
    if smooth:
        ds = smooth_xr(ds)
        ds_cls = smooth_xr(ds_cls)
    df_syn = ds.to_dataframe()
    df_syn_cls = ds_cls.to_dataframe()
    df_syn = df_syn.join(df_syn_cls)
    df = df.join(df_syn)
    # sort cols:
    df.columns = [str(x) for x in df.columns]
    cols = sorted([x for x in df.columns])
    df = df[cols]
#    df = df.dropna()
    df = df_pw.join(df)
    if corr_thresh is not None:
        corr = df.corr()['pwv']
        corr= corr[abs(corr)>corr_thresh]
        inds = corr.index
        return df[inds]
    else:
        return df


def preprocess_interannual_df(df, yname='pwv'):
    df = df.dropna()
    y = df[yname].to_xarray()
    xnames = [x for x in df.columns if yname not in x]
    X = df[xnames].to_xarray().to_array('regressors')
    X = X.transpose('time', 'regressors')
    return X, y


def load_all_indicies(path=climate_path):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(path, '*_index.nc')
    ds_list = [xr.load_dataset(file) for file in files]
    ds = xr.merge(ds_list)
    return ds


def load_z_from_ERA5(savepath=climate_path):
    import xarray as xr
    from aux_gps import save_ncfile
    ds = xr.load_dataset(savepath / 'ERA5_Z_500_hPa_for_NCPI_1979-2020.nc')
    if 'expver' in ds.dims:
        ds = ds.sel(expver=1)
    z = ds['z']
    z = z.rename({'latitude': 'lat', 'longitude': 'lon'})
    z = z.sortby('lat')
    save_ncfile(z, climate_path, 'ERA5_Z_500_hPa_NCP_1979-2020.nc')
    return


def calculate_NCPI(savepath=climate_path):
    import xarray as xr
    from aux_gps import anomalize_xr
    from aux_gps import save_ncfile
    z = xr.load_dataarray(climate_path / 'ERA5_Z_500_hPa_NCP_1979-2020.nc')
    # positive NCP pole:
    pos = z.sel(lat=55, lon=slice(0, 10)).mean('lon')
    neg = z.sel(lat=45, lon=slice(50, 60)).mean('lon')
    ncp = pos - neg
    ncp_anoms = anomalize_xr(ncp, 'MS')
    ncpi = ncp_anoms.groupby('time.month') / ncp.groupby('time.month').std()
    ncpi = ncpi.reset_coords(drop=True)
    ncpi.name = 'NCPI'
    ncpi.attrs['long_name'] = 'North sea Caspian Pattern Index'
    save_ncfile(ncpi, savepath, 'ncp_index.nc')
    return ncpi


def read_old_ncp(savepath=climate_path):
    import pandas as pd
    df = pd.read_csv(savepath / 'ncp.dat', delim_whitespace=True)
    df.columns = ['year', 'month', 'ncpi']
    df['dt'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str))
    df.set_index('dt', inplace=True)
    df = df.drop(['year', 'month'], axis=1)
    df = df.sort_index()
    df.index.name = 'time'
    da = df.to_xarray()['ncpi']
    da.name = 'Old_NCPI'
    return da


def DI_and_PWV_lag_analysis(bin_di, path=work_yuval, station='tela',
                            hour_interval=48):
    import xarray as xr
    from aux_gps import xr_reindex_with_date_range
    bin_di = xr_reindex_with_date_range(bin_di, freq='5min')
    pw = xr.open_dataset(path /'GNSS_PW_thresh_50_for_diurnal_analysis.nc')[station]
    print('loaded {} pwv station.'.format(station))
    pw.load()
    df = pw.to_dataframe()
    df['bins'] = bin_di.to_dataframe()
    cats = df['bins'].value_counts().index.values
    pw_time_cat_list = []
#    for di_cat in cats:
#                
#    return df


def bin_DIs(di, bins=[300, 500, 700, 900, 1030]):
    import pandas as pd
    import numpy as np
    df = di.to_dataframe()
    df = df.dropna()
    labels = np.arange(1, len(bins))
    df_bins = pd.cut(df[di.name], bins=bins, labels=labels)
    da = df_bins.to_xarray()
    return da


def read_all_DIs(path=climate_path, sample_rate='12H'):
    from aux_gps import path_glob
    import xarray as xr
    if sample_rate == '12H':
        files = path_glob(path, 'data_DIs_Bet_Dagan_*.mat')
    elif sample_rate == '3H':
        files = path_glob(path, 'data_DIs_Bet_Dagan_hr_*.mat')
    da_list = [read_DIs_matfile(x, sample_rate=sample_rate) for x in files]
    da = xr.concat(da_list, 'time')
    da = da.sortby('time')
    return da


def read_DIs_matfile(file, sample_rate='12H'):
    from scipy.io import loadmat
    import datetime
    import pandas as pd
    from aux_gps import xr_reindex_with_date_range
    print('sample rate is {}'.format(sample_rate))
#    file = path / 'data_DIs_Bet_Dagan_2015.mat'
#    name = file.as_posix().split('/')[-1].split('.')[0]
    mat = loadmat(file)
    real_name = [x for x in mat.keys() if '__' not in x and 'None' not in x][0]
    arr = mat[real_name]
    startdate = datetime.datetime.strptime("0001-01-01", "%Y-%m-%d")
    dts = [pd.to_datetime(startdate + datetime.timedelta(arr[x, 1])) -
           pd.Timedelta(366, unit='D') for x in range(arr[:, 1].shape[0])]
    vals = arr[:, 0]
    df = pd.DataFrame(vals, index=dts)
    df.columns = ['p']
    df.index.name = 'time'
    da = df.to_xarray()
    da = xr_reindex_with_date_range(da, freq=sample_rate)['p']
    return da


def run_MLR(X, y, plot=True):
    from sklearn.linear_model import LinearRegression
    model = ImprovedRegressor(LinearRegression(fit_intercept=True),
                              reshapes='regressors', sample_dim='time')
    model.fit(X, y)
    print(model.results_['explained_variance'])
    results = model.results_['original'].to_dataframe()
    results['predict'] = model.results_['predict'].to_dataframe()
    if plot:
        ax = results.plot()
        ax.set_ylabel('PWV anomalies [mm]')
        ax.set_title('PWV monthly means anomalies and reconstruction')
        ax.grid()
    return model


def sk_attr(est, attr):
    """check weather an attr exists in sklearn model"""
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    try:
        check_is_fitted(est, attr)
        return True
    except NotFittedError:
        return False


def get_p_values(X, y, sample_dim):
    """produce p_values and return an xarray with the proper dims"""
    import numpy as np
    from sklearn.feature_selection import f_regression
    feature_dim, mt_dim = get_feature_multitask_dim(X, y, sample_dim)
    if mt_dim:
        pval = np.empty((y[mt_dim].size, X[feature_dim].size))
        for i in range(y[mt_dim].size):
            f, pval[i, :] = f_regression(X, y.isel({mt_dim: i}))
    else:
        pval = np.empty((X[feature_dim].size))
        f, pval[:] = f_regression(X, y)
    return pval


def get_feature_multitask_dim(X, y, sample_dim):
    """return feature dim and multitask dim if exists, otherwise return empty
    lists"""
    # check if y has a multitask dim, i.e., y(sample, multitask)
    mt_dim = [x for x in y.dims if x != sample_dim]
    # check if X has a feature dim, i.e., X(sample, regressors)
    feature_dim = [x for x in X.dims if x != sample_dim]
    if mt_dim:
        mt_dim = mt_dim[0]
    if feature_dim:
        feature_dim = feature_dim[0]
    return feature_dim, mt_dim


class ImprovedRegressor(RegressorWrapper):
    def __init__(self, estimator=None, reshapes=None, sample_dim=None,
                 **kwargs):
        # call parent constructor to set estimator, reshapes, sample_dim,
        # **kwargs
        super().__init__(estimator, reshapes, sample_dim, **kwargs)

    def fit(self, X, y=None, verbose=True, **fit_params):
        """ A wrapper around the fitting function.
        Improved: adds the X_ and y_ and results_ attrs to class.
        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The training input samples.

        y : xarray DataArray, Dataset other other array-like
            The target values.

        Returns
        -------
        Returns self.
        """
        self = super().fit(X, y, **fit_params)
        # set results attr
        self.results_ = self.make_results(X, y, verbose)
        setattr(self, 'results_', self.results_)
        # set X_ and y_ attrs:
        setattr(self, 'X_', X)
        setattr(self, 'y_', y)
        return self

    def make_results(self, X, y, verbose=True):
        """ make results for all models type into xarray"""
        import xarray as xr
        from sklearn.metrics import r2_score
        from sklearn.metrics import explained_variance_score
        feature_dim, mt_dim = get_feature_multitask_dim(X, y, self.sample_dim)
        rds = y.to_dataset(name='original').copy(deep=False, data=None)
        if sk_attr(self, 'coef_') and sk_attr(self, 'intercept_'):
            rds[feature_dim] = X[feature_dim]
            if mt_dim:
                rds['params'] = xr.DataArray(self.coef_, dims=[mt_dim,
                                                               feature_dim])
                rds['intercept'] = xr.DataArray(self.intercept_, dims=[mt_dim])
                pvals = get_p_values(X, y, self.sample_dim)
                rds['pvalues'] = xr.DataArray(pvals, dims=[mt_dim,
                                                           feature_dim])
            else:
                rds['params'] = xr.DataArray(self.coef_, dims=feature_dim)
                rds['intercept'] = xr.DataArray(self.intercept_)
                pvals = get_p_values(X, y, self.sample_dim)
                rds['pvalues'] = xr.DataArray(pvals, dims=feature_dim)
        elif sk_attr(self, 'feature_importances_'):
            if mt_dim:
                rds['feature_importances'] = xr.DataArray(self.
                                                          feature_importances_,
                                                          dims=[mt_dim,
                                                                feature_dim])
            else:
                rds['feature_importances'] = xr.DataArray(self.
                                                          feature_importances_,
                                                          dims=[feature_dim])
        predict = self.predict(X)
        if mt_dim:
            predict = predict.rename({self.reshapes: mt_dim})
            rds['predict'] = predict
            r2 = r2_score(y, predict, multioutput='raw_values')
            rds['r2'] = xr.DataArray(r2, dims=mt_dim)
        else:
            rds['predict'] = predict
            r2 = r2_score(y, predict)
            rds['r2'] = xr.DataArray(r2)
        if feature_dim:
            r2_adj = 1.0 - (1.0 - rds['r2']) * (len(y) - 1.0) / \
                 (len(y) - X.shape[1])
        else:
            r2_adj = 1.0 - (1.0 - rds['r2']) * (len(y) - 1.0) / (len(y))
        rds['r2_adj'] = r2_adj
        rds['predict'].attrs = y.attrs
        rds['resid'] = y - rds['predict']
        rds['resid'].attrs = y.attrs
        rds['resid'].attrs['long_name'] = 'Residuals'
        rds['dw_score'] = (rds['resid'].diff(self.sample_dim)**2).sum(self.sample_dim,
                                                                  keep_attrs=True) / (rds['resid']**2).sum(self.sample_dim, keep_attrs=True)
        exp_var =  explained_variance_score(y, rds['predict'].values)
        rds['explained_variance'] = exp_var

#        rds['corrcoef'] = self.corrcoef(X, y)
        # unstack dims:
        if mt_dim:
            rds = rds.unstack(mt_dim)
        # put coords attrs back:
#        for coord, attr in y.attrs['coords_attrs'].items():
#            rds[coord].attrs = attr
#        # remove coords attrs from original, predict and resid:
#        rds.original.attrs.pop('coords_attrs')
#        rds.predict.attrs.pop('coords_attrs')
#        rds.resid.attrs.pop('coords_attrs')
        all_var_names = [x for x in rds.data_vars.keys()]
        sample_types = [x for x in rds.data_vars.keys()
                        if self.sample_dim in rds[x].dims]
        feature_types = [x for x in rds.data_vars.keys()
                         if feature_dim in rds[x].dims]
        error_types = list(set(all_var_names) - set(sample_types +
                                                    feature_types))
        rds.attrs['sample_types'] = sample_types
        rds.attrs['feature_types'] = feature_types
        rds.attrs['error_types'] = error_types
        rds.attrs['sample_dim'] = self.sample_dim
        rds.attrs['feature_dim'] = feature_dim
        # add X to results:
        rds['X'] = X
        if verbose:
            print('Producing results...Done!')
        return rds

    def save_results(self, path_like):
        ds = self.results_
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path_like, 'w', encoding=encoding)
        print('saved results to {}.'.format(path_like))
        return

    def plot_like(self, field, flist=None, fmax=False, tol=0.0,
                  mean_lonlat=[True, False], title=None, **kwargs):
        # div=False, robust=False, vmax=None, vmin=None):
        """main plot for the results_ product of ImrovedRegressor
        flist: list of regressors to plot,
        fmax: wether to normalize color map on the plotted regressors,
        tol: used to control what regressors to show,
        mean_lonlat: wether to mean fields on the lat or lon dim"""
        from matplotlib.ticker import ScalarFormatter
        import matplotlib.pyplot as plt
        import aux_functions_strat as aux
        import pandas as pd
        # TODO: add area_mean support
        if not hasattr(self, 'results_'):
            raise AttributeError('No results yet... run model.fit(X,y) first!')
        rds = self.results_
        if field not in rds.data_vars:
            raise KeyError('No {} in results_!'.format(field))
        # if 'div' in keys:
        #     cmap = 'bwr'
        # else:
        #     cmap = 'viridis'
        plt_kwargs = {'yscale': 'log', 'yincrease': False, 'cmap': 'bwr'}
        if field in rds.attrs['sample_types']:
            orig = aux.xr_weighted_mean(rds['original'])
            try:
                times = aux.xr_weighted_mean(rds[field])
            except KeyError:
                print('Field not found..')
                return
            except AttributeError:
                times = rds[field]
            fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15, 7),
                                     num='Time_Level_Comperison')
            cmap_max = abs(max(abs(orig.min().values), abs(orig.max().values)))
            orig = orig.reindex({'time': pd.date_range(orig.time[0].values,
                                                       orig.time[-1].values,
                                                       freq='MS')})
            plt_sample = {**plt_kwargs}
            plt_sample.update({'center': 0.0, 'levels': 41, 'vmax': cmap_max})
            plt_sample.update(kwargs)
            con = orig.T.plot.contourf(ax=axes[0], **plt_sample)
            cb = con.colorbar
            cb.set_label(orig.attrs['units'], fontsize=10)
            ax = axes[0]
            ax.set_title(orig.attrs['long_name'] + ' original', loc='center')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            # plot the PREDICTED :
            times = times.reindex({'time': pd.date_range(times.time[0].values,
                                                         times.time[-1].values,
                                                         freq='MS')})
            plt_sample.update({'extend': 'both'})
            con = times.T.plot.contourf(ax=axes[1], **plt_sample)
            # robust=robust)
            cb = con.colorbar
            try:
                cb.set_label(times.attrs['units'], fontsize=10)
            except KeyError:
                print('no units found...''')
                cb.set_label(' ', fontsize=10)
            ax = axes[1]
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_title(times.attrs['long_name'] + ' predicted', loc='center')
            plt.subplots_adjust(left=0.05, right=0.995)
            [ax.invert_yaxis() for ax in con.ax.figure.axes]
            plt.show()
            return con
        elif field in rds.attrs['error_types']:
            # TODO: add contour lines
            if title is not None:
                suptitle = title
            else:
                suptitle = rds[field].name
            plt_error = {**plt_kwargs}
            plt_error.update({'cmap': 'viridis', 'add_colorbar': True,
                             'figsize': (6, 8)})
            plt_error.update(kwargs)
            if 'lon' in rds[field].dims:
                error_field = aux.xr_weighted_mean(rds[field],
                                                   mean_on_lon=mean_lonlat[0],
                                                   mean_on_lat=mean_lonlat[1])
            else:
                error_field = rds[field]
            try:
                con = error_field.plot.contourf(**plt_error)
                ax = plt.gca()
                ax.yaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            except KeyError:
                print('Field not found or units not found...')
                return
            except ValueError:
                con = error_field.plot(xscale='log', xincrease=False,
                                      figsize=(6, 8))
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            plt.show()
            plt.gca().invert_yaxis()
            return con
        elif field in rds.attrs['feature_types']:
            # TODO: add contour lines
            con_levels = [0.001, 0.005, 0.01, 0.05]  # for pvals
            con_colors = ['blue', 'cyan', 'yellow', 'red']  # for pvals
            import xarray as xr
            fdim = rds.attrs['feature_dim']
            if flist is None:
                flist = [x for x in rds[fdim].values if
                         xr.ufuncs.fabs(rds[field].sel({fdim: x})).mean() > tol]
            if rds[fdim].sel({fdim: flist}).size > 6:
                colwrap = 6
            else:
                colwrap = None
            if 'lon' in rds[field].dims:
                feature_field = aux.xr_weighted_mean(rds[field],
                                                     mean_on_lon=mean_lonlat[0],
                                                     mean_on_lat=mean_lonlat[1])
            else:
                feature_field = rds[field]
            vmax = feature_field.max()
            if fmax:
                vmax = feature_field.sel({fdim: flist}).max()
            if title is not None:
                suptitle = title
            else:
                suptitle = feature_field.name
            plt_feature = {**plt_kwargs}
            plt_feature.update({'add_colorbar': False, 'levels': 41,
                                'figsize': (15, 4),
                                'extend': 'min', 'col_wrap': colwrap})
            plt_feature.update(**kwargs)
            try:
                if feature_field.name == 'pvalues':
                    plt_feature.update({'colors': con_colors,
                                        'levels': con_levels, 'extend': 'min'})
                    plt_feature.update(**kwargs)
                    plt_feature.pop('cmap', None)
                else:
                    plt_feature.update({'cmap': 'bwr',
                                        'vmax': vmax})
                    plt_feature.update(**kwargs)
                fg = feature_field.sel({fdim: flist}).plot.contourf(col=fdim,
                                                                    **plt_feature)
                ax = plt.gca()
                ax.yaxis.set_major_formatter(ScalarFormatter())
                fg.fig.subplots_adjust(bottom=0.3, top=0.85, left=0.05)
                cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                fg.add_colorbar(
                    cax=cbar_ax,
                    orientation="horizontal",
                    format='%0.3f')
                fg.fig.suptitle(suptitle, fontsize=12, fontweight=750)
            except KeyError:
                print('Field not found or units not found...')
                return
            except ValueError as valerror:
                print(valerror)
                fg = feature_field.plot(col=fdim, xscale='log', xincrease=False,
                                        figsize=(15, 4))
                fg.fig.subplots_adjust(bottom=0.3, top=0.85, left=0.05)
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            plt.show()
            [ax.invert_yaxis() for ax in fg.fig.axes]
            return fg