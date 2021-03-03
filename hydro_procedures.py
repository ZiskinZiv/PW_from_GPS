#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:08:43 2019

@author: ziskin
"""

from PW_paths import savefig_path
from PW_paths import work_yuval
hydro_path = work_yuval / 'hydro'
gis_path = work_yuval / 'gis'
ims_path = work_yuval / 'IMS_T'
hydro_ml_path = hydro_path / 'hydro_ML'
# 'tela': 17135
hydro_pw_dict = {'nizn': 25191, 'klhv': 21105, 'yrcm': 55165,
                 'ramo': 56140, 'drag': 48125, 'dsea': 48192,
                 'spir': 56150, 'nrif': 60105, 'elat': 60190
                 }

hydro_st_name_dict = {25191: 'Lavan - new nizana road',
                      21105: 'Shikma - Tel milcha',
                      55165: 'Mamsheet',
                      56140: 'Ramon',
                      48125: 'Draga',
                      48192: 'Chiemar - down the cliff',
                      46150: 'Nekrot - Top',
                      60105: 'Yaelon - Kibutz Yahel',
                      60190: 'Solomon - Eilat'}
# TODO: treat all pwv from events as follows:
#    For each station:
#    0) rolling mean to all pwv 1 hour
#    1) take 288 points before events, if < 144 gone then drop
#    2) interpolate them 12H using spline/other
#    3) then, check if dts coinside 1 day before, if not concat all dts+pwv for each station
#    4) prepare features, such as pressure, doy, try to get pressure near the stations and remove the longterm hour dayofyear
#   pressure in BD anoms is highly correlated with SEDOM (0.9) and ELAT (0.88) so no need for local pressure features
# fixed filling with jerusalem centre since 2 drag events dropped due to lack of data 2018-11 2019-02 in pressure
#   5) feature addition: should be like pwv steps 1-3,
#   6) negative events should be sampled separtely, for
# 7) now prepare pwv and pressure to single ds with 1 hourly sample rate
# 8) produce positives and save them to file!
# 9) produce a way to get negatives considering the positives


# maybe implement permutaion importance to pwv ? see what is more important to
# the model in 24 hours ? only on SVC and MLP ?
# implemetn TSS and HSS scores and test them (make_scorer from confusion matrix)
# redo results but with inner and outer splits of 4, 4
# plot and see best_score per refit-scorrer - this is the best score of GridSearchCV on the entire
# train/validation subset per each outerfold - basically see if the test_metric increased after the gridsearchcv as it should
# use holdout set
# implement repeatedstratifiedkfold and run it...
# check for stability of the gridsearch CV...also run with 4-folds ?
# finalize the permutation_importances and permutation_test_scores

def prepare_tide_events_GNSS_dataset(hydro_path=hydro_path):
    import xarray as xr
    import pandas as pd
    import numpy as np
    from aux_gps import xr_reindex_with_date_range
    feats = xr.load_dataset(
        hydro_path/'hydro_tides_hourly_features_with_positives.nc')
    ds = feats['Tides'].to_dataset('GNSS').rename({'tide_event': 'time'})
    da_list = []
    for da in ds:
        time = ds[da].dropna('time')
        daa = time.copy(data=np.ones(time.shape))
        daa['time'] = pd.to_datetime(time.values)
        daa.name = time.name + '_tide'
        da_list.append(daa)
    ds = xr.merge(da_list)
    li = [xr_reindex_with_date_range(ds[x], freq='H') for x in ds]
    ds = xr.merge(li)
    return ds


def select_features_from_X(X, features='pwv'):
    if isinstance(features, str):
        f = [x for x in X.feature.values if features in x]
        X = X.sel(feature=f)
    elif isinstance(features, list):
        fs = []
        for f in features:
            fs += [x for x in X.feature.values if f in x]
        X = X.sel(feature=fs)
    return X


def combine_pos_neg_from_nc_file(hydro_path=hydro_path, all_neg=False, seed=1):
    from aux_gps import path_glob
    import xarray as xr
    import numpy as np
    # import pandas as pd
    np.random.seed(seed)
    file = path_glob(
        hydro_path, 'hydro_tides_hourly_features_with_positives_negatives_*.nc')[-1]
    ds = xr.open_dataset(file)
    # get the positive features and produce target:
    X_pos = ds['X_pos'].rename({'positive_sample': 'sample'})
    y_pos = xr.DataArray(np.ones(X_pos['sample'].shape), dims=['sample'])
    y_pos['sample'] = X_pos['sample']
    # choose at random y_pos size of negative class:
    X_neg = ds['X_neg'].rename({'negative_sample': 'sample'})
    if not all_neg:
        dts = np.random.choice(
            X_neg['sample'], y_pos['sample'].size, replace=False)
        X_neg = X_neg.sel(sample=dts)
    y_neg = xr.DataArray(np.zeros(X_neg['sample'].shape), dims=['sample'])
    y_neg['sample'] = X_neg['sample']
    # now concat all X's and y's:
    X = xr.concat([X_pos, X_neg], 'sample')
    y = xr.concat([y_pos, y_neg], 'sample')
    X.name = 'X'
    return X, y


def drop_hours_in_pwv_pressure_features(X, last_hours=7, verbose=True):
    import numpy as np
    Xcopy = X.copy()
    pwvs_to_drop = ['pwv_{}'.format(x) for x in np.arange(24-last_hours + 1, 25)]
    if set(pwvs_to_drop).issubset(set(X.feature.values)):
        if verbose:
            print('dropping {} from X.'.format(', '.join(pwvs_to_drop)))
        Xcopy = Xcopy.drop_sel(feature=pwvs_to_drop)
    pressures_to_drop = ['pressure_{}'.format(x) for x in np.arange(24-last_hours + 1, 25)]
    if set(pressures_to_drop).issubset(set(X.feature.values)):
        if verbose:
            print('dropping {} from X.'.format(', '.join(pressures_to_drop)))
        Xcopy = Xcopy.drop_sel(feature=pressures_to_drop)
    return Xcopy


def check_if_negatives_are_within_positives(neg_da, hydro_path=hydro_path):
    import xarray as xr
    import pandas as pd
    pos_da = xr.open_dataset(
        hydro_path / 'hydro_tides_hourly_features_with_positives.nc')['X']
    dt_pos = pos_da.sample.to_dataframe()
    dt_neg = neg_da.sample.to_dataframe()
    dt_all = dt_pos.index.union(dt_neg.index)
    dff = pd.DataFrame(dt_all, index=dt_all)
    dff = dff.sort_index()
    samples_within = dff[(dff.diff()['sample'] <= pd.Timedelta(1, unit='D'))]
    num = samples_within.size
    print('samples that are within a day of each other: {}'.format(num))
    print('samples are: {}'.format(samples_within))
    return dff


def produce_negatives_events_from_feature_file(hydro_path=hydro_path, seed=42,
                                               batches=1, verbose=1):
    # do the same thing for pressure (as for pwv), but not for
    import xarray as xr
    import numpy as np
    import pandas as pd
    from aux_gps import save_ncfile
    feats = xr.load_dataset(hydro_path / 'hydro_tides_hourly_features.nc')
    feats = feats.rename({'doy': 'DOY'})
    all_tides = xr.open_dataset(
        hydro_path / 'hydro_tides_hourly_features_with_positives.nc')['X_pos']
    # pos_tides = xr.open_dataset(hydro_path / 'hydro_tides_hourly_features_with_positives.nc')['tide_datetimes']
    tides = xr.open_dataset(
        hydro_path / 'hydro_tides_hourly_features_with_positives.nc')['Tides']
    # get the positives (tide events) for each station:
    df_stns = tides.to_dataset('GNSS').to_dataframe()
    # get all positives (tide events) for all stations:
    df = all_tides.positive_sample.to_dataframe()['positive_sample']
    df.columns = ['sample']
    stns = [x for x in hydro_pw_dict.keys()]
    other_feats = ['DOY', 'doy_sin', 'doy_cos']
    # main stns df features (pwv)
    pwv_df = feats[stns].to_dataframe()
    pressure = feats['bet-dagan'].to_dataframe()['bet-dagan']
    # define the initial no_choice_dt_range from the positive dt_range:
    no_choice_dt_range = [pd.date_range(
        start=dt, periods=48, freq='H') for dt in df]
    no_choice_dt_range = pd.DatetimeIndex(
        np.unique(np.hstack(no_choice_dt_range)))
    dts_to_choose_from = pwv_df.index.difference(no_choice_dt_range)
    # dts_to_choose_from_pressure = pwv_df.index.difference(no_choice_dt_range)
    # loop over all stns and produce negative events:
    np.random.seed(seed)
    neg_batches = []
    for i in np.arange(1, batches + 1):
        if verbose >= 0:
            print('preparing batch {}:'.format(i))
        neg_stns = []
        for stn in stns:
            dts_df = df_stns[stn].dropna()
            pwv = pwv_df[stn].dropna()
            # loop over all events in on stn:
            negatives = []
            negatives_pressure = []
            # neg_samples = []
            if verbose >= 1:
                print('finding negatives for station {}, events={}'.format(
                    stn, len(dts_df)))
            # print('finding negatives for station {}, dt={}'.format(stn, dt.strftime('%Y-%m-%d %H:%M')))
            cnt = 0
            while cnt < len(dts_df):
                # get random number from each stn pwv:
                # r = np.random.randint(low=0, high=len(pwv.index))
                # random_dt = pwv.index[r]
                random_dt = np.random.choice(dts_to_choose_from)
                negative_dt_range = pd.date_range(
                    start=random_dt, periods=24, freq='H')
                if not (no_choice_dt_range.intersection(negative_dt_range)).empty:
                    # print('#')
                    if verbose >= 2:
                        print('Overlap!')
                    continue
                # get the actual pwv and check it is full (24hours):
                negative = pwv.loc[pwv.index.intersection(negative_dt_range)]
                neg_pressure = pressure.loc[pwv.index.intersection(
                    negative_dt_range)]
                if len(negative.dropna()) != 24 or len(neg_pressure.dropna()) != 24:
                    # print('!')
                    if verbose >= 2:
                        print('NaNs!')
                    continue
                if verbose >= 2:
                    print('number of dts that are already chosen: {}'.format(
                        len(no_choice_dt_range)))
                negatives.append(negative)
                negatives_pressure.append(neg_pressure)
                # now add to the no_choice_dt_range the negative dt_range we just aquired:
                negative_dt_range_with_padding = pd.date_range(
                    start=random_dt-pd.Timedelta(24, unit='H'), end=random_dt+pd.Timedelta(23, unit='H'), freq='H')
                no_choice_dt_range = pd.DatetimeIndex(
                    np.unique(np.hstack([no_choice_dt_range, negative_dt_range_with_padding])))
                dts_to_choose_from = dts_to_choose_from.difference(
                    no_choice_dt_range)
                if verbose >= 2:
                    print('number of dts to choose from: {}'.format(
                        len(dts_to_choose_from)))
                cnt += 1
            neg_da = xr.DataArray(negatives, dims=['sample', 'feature'])
            neg_da['feature'] = ['{}_{}'.format(
                'pwv', x) for x in np.arange(1, 25)]
            neg_samples = [x.index[0] for x in negatives]
            neg_da['sample'] = neg_samples
            neg_pre_da = xr.DataArray(
                negatives_pressure, dims=['sample', 'feature'])
            neg_pre_da['feature'] = ['{}_{}'.format(
                'pressure', x) for x in np.arange(1, 25)]
            neg_pre_samples = [x.index[0] for x in negatives_pressure]
            neg_pre_da['sample'] = neg_pre_samples
            neg_da = xr.concat([neg_da, neg_pre_da], 'feature')
            neg_da = neg_da.sortby('sample')
            neg_stns.append(neg_da)
        da_stns = xr.concat(neg_stns, 'sample')
        da_stns = da_stns.sortby('sample')
        # now loop over the remaining features (which are stns agnostic)
        # and add them with the same negative datetimes of the pwv already aquired:
        dts = [pd.date_range(x.item(), periods=24, freq='H')
                for x in da_stns['sample']]
        dts_samples = [x[0] for x in dts]
        other_feat_list = []
        for feat in feats[other_feats]:
            # other_feat_sample_list = []
            da_other = xr.DataArray(feats[feat].sel(time=dts_samples).values, dims=['sample'])
            # for dt in dts_samples:
            #     da_other = xr.DataArray(feats[feat].sel(
            #         time=dt).values, dims=['feature'])
            da_other['sample'] = dts_samples
            other_feat_list.append(da_other)
            # other_feat_da = xr.concat(other_feat_sample_list, 'feature')
        da_other_feats = xr.concat(other_feat_list, 'feature')
        da_other_feats['feature'] = other_feats
        da_stns = xr.concat([da_stns, da_other_feats], 'feature')
        neg_batches.append(da_stns)
    neg_batch_da = xr.concat(neg_batches, 'sample')
    # neg_batch_da['batch'] = np.arange(1, batches + 1)
    neg_batch_da.name = 'X_neg'
    feats['X_neg'] = neg_batch_da
    feats['X_pos'] = all_tides
    feats['X_pwv_stns'] = tides
    # feats['tide_datetimes'] = pos_tides
    feats = feats.rename({'sample': 'negative_sample'})
    filename = 'hydro_tides_hourly_features_with_positives_negatives_{}.nc'.format(
        batches)
    save_ncfile(feats, hydro_path, filename)
    return neg_batch_da


def produce_positives_from_feature_file(hydro_path=hydro_path):
    import xarray as xr
    import pandas as pd
    import numpy as np
    from aux_gps import save_ncfile
    # load features:
    file = hydro_path / 'hydro_tides_hourly_features.nc'
    feats = xr.load_dataset(file)
    feats = feats.rename({'doy': 'DOY'})
    # load positive event for each station:
    dfs = [read_station_from_tide_database(hydro_pw_dict.get(
        x), rounding='1H') for x in hydro_pw_dict.keys()]
    dfs = check_if_tide_events_from_stations_are_within_time_window(
        dfs, days=1, rounding=None, return_hs_list=True)
    da_list = []
    positives_per_station = []
    for i, feat in enumerate(feats):
        try:
            _, _, pr = produce_pwv_days_before_tide_events(feats[feat], dfs[i],
                                                           plot=False, rolling=None,
                                                           days_prior=1,
                                                           drop_thresh=0.75,
                                                           max_gap='6H',
                                                           verbose=0)
            print('getting positives from station {}'.format(feat))

            positives = [pd.to_datetime(
                (x[-1].time + pd.Timedelta(1, unit='H')).item()) for x in pr]
            da = xr.DataArray(pr, dims=['sample', 'feature'])
            da['sample'] = positives
            positives_per_station.append(positives)
            da['feature'] = ['pwv_{}'.format(x) for x in np.arange(1, 25)]
            da_list.append(da)
        except IndexError:
            continue
    da_pwv = xr.concat(da_list, 'sample')
    da_pwv = da_pwv.sortby('sample')
    # now add more features:
    da_list = []
    for feat in ['bet-dagan']:
        print('getting positives from feature {}'.format(feat))
        positives = []
        for dt_end in da_pwv.sample:
            dt_st = pd.to_datetime(dt_end.item()) - pd.Timedelta(24, unit='H')
            dt_end_end = pd.to_datetime(
                dt_end.item()) - pd.Timedelta(1, unit='H')
            positive = feats[feat].sel(time=slice(dt_st, dt_end_end))
            positives.append(positive)
        da = xr.DataArray(positives, dims=['sample', 'feature'])
        da['sample'] = da_pwv.sample
        if feat == 'bet-dagan':
            feat_name = 'pressure'
        else:
            feat_name = feat
        da['feature'] = ['{}_{}'.format(feat_name, x)
                         for x in np.arange(1, 25)]
        da_list.append(da)
    da_f = xr.concat(da_list, 'feature')
    da_list = []
    for feat in ['DOY', 'doy_sin', 'doy_cos']:
        print('getting positives from feature {}'.format(feat))
        positives = []
        for dt in da_pwv.sample:
            positive = feats[feat].sel(time=dt)
            positives.append(positive)
        da = xr.DataArray(positives, dims=['sample'])
        da['sample'] = da_pwv.sample
        # da['feature'] = feat
        da_list.append(da)
    da_ff = xr.concat(da_list, 'feature')
    da_ff['feature'] = ['DOY', 'doy_sin', 'doy_cos']
    da = xr.concat([da_pwv, da_f, da_ff], 'feature')
    filename = 'hydro_tides_hourly_features_with_positives.nc'
    feats['X_pos'] = da
    # now add positives per stations:
    pdf = pd.DataFrame(positives_per_station).T
    pdf.index.name = 'tide_event'
    pos_da = pdf.to_xarray().to_array('GNSS')
    pos_da['GNSS'] = [x for x in hydro_pw_dict.keys()]
    pos_da.attrs['info'] = 'contains the datetimes of the tide events per GNSS station.'
    feats['Tides'] = pos_da
    # rename sample to positive sample:
    feats = feats.rename({'sample': 'positive_sample'})
    save_ncfile(feats, hydro_path, filename)
    return feats


def prepare_features_and_save_hourly(work_path=work_yuval, ims_path=ims_path,
                                     savepath=hydro_path):
    import xarray as xr
    from aux_gps import save_ncfile
    import numpy as np
    # pwv = xr.load_dataset(
    #     work_path / 'GNSS_PW_thresh_0_hour_dayofyear_anoms.nc')
    pwv = xr.load_dataset(work_path /'GNSS_PW_thresh_0_hour_dayofyear_anoms_sd.nc')
    pwv_stations = [x for x in hydro_pw_dict.keys()]
    pwv = pwv[pwv_stations]
    # pwv = pwv.rolling(time=12, keep_attrs=True).mean(keep_attrs=True)
    pwv = pwv.resample(time='1H', keep_attrs=True).mean(keep_attrs=True)
    # bd = xr.load_dataset(ims_path / 'IMS_BD_anoms_5min_ps_1964-2020.nc')
    bd = xr.load_dataset(ims_path / 'IMS_BD_hourly_anoms_std_ps_1964-2020.nc')
    # min_time = pwv.dropna('time')['time'].min()
    # bd = bd.sel(time=slice('1996', None)).resample(time='1H').mean()
    bd = bd.sel(time=slice('1996', None))
    pressure = bd['bet-dagan']
    doy = pwv['time'].copy(data=pwv['time'].dt.dayofyear)
    doy.name = 'doy'
    doy_sin = np.sin(doy * np.pi / 183)
    doy_sin.name = 'doy_sin'
    doy_cos = np.cos(doy * np.pi / 183)
    doy_cos.name = 'doy_cos'
    ds = xr.merge([pwv, pressure, doy, doy_sin, doy_cos])
    filename = 'hydro_tides_hourly_features.nc'
    save_ncfile(ds, savepath, filename)
    return ds


def plot_all_decompositions(X, y, n=2):
    import xarray as xr
    models = [
        'PCA',
        'LDA',
        'ISO_MAP',
        'LLE',
        'LLE-modified',
        'LLE-hessian',
        'LLE-ltsa',
        'MDA',
        'RTE',
        'SE',
        'TSNE',
        'NCA']
    names = [
        'Principal Components',
        'Linear Discriminant',
        'Isomap',
        'Locally Linear Embedding',
        'Modified LLE',
        'Hessian LLE',
        'Local Tangent Space Alignment',
        'MDS embedding',
        'Random forest',
        'Spectral embedding',
        't-SNE',
        'NCA embedding']
    name_dict = dict(zip(models, names))
    da = xr.DataArray(models, dims=['model'])
    da['model'] = models
    fg = xr.plot.FacetGrid(da, col='model', col_wrap=4,
                           sharex=False, sharey=False)
    for model_str, ax in zip(da['model'].values, fg.axes.flatten()):
        model = model_str.split('-')[0]
        method = model_str.split('-')[-1]
        if model == method:
            method = None
        try:
            ax = scikit_decompose(X, y, model=model, n=n, method=method, ax=ax)
        except ValueError:
            pass
        ax.set_title(name_dict[model_str])
        ax.set_xlabel('')
        ax.set_ylabel('')
    fg.fig.suptitle('various decomposition projections (n={})'.format(n))
    return


def scikit_decompose(X, y, model='PCA', n=2, method=None, ax=None):
    from sklearn import (manifold, decomposition, ensemble,
                         discriminant_analysis, neighbors)
    import matplotlib.pyplot as plt
    import pandas as pd
    # from mpl_toolkits.mplot3d import Axes3D
    n_neighbors = 30
    if model == 'PCA':
        X_decomp = decomposition.TruncatedSVD(n_components=n).fit_transform(X)
    elif model == 'LDA':
        X2 = X.copy()
        X2.values.flat[::X.shape[1] + 1] += 0.01
        X_decomp = discriminant_analysis.LinearDiscriminantAnalysis(n_components=n
                                                                    ).fit_transform(X2, y)
    elif model == 'ISO_MAP':
        X_decomp = manifold.Isomap(
            n_neighbors, n_components=n).fit_transform(X)
    elif model == 'LLE':
        # method = 'standard', 'modified', 'hessian' 'ltsa'
        if method is None:
            method = 'standard'
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                              method=method)
        X_decomp = clf.fit_transform(X)
    elif model == 'MDA':
        clf = manifold.MDS(n_components=n, n_init=1, max_iter=100)
        X_decomp = clf.fit_transform(X)
    elif model == 'RTE':
        hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                               max_depth=5)
        X_transformed = hasher.fit_transform(X)
        pca = decomposition.TruncatedSVD(n_components=n)
        X_decomp = pca.fit_transform(X_transformed)
    elif model == 'SE':
        embedder = manifold.SpectralEmbedding(n_components=n, random_state=0,
                                              eigen_solver="arpack")
        X_decomp = embedder.fit_transform(X)
    elif model == 'TSNE':
        tsne = manifold.TSNE(n_components=n, init='pca', random_state=0)
        X_decomp = tsne.fit_transform(X)
    elif model == 'NCA':
        nca = neighbors.NeighborhoodComponentsAnalysis(init='random',
                                                       n_components=n, random_state=0)
        X_decomp = nca.fit_transform(X, y)

    df = pd.DataFrame(X_decomp)
    df.columns = [
        '{}_{}'.format(
            model,
            x +
            1) for x in range(
            X_decomp.shape[1])]
    df['flood'] = y
    df['flood'] = df['flood'].astype(int)
    df_1 = df[df['flood'] == 1]
    df_0 = df[df['flood'] == 0]
    if X_decomp.shape[1] == 1:
        if ax is not None:
            df_1.plot.scatter(ax=ax,
                              x='{}_1'.format(model),
                              y='{}_1'.format(model),
                              color='b', marker='s', alpha=0.3,
                              label='1',
                              s=50)
        else:
            ax = df_1.plot.scatter(
                x='{}_1'.format(model),
                y='{}_1'.format(model),
                color='b',
                label='1',
                s=50)
        df_0.plot.scatter(
            ax=ax,
            x='{}_1'.format(model),
            y='{}_1'.format(model),
            color='r', marker='x',
            label='0',
            s=50)
    elif X_decomp.shape[1] == 2:
        if ax is not None:
            df_1.plot.scatter(ax=ax,
                              x='{}_1'.format(model),
                              y='{}_2'.format(model),
                              color='b', marker='s', alpha=0.3,
                              label='1',
                              s=50)
        else:
            ax = df_1.plot.scatter(
                x='{}_1'.format(model),
                y='{}_2'.format(model),
                color='b',
                label='1',
                s=50)
        df_0.plot.scatter(
            ax=ax,
            x='{}_1'.format(model),
            y='{}_2'.format(model),
            color='r',
            label='0',
            s=50)
    elif X_decomp.shape[1] == 3:
        ax = plt.figure().gca(projection='3d')
        # df_1.plot.scatter(x='{}_1'.format(model), y='{}_2'.format(model), z='{}_3'.format(model), color='b', label='1', s=50, ax=threedee)
        ax.scatter(df_1['{}_1'.format(model)],
                   df_1['{}_2'.format(model)],
                   df_1['{}_3'.format(model)],
                   color='b',
                   label='1',
                   s=50)
        ax.scatter(df_0['{}_1'.format(model)],
                   df_0['{}_2'.format(model)],
                   df_0['{}_3'.format(model)],
                   color='r',
                   label='0',
                   s=50)
        ax.set_xlabel('{}_1'.format(model))
        ax.set_ylabel('{}_2'.format(model))
        ax.set_zlabel('{}_3'.format(model))
    return ax


def permutation_scikit(X, y, cv=False, plot=True):
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import permutation_test_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    if not cv:
        clf = SVC(C=0.01, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=0.032374575428176434,
                  kernel='poly', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
        clf = SVC(kernel='linear')
#        clf = LinearDiscriminantAnalysis()
        cv = StratifiedKFold(4, shuffle=True)
        # cv = KFold(4, shuffle=True)
        n_classes = 2
        score, permutation_scores, pvalue = permutation_test_score(
            clf, X, y, scoring="f1", cv=cv, n_permutations=1000, n_jobs=-1, verbose=2)

        print("Classification score %s (pvalue : %s)" % (score, pvalue))
        plt.hist(permutation_scores, 20, label='Permutation scores',
                 edgecolor='black')
        ylim = plt.ylim()
        plt.plot(2 * [score], ylim, '--g', linewidth=3,
                 label='Classification Score'
                 ' (pvalue %s)' % pvalue)
        plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Score')
        plt.show()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42)
        param_grid = {
            'C': np.logspace(-2, 3, 50), 'gamma': np.logspace(-2, 3, 50),
            'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(X_test)
        print(confusion_matrix(y_test, grid_predictions))
        print(classification_report(y_test, grid_predictions))
    return


def grab_y_true_and_predict_from_sklearn_model(model, X, y, cv,
                                               kfold_name='inner_kfold'):
    from sklearn.model_selection import GridSearchCV
    import xarray as xr
    import numpy as np
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    ds_list = []
    for i, (train, val) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        y_true = y[val]
        y_pred = model.predict(X[val])
        try:
            lr_probs = model.predict_proba(X[val])
            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]
        except AttributeError:
            lr_probs = model.decision_function(X[val])
        y_true_da = xr.DataArray(y_true, dims=['sample'])
        y_pred_da = xr.DataArray(y_pred, dims=['sample'])
        y_prob_da = xr.DataArray(lr_probs, dims=['sample'])
        ds = xr.Dataset()
        ds['y_true'] = y_true_da
        ds['y_pred'] = y_pred_da
        ds['y_prob'] = y_prob_da
        ds['sample'] = np.arange(0, len(X[val]))
        ds_list.append(ds)
    ds = xr.concat(ds_list, kfold_name)
    ds[kfold_name] = np.arange(1, cv.n_splits + 1)
    return ds


def produce_ROC_curves_from_model(model, X, y, cv, kfold_name='inner_kfold'):
    import numpy as np
    import xarray as xr
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    # TODO: collect all predictions and y_tests from this, also predict_proba
    # and save, then calculte everything elsewhere.
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    tprs = []
    aucs = []
    pr = []
    pr_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, val) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[val])
        try:
            lr_probs = model.predict_proba(X[val])
            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]
        except AttributeError:
            lr_probs = model.decision_function(X[val])
        fpr, tpr, _ = roc_curve(y[val], y_pred)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(y[val], y_pred))
        precision, recall, _ = precision_recall_curve(y[val], lr_probs)
        pr.append(recall)
        average_precision = average_precision_score(y[val], y_pred)
        pr_aucs.append(average_precision)
#    mean_tpr = np.mean(tprs, axis=0)
#    mean_tpr[-1] = 1.0
#    mean_auc = auc(mean_fpr, mean_tpr)
#    std_auc = np.std(aucs)
#    std_tpr = np.std(tprs, axis=0)
    tpr_da = xr.DataArray(tprs, dims=[kfold_name, 'fpr'])
    auc_da = xr.DataArray(aucs, dims=[kfold_name])
    ds = xr.Dataset()
    ds['TPR'] = tpr_da
    ds['AUC'] = auc_da
    ds['fpr'] = mean_fpr
    ds[kfold_name] = np.arange(1, cv.n_splits + 1)
    # variability for each tpr is ds['TPR'].std('kfold')
    return ds


def cross_validation_with_holdout(X, y, model_name='SVC', features='pwv',
                                  n_splits=3, test_ratio=0.25,
                                  scorers=['f1', 'recall', 'tss', 'hss',
                                           'precision', 'accuracy'],
                                  seed=42, savepath=None, verbose=0,
                                  param_grid='normal', n_jobs=-1,
                                  n_repeats=None):
    # from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import make_scorer
    # from string import digits
    import numpy as np
    # import xarray as xr
    scores_dict = {s: s for s in scorers}
    if 'tss' in scorers:
        scores_dict['tss'] = make_scorer(tss_score)
    if 'hss' in scorers:
        scores_dict['hss'] = make_scorer(hss_score)

    X = select_doy_from_feature_list(X, model_name, features)
    if param_grid == 'light':
        print(np.unique(X.feature.values))
    # first take out the hold-out set:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,
                                                        random_state=seed,
                                                        stratify=y)
    if n_repeats is None:
        # configure the cross-validation procedure
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=seed)
        print('CV StratifiedKfolds of {}.'.format(n_splits))
        # define the model and search space:
    else:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        print('CV RepeatedStratifiedKFold of {} with {} repeats.'.format(n_splits, n_repeats))
    ml = ML_Classifier_Switcher()
    print('param grid group is set to {}.'.format(param_grid))
    sk_model = ml.pick_model(model_name, pgrid=param_grid)
    search_space = ml.param_grid
    # define search
    gr_search = GridSearchCV(estimator=sk_model, param_grid=search_space,
                             cv=cv, n_jobs=n_jobs,
                             scoring=scores_dict,
                             verbose=verbose,
                             refit=False, return_train_score=True)

    gr_search.fit(X, y)
    if isinstance(features, str):
        features = [features]
    if savepath is not None:
        filename = 'GRSRCHCV_holdout_{}_{}_{}_{}_{}_{}_{}.pkl'.format(
            model_name, '+'.join(features), '+'.join(scorers), n_splits,
            int(test_ratio*100), param_grid, seed)
        save_gridsearchcv_object(gr_search, savepath, filename)
    # gr, _ = process_gridsearch_results(
    #         gr_search, model_name, split_dim='kfold', features=X.feature.values)
    # remove_digits = str.maketrans('', '', digits)
    # features = list(set([x.translate(remove_digits).split('_')[0]
    #                      for x in X.feature.values]))
    # # add more attrs, features etc:
    # gr.attrs['features'] = features

    return gr_search


def select_doy_from_feature_list(X, model_name='RF', features='pwv'):
        # first if RF chosen, replace the cyclic coords of DOY (sin and cos) with
    # the DOY itself.
    if isinstance(features, list):
        feats = features.copy()
    else:
        feats = features
    if model_name == 'RF' and 'doy' in features:
        if isinstance(features, list):
            feats.remove('doy')
            feats.append('DOY')
        elif isinstance(features, str):
            feats = 'DOY'
    elif model_name != 'RF' and 'doy' in features:
        if isinstance(features, list):
            feats.remove('doy')
            feats.append('doy_sin')
            feats.append('doy_cos')
        elif isinstance(features, str):
            feats = ['doy_sin']
            feats.append('doy_cos')
    X = select_features_from_X(X, feats)
    return X


def single_cross_validation(X_val, y_val, model_name='SVC', features='pwv',
                            n_splits=4, scorers=['f1', 'recall', 'tss', 'hss',
                                                 'precision', 'accuracy'],
                            seed=42, savepath=None, verbose=0,
                            param_grid='normal', n_jobs=-1,
                            n_repeats=None, outer_split='1-1'):
    # from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import train_test_split
    from sklearn.metrics import make_scorer
    # from string import digits
    import numpy as np
    # import xarray as xr
    scores_dict = {s: s for s in scorers}
    if 'tss' in scorers:
        scores_dict['tss'] = make_scorer(tss_score)
    if 'hss' in scorers:
        scores_dict['hss'] = make_scorer(hss_score)

    X = select_doy_from_feature_list(X_val, model_name, features)
    y = y_val

    if param_grid == 'light':
        print(np.unique(X.feature.values))

    if n_repeats is None:
        # configure the cross-validation procedure
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=seed)
        print('CV StratifiedKfolds of {}.'.format(n_splits))
        # define the model and search space:
    else:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        print('CV RepeatedStratifiedKFold of {} with {} repeats.'.format(
            n_splits, n_repeats))
    ml = ML_Classifier_Switcher()
    print('param grid group is set to {}.'.format(param_grid))
    if outer_split == '1-1':
        cv_type = 'holdout'
        print('holdout cv is selected.')
    else:
        cv_type = 'nested'
        print('nested cv {} out of {}.'.format(
            outer_split.split('-')[0], outer_split.split('-')[1]))
    sk_model = ml.pick_model(model_name, pgrid=param_grid)
    search_space = ml.param_grid
    # define search
    gr_search = GridSearchCV(estimator=sk_model, param_grid=search_space,
                             cv=cv, n_jobs=n_jobs,
                             scoring=scores_dict,
                             verbose=verbose,
                             refit=False, return_train_score=True)

    gr_search.fit(X, y)
    if isinstance(features, str):
        features = [features]
    if savepath is not None:
        filename = 'GRSRCHCV_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(cv_type,
                                                                 model_name, '+'.join(features), '+'.join(
                                                                     scorers), n_splits,
                                                                 outer_split, param_grid, seed)
        save_gridsearchcv_object(gr_search, savepath, filename)
    return gr_search


def save_cv_params_to_file(cv_obj, path, name):
    import pandas as pd
    di = vars(cv_obj)
    splitter_type = cv_obj.__repr__().split('(')[0]
    di['splitter_type'] = splitter_type
    (pd.DataFrame.from_dict(data=di, orient='index')
     .to_csv(path / '{}.csv'.format(name), header=False))
    print('{}.csv saved to {}.'.format(name, path))
    return


def read_cv_params_and_instantiate(filepath):
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    df = pd.read_csv(filepath, header=None, index_col=0)
    d = {}
    for row in df.iterrows():
        dd = pd.to_numeric(row[1], errors='ignore')
        if dd.item() == 'True' or dd.item() == 'False':
            dd = dd.astype(bool)
        d[dd.to_frame().columns.item()] = dd.item()
    s_type = d.pop('splitter_type')
    if s_type == 'StratifiedKFold':
        cv = StratifiedKFold(**d)
    return cv


def nested_cross_validation_procedure(X, y, model_name='SVC', features='pwv',
                                      outer_splits=4, inner_splits=2,
                                      refit_scorer='roc_auc',
                                      scorers=['f1', 'recall', 'tss', 'hss',
                                               'roc_auc', 'precision',
                                               'accuracy'],
                                      seed=42, savepath=None, verbose=0,
                                      param_grid='normal', n_jobs=-1):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.inspection import permutation_importance
    from string import digits
    import numpy as np
    import xarray as xr

    assert refit_scorer in scorers
    scores_dict = {s: s for s in scorers}
    if 'tss' in scorers:
        scores_dict['tss'] = make_scorer(tss_score)
    if 'hss' in scorers:
        scores_dict['hss'] = make_scorer(hss_score)

    X = select_doy_from_feature_list(X, model_name, features)
    # if model_name == 'RF':
    #     doy = X['sample'].dt.dayofyear
    #     sel_doy = [x for x in X.feature.values if 'doy_sin' in x]
    #     doy_X = doy.broadcast_like(X.sel(feature=sel_doy))
    #     doy_X['feature'] = [
    #         'doy_{}'.format(x) for x in range(
    #             doy_X.feature.size)]
    #     no_doy = [x for x in X.feature.values if 'doy' not in x]
    #     X = X.sel(feature=no_doy)
    #     X = xr.concat([X, doy_X], 'feature')
    # else:
    #     # first slice X for features:
    #     if isinstance(features, str):
    #         f = [x for x in X.feature.values if features in x]
    #         X = X.sel(feature=f)
    #     elif isinstance(features, list):
    #         fs = []
    #         for f in features:
    #             fs += [x for x in X.feature.values if f in x]
    #         X = X.sel(feature=fs)
    if param_grid == 'light':
        print(np.unique(X.feature.values))
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                               random_state=seed)
    print('Inner CV StratifiedKfolds of {}.'.format(inner_splits))
    # define the model and search space:
    ml = ML_Classifier_Switcher()
    if param_grid == 'light':
        print('disgnostic mode light.')
    sk_model = ml.pick_model(model_name, pgrid=param_grid)
    search_space = ml.param_grid
    # define search
    gr_search = GridSearchCV(estimator=sk_model, param_grid=search_space,
                             cv=cv_inner, n_jobs=n_jobs,
                             scoring=scores_dict,
                             verbose=verbose,
                             refit=refit_scorer, return_train_score=True)
#    gr.fit(X, y)
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=seed)
    # execute the nested cross-validation
    scores_est_dict = cross_validate(gr_search, X, y,
                                     scoring=scores_dict,
                                     cv=cv_outer, n_jobs=n_jobs,
                                     return_estimator=True, verbose=verbose)
#    perm = []
#    for i, (train, val) in enumerate(cv_outer.split(X, y)):
#        gr_model = scores_est_dict['estimator'][i]
#        gr_model.fit(X[train], y[train])
#        r = permutation_importance(gr_model, X[val], y[val],scoring='f1',
#                                   n_repeats=30, n_jobs=-1,
#                                   random_state=0)
#        perm.append(r)
    # get the test scores:
    test_keys = [x for x in scores_est_dict.keys() if 'test' in x]
    ds = xr.Dataset()
    for key in test_keys:
        ds[key] = xr.DataArray(scores_est_dict[key], dims=['outer_kfold'])
    preds_ds = []
    gr_ds = []
    for est in scores_est_dict['estimator']:
        gr, _ = process_gridsearch_results(
            est, model_name, split_dim='inner_kfold', features=X.feature.values)
        # somehow save gr:
        gr_ds.append(gr)
        preds_ds.append(
            grab_y_true_and_predict_from_sklearn_model(est, X, y, cv_inner))
#        tpr_ds.append(produce_ROC_curves_from_model(est, X, y, cv_inner))
    dss = xr.concat(preds_ds, 'outer_kfold')
    gr_dss = xr.concat(gr_ds, 'outer_kfold')
    dss['outer_kfold'] = np.arange(1, cv_outer.n_splits + 1)
    gr_dss['outer_kfold'] = np.arange(1, cv_outer.n_splits + 1)
    # aggragate results:
    dss = xr.merge([ds, dss])
    dss = xr.merge([dss, gr_dss])
    dss.attrs = gr_dss.attrs
    dss.attrs['outer_kfold_splits'] = outer_splits
    remove_digits = str.maketrans('', '', digits)
    features = list(set([x.translate(remove_digits).split('_')[0]
                         for x in X.feature.values]))
    # add more attrs, features etc:
    dss.attrs['features'] = features

    # rename major data_vars with model name:
    # ys = [x for x in dss.data_vars if 'y_' in x]
    # new_ys = [y + '_{}'.format(model_name) for y in ys]
    # dss = dss.rename(dict(zip(ys, new_ys)))
    # new_test_keys = [y + '_{}'.format(model_name) for y in test_keys]
    # dss = dss.rename(dict(zip(test_keys, new_test_keys)))

    # if isinstance(X.attrs['pwv_id'], list):
    #     dss.attrs['pwv_id'] = '-'.join(X.attrs['pwv_id'])
    # else:
    #     dss.attrs['pwv_id'] = X.attrs['pwv_id']
    # if isinstance(y.attrs['hydro_station_id'], list):
    #     dss.attrs['hs_id'] = '-'.join([str(x) for x in y.attrs['hydro_station_id']])
    # else:
    #     dss.attrs['hs_id'] = y.attrs['hydro_station_id']
    # dss.attrs['hydro_max_flow'] = y.attrs['max_flow']
    # dss.attrs['neg_pos_ratio'] = y.attrs['neg_pos_ratio']
    # save results to file:
    if savepath is not None:
        save_cv_results(dss, savepath=savepath)
    return dss


# def ML_main_procedure(X, y, estimator=None, model_name='SVC', features='pwv',
#                       val_size=0.18, n_splits=None, test_size=0.2, seed=42, best_score='f1',
#                       savepath=None, plot=True):
#     """split the X,y for train and test, either do HP tuning using HP_tuning
#     with val_size or use already tuned (or not) estimator.
#     models to play with = MLP, RF and SVC.
#     n_splits = 2, 3, 4.
#     features = pwv, pressure.
#     best_score = f1, roc_auc, accuracy.
#     can do loop on them. RF takes the most time to tune."""
#     X = select_features_from_X(X, features)
#     X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                         test_size=test_size,
#                                                         shuffle=True,
#                                                         random_state=seed)
#     # do HP_tuning:
#     if estimator is None:
#         cvr, model = HP_tuning(X_train, y_train, model_name=model_name, val_size=val_size, test_size=test_size,
#                         best_score=best_score, seed=seed, savepath=savepath, n_splits=n_splits)
#     else:
#         model = estimator
#     if plot:
#         ax = plot_many_ROC_curves(model, X_test, y_test, name=model_name,
#                                   ax=None)
#         return ax
#     else:
#         return model


def plot_hyper_parameters_heatmaps_from_nested_CV_model(path=hydro_path, model_name='MLP',
                                                        features='pwv+pressure+doy', save=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    ds = xr.load_dataset(
        path / 'nested_CV_test_results_{}_all_features_with_hyper.nc'.format(model_name))
    ds = ds.sel(features=features).reset_coords(drop=True)
    non_hp_vars = ['mean_score', 'std_score',
                   'test_score', 'roc_auc_score', 'TPR']
    if model_name == 'RF':
        non_hp_vars.append('feature_importances')
    ds = ds[[x for x in ds if x not in non_hp_vars]]
    seq = 'Blues'
    cat = 'Dark2'
    cmap_hp_dict = {
        'alpha': seq, 'activation': cat,
        'hidden_layer_sizes': cat, 'learning_rate': cat,
        'solver': cat, 'kernel': cat, 'C': seq,
        'gamma': seq, 'degree': seq, 'coef0': seq,
        'max_depth': seq, 'max_features': cat,
        'min_samples_leaf': seq, 'min_samples_split': seq,
        'n_estimators': seq
    }
    # fix stuff for SVC:
    if model_name == 'SVC':
        ds['degree'] = ds['degree'].where(ds['kernel']=='poly')
        ds['coef0'] = ds['coef0'].where(ds['kernel']=='poly')
    # da = ds.to_arrray('hyper_parameters')
    # fg = xr.plot.FacetGrid(
    #     da,
    #     col='hyper_parameters',
    #     sharex=False,
    #     sharey=False, figsize=(16, 10))
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 10))
    for i, da in enumerate(ds):
        df = ds[da].reset_coords(drop=True).to_dataset('scorer').to_dataframe()
        df.index.name = 'Outer Split'
        cmap = cmap_hp_dict.get(da, 'Set1')
        plot_heatmap_for_hyper_parameters_df(df, ax=axes[i], title=da, cmap=cmap)
    fig.tight_layout()
    if save:
        filename = 'Hyper-parameters_nested_{}.png'.format(
            model_name)
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_heatmap_for_hyper_parameters_df(df, ax=None, cmap='colorblind',
                                         title=None, fontsize=12):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    sns.set_style('ticks')
    sns.set_style('whitegrid')
    sns.set(font_scale=1.2)
    value_to_int = {j: i for i, j in enumerate(
        sorted(pd.unique(df.values.ravel())))} # like you did
    # for key in value_to_int.copy().keys():
    #     try:
    #         if np.isnan(key):
    #             value_to_int['NA'] = value_to_int.pop(key)
    #             df = df.fillna('NA')
    #     except TypeError:
    #         pass
    try:
        sorted_v_to_i = dict(sorted(value_to_int.items()))
    except TypeError:
        sorted_v_to_i = value_to_int
    n = len(value_to_int)
    # discrete colormap (n samples from a given cmap)
    cmap = sns.color_palette(cmap, n)
    if ax is None:
        ax = sns.heatmap(df.replace(sorted_v_to_i), cmap=cmap,
                         linewidth=1, linecolor='k', square=False,
                         cbar_kws={"shrink": .9})
    else:
        ax = sns.heatmap(df.replace(sorted_v_to_i), cmap=cmap,
                         ax=ax, linewidth=1, linecolor='k',
                         square=False, cbar_kws={"shrink": .9})
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(value_to_int.keys()))
    return ax


def plot_ROC_curves_for_all_models_and_scorers(dss, save=False,
                                               fontsize=24, fig_split=1,
                                               feat=['pwv', 'pwv+pressure', 'pwv+pressure+doy']):
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    cmap = sns.color_palette('tab10', len(feat))
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    if fig_split == 1:
        dss = dss.sel(scorer=['precision', 'recall', 'f1'])
    elif fig_split == 2:
        dss = dss.sel(scorer=['accuracy', 'tss', 'hss'])
    fg = xr.plot.FacetGrid(
        dss,
        col='model',
        row='scorer',
        sharex=True,
        sharey=True, figsize=(20, 20))
    for i in range(fg.axes.shape[0]):  # i is rows
        for j in range(fg.axes.shape[1]):  # j is cols
            ax = fg.axes[i, j]
            modelname = dss['model'].isel(model=j).item()
            scorer = dss['scorer'].isel(scorer=i).item()
            chance_plot = [False for x in feat]
            chance_plot[-1] = True
            for k, f in enumerate(feat):
                #     name = '{}-{}-{}'.format(modelname, scoring, feat)
                # model = dss.isel({'model': j, 'scoring': i}).sel(
                #     {'features': feat})
                model = dss.isel({'model': j, 'scorer': i}
                                 ).sel({'features': f})
                # return model
                title = 'ROC of {} model ({})'.format(modelname.replace('SVC', 'SVM'), scorer)
                try:
                    ax = plot_ROC_curve_from_dss_nested_CV(model, outer_dim='outer_split',
                                                           plot_chance=[k],
                                                           main_label=f,
                                                           ax=ax,
                                                           color=cmap[k], title=title,
                                                           fontsize=fontsize)
                except ValueError:
                    ax.grid('on')
                    continue
            handles, labels = ax.get_legend_handles_labels()
            lh_ser = pd.Series(labels, index=handles).drop_duplicates()
            lh_ser = lh_ser.sort_values(ascending=False)
            hand = lh_ser.index.values
            labe = lh_ser.values
            ax.legend(handles=hand.tolist(), labels=labe.tolist(), loc="lower right",
                      fontsize=fontsize-7)
            ax.grid('on')
            if j >= 1:
                ax.set_ylabel('')
            if fig_split == 1:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)
            else:
                if i <= 1:
                    ax.set_xlabel('')
    # title = '{} station: {} total events'.format(
    #         station.upper(), events)
    # if max_flow > 0:
    #     title = '{} station: {} total events (max flow = {} m^3/sec)'.format(
    #         station.upper(), events, max_flow)
    # fg.fig.suptitle(title, fontsize=fontsize)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.937,
                           bottom=0.054,
                           left=0.039,
                           right=0.993,
                           hspace=0.173,
                           wspace=0.051)
    if save:
        filename = 'ROC_curves_nested_{}_figsplit_{}.png'.format(
            dss['outer_split'].size, fig_split)
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_hydro_ML_models_results_from_dss(dss, std_on='outer',
                                          save=False, fontsize=16,
                                          plot_type='ROC', split=1,
                                          feat=['pwv', 'pressure+pwv', 'doy+pressure+pwv']):
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    cmap = sns.color_palette("colorblind", len(feat))
    if split == 1:
        dss = dss.sel(scoring=['f1', 'precision', 'recall'])
    elif split == 2:
        dss = dss.sel(scoring=['tss', 'hss', 'roc-auc', 'accuracy'])
    fg = xr.plot.FacetGrid(
        dss,
        col='model',
        row='scoring',
        sharex=True,
        sharey=True, figsize=(20, 20))
    for i in range(fg.axes.shape[0]):  # i is rows
        for j in range(fg.axes.shape[1]):  # j is cols
            ax = fg.axes[i, j]
            modelname = dss['model'].isel(model=j).item()
            scoring = dss['scoring'].isel(scoring=i).item()
            chance_plot = [False for x in feat]
            chance_plot[-1] = True
            for k, f in enumerate(feat):
                #     name = '{}-{}-{}'.format(modelname, scoring, feat)
                # model = dss.isel({'model': j, 'scoring': i}).sel(
                #     {'features': feat})
                model = dss.isel({'model': j, 'scoring': i}
                                 ).sel({'features': f})
                title = '{} of {} model ({})'.format(
                    plot_type, modelname, scoring)
                try:
                    plot_ROC_PR_curve_from_dss(model, outer_dim='outer_kfold',
                                               inner_dim='inner_kfold',
                                               plot_chance=[k],
                                               main_label=f, plot_type=plot_type,
                                               plot_std_legend=False, ax=ax,
                                               color=cmap[k], title=title,
                                               std_on=std_on, fontsize=fontsize)
                except ValueError:
                    ax.grid('on')
                    continue
            handles, labels = ax.get_legend_handles_labels()
            hand = pd.Series(
                labels, index=handles).drop_duplicates().index.values
            labe = pd.Series(labels, index=handles).drop_duplicates().values
            ax.legend(handles=hand.tolist(), labels=labe.tolist(), loc="lower right",
                      fontsize=14)
            ax.grid('on')
    # title = '{} station: {} total events'.format(
    #         station.upper(), events)
    # if max_flow > 0:
    #     title = '{} station: {} total events (max flow = {} m^3/sec)'.format(
    #         station.upper(), events, max_flow)
    # fg.fig.suptitle(title, fontsize=fontsize)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.937,
                           bottom=0.054,
                           left=0.039,
                           right=0.993,
                           hspace=0.173,
                           wspace=0.051)
    if save:
        filename = 'hydro_models_on_{}_{}_std_on_{}_{}.png'.format(
            dss['inner_kfold'].size, dss['outer_kfold'].size,
            std_on, plot_type)
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


# def plot_hydro_ML_models_result(model_da, nsplits=2, station='drag',
#                                test_size=20, n_splits_plot=None, save=False):
#    import xarray as xr
#    import seaborn as sns
#    import matplotlib.pyplot as plt
#    from sklearn.model_selection import train_test_split
#    # TODO: add plot_roc_curve(model, X_other_station, y_other_station)
#    # TODO: add pw_station, hs_id
#    cmap = sns.color_palette("colorblind", 3)
#    X, y = produce_X_y(station, hydro_pw_dict[station], neg_pos_ratio=1)
#    events = int(y[y == 1].sum().item())
#    model_da = model_da.sel(
#        splits=nsplits,
#        test_size=test_size).reset_coords(
#            drop=True)
##    just_pw = [x for x in X.feature.values if 'pressure' not in x]
##    X_pw = X.sel(feature=just_pw)
#    fg = xr.plot.FacetGrid(
#        model_da,
#        col='model',
#        row='scoring',
#        sharex=True,
#        sharey=True, figsize=(20, 20))
#    for i in range(fg.axes.shape[0]):  # i is rows
#        for j in range(fg.axes.shape[1]):  # j is cols
#            ax = fg.axes[i, j]
#            modelname = model_da['model'].isel(model=j).item()
#            scoring = model_da['scoring'].isel(scoring=i).item()
#            chance_plot = [False, False, True]
#            for k, feat in enumerate(model_da['feature'].values):
#                name = '{}-{}-{}'.format(modelname, scoring, feat)
#                model = model_da.isel({'model': j, 'scoring': i}).sel({'feature': feat}).item()
#                title = 'ROC of {} model ({})'.format(modelname, scoring)
#                if not '+' in feat:
#                    f = [x for x in X.feature.values if feat in x]
#                    X_f = X.sel(feature=f)
#                else:
#                    X_f = X
# X_train, X_test, y_train, y_test = train_test_split(
# X_f, y, test_size=test_size/100, shuffle=True, random_state=42)
#
#                plot_many_ROC_curves(model, X_f, y, name=name,
#                                     color=cmap[k], ax=ax,
#                                     plot_chance=chance_plot[k],
#                                     title=title, n_splits=n_splits_plot)
#    fg.fig.suptitle('{} station: {} total_events, test_events = {}, n_splits = {}'.format(station.upper(), events, int(events* test_size/100), nsplits))
#    fg.fig.tight_layout()
#    fg.fig.subplots_adjust(top=0.937,
#                           bottom=0.054,
#                           left=0.039,
#                           right=0.993,
#                           hspace=0.173,
#                           wspace=0.051)
#    if save:
#        plt.savefig(savefig_path / 'try.png', bbox_inches='tight')
#    return fg


def order_features_list(flist):
    """ order the feature list in load_ML_run_results
    so i don't get duplicates"""
    import pandas as pd
    import numpy as np
    # first get all features:
    li = [x.split('+') for x in flist]
    flat_list = [item for sublist in li for item in sublist]
    f = list(set(flat_list))
    nums = np.arange(1, len(f)+1)
    # now assagin a number for each entry:
    inds = []
    for x in flist:
        for fe, num in zip(f, nums):
            x = x.replace(fe, str(10**num))
        inds.append(eval(x))
    ser = pd.Series(inds)
    ser.index = flist
    ser1 = ser.drop_duplicates()
    di = dict(zip(ser1.values, ser1.index))
    new_flist = []
    for ind, feat in zip(inds, flist):
        new_flist.append(di.get(ind))
    return new_flist


def smart_add_dataarray_to_ds_list(dsl, da_name='feature_importances'):
    """add data array to ds_list even if it does not exist, use shape of
    data array that exists in other part of ds list"""
    import numpy as np
    import xarray as xr
    # print(da_name)
    fi = [x for x in dsl if da_name in x][0]
    print(da_name, fi[da_name].shape)
    fi = fi[da_name].copy(data=np.zeros(shape=fi[da_name].shape))
    new_dsl = []
    for ds in dsl:
        if da_name not in ds:
            ds = xr.merge([ds, fi], combine_attrs='no_conflicts')
        new_dsl.append(ds)
    return new_dsl

def load_ML_run_results(path=hydro_ml_path, prefix='CVR',
                        change_DOY_to_doy=True):
    from aux_gps import path_glob
    import xarray as xr
#    from aux_gps import save_ncfile
    import pandas as pd
    import numpy as np

    print('loading hydro ML results for all models and features')
    # print('loading hydro ML results for station {}'.format(pw_station))
    model_files = path_glob(path, '{}_*.nc'.format(prefix))
    model_files = sorted(model_files)
    # model_files = [x for x in model_files if pw_station in x.as_posix()]
    ds_list = [xr.load_dataset(x) for x in model_files]
    if change_DOY_to_doy:
        for ds in ds_list:
            if 'DOY' in ds.features:
                new_feats = [x.replace('DOY', 'doy') for x in ds['feature'].values]
                ds['feature'] = new_feats
                ds.attrs['features'] = [x.replace('DOY', 'doy') for x in ds.attrs['features']]
    model_as_str = [x.as_posix().split('/')[-1].split('.')[0]
                    for x in model_files]
    model_names = [x.split('_')[1] for x in model_as_str]
    model_scores = [x.split('_')[3] for x in model_as_str]
    model_features = [x.split('_')[2] for x in model_as_str]
    if change_DOY_to_doy:
        model_features = [x.replace('DOY', 'doy') for x in model_features]
    new_model_features = order_features_list(model_features)
    ind = pd.MultiIndex.from_arrays(
        [model_names,
            new_model_features,
            model_scores],
        names=(
            'model',
            'features',
            'scoring'))
    #    ind1 = pd.MultiIndex.from_product([model_names, model_scores, model_features], names=[
    #                                     'model', 'scoring', 'feature'])
    #    ds_list = [x[data_vars] for x in ds_list]
    # complete non-existant fields like best and fi for all ds:
    data_vars = [x for x in ds_list[0] if x.startswith('test')]
    #    data_vars += ['AUC', 'TPR']
    data_vars += [x for x in ds_list[0] if x.startswith('y_')]
    bests = [[x for x in y if x.startswith('best')] for y in ds_list]
    data_vars += list(set([y for x in bests for y in x]))
    if 'RF' in model_names:
        data_vars += ['feature_importances']
    new_ds_list = []
    for dvar in data_vars:
        ds_list = smart_add_dataarray_to_ds_list(ds_list, dvar)
    #    # check if all data vars are in each ds and merge them:
    new_ds_list = [xr.merge([y[x] for x in data_vars if x in y],
                            combine_attrs='no_conflicts') for y in ds_list]
    # concat all
    dss = xr.concat(new_ds_list, dim='dim_0')
    dss['dim_0'] = ind
    dss = dss.unstack('dim_0')
    # dss.attrs['pwv_id'] = pw_station
    # fix roc_auc to roc-auc in dss datavars
    dss = dss.rename_vars({'test_roc_auc': 'test_roc-auc'})
    # dss['test_roc_auc'].name = 'test_roc-auc'
    print('calculating ROC, PR metrics.')
    dss = calculate_metrics_from_ML_dss(dss)
    print('Done!')
    return dss


def plot_nested_CV_test_scores(dss, feats='pwv+pressure+doy'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    def show_values_on_bars(axs, fs=12, fw='bold'):
        import numpy as np
        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center",
                        fontsize=fs, fontweight=fw, zorder=20)

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)
    if feats is None:
        feats = ['pwv', 'pwv+pressure', 'pwv+pressure+doy']
    dst = dss.sel(features=feats)  # .reset_coords(drop=True)
    df = dst['test_score'].to_dataframe()
    df['scorer'] = df.index.droplevel(2).droplevel(1).droplevel(0)
    df['model'] = df.index.droplevel(3).droplevel(2).droplevel(1)
    df['features'] = df.index.droplevel(2).droplevel(2).droplevel(0)
    df['outer_splits'] = df.index.droplevel(0).droplevel(2).droplevel(0)
    df['model'] = df['model'].str.replace('SVC', 'SVM')
    df = df.melt(value_vars='test_score', id_vars=[
        'features', 'model', 'scorer', 'outer_splits'], var_name='test_score',
        value_name='score')
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    g = sns.catplot(x="model", y="score", hue='features',
                    col="scorer", ci='sd', row=None,
                    col_wrap=3,
                    data=df, kind="bar", capsize=0.25,
                    height=4, aspect=1.5, errwidth=1.5)
    g.set_xticklabels(rotation=45)
    [x.grid(True) for x in g.axes.flatten()]
    show_values_on_bars(g.axes)
    filename = 'ML_scores_models_nested_CV_{}.png'.format('_'.join(feats))
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return df


def plot_holdout_test_scores(dss, feats='pwv+pressure+doy'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    def show_values_on_bars(axs, fs=12, fw='bold'):
        import numpy as np

        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", fontsize=fs, fontweight=fw)

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)
    if feats is None:
        feats = ['pwv', 'pwv+pressure', 'pwv+pressure+doy']
    dst = dss.sel(features=feats)  # .reset_coords(drop=True)
    df = dst['holdout_test_scores'].to_dataframe()
    df['scorer'] = df.index.droplevel(1).droplevel(0)
    df['model'] = df.index.droplevel(2).droplevel(1)
    df['features'] = df.index.droplevel(2).droplevel(0)
    df['model'] = df['model'].str.replace('SVC', 'SVM')
    df = df.melt(value_vars='holdout_test_scores', id_vars=[
        'features', 'model', 'scorer'], var_name='test_score')
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    g = sns.catplot(x="model", y="value", hue='features',
                    col="scorer", ci='sd', row=None,
                    col_wrap=3,
                    data=df, kind="bar", capsize=0.15,
                    height=4, aspect=1.5, errwidth=0.8)
    g.set_xticklabels(rotation=45)
    [x.grid(True) for x in g.axes.flatten()]
    show_values_on_bars(g.axes)
    filename = 'ML_scores_models_holdout_{}.png'.format('_'.join(feats))
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return df


def prepare_test_df_to_barplot_from_dss(dss, feats='doy+pwv+pressure',
                                        plot=True, splitfigs=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    dvars = [x for x in dss if 'test_' in x]
    scores = [x.split('_')[-1] for x in dvars]
    dst = dss[dvars]
    # dst['scoring'] = [x+'_inner' for x in dst['scoring'].values]
    # for i, ds in enumerate(dst):
    #     dst[ds] = dst[ds].sel(scoring=scores[i]).reset_coords(drop=True)
    if feats is None:
        feats = ['pwv', 'pressure+pwv', 'doy+pressure+pwv']
    dst = dst.sel(features=feats)  # .reset_coords(drop=True)
    dst = dst.rename_vars(dict(zip(dvars, scores)))
    # dst = dst.drop('scoring')
    df = dst.to_dataframe()
    # dfu = df
    df['inner score'] = df.index.droplevel(2).droplevel(1).droplevel(0)
    df['features'] = df.index.droplevel(2).droplevel(2).droplevel(1)
    df['model'] = df.index.droplevel(2).droplevel(0).droplevel(1)
    df = df.melt(value_vars=scores, id_vars=[
        'features', 'model', 'inner score'], var_name='outer score')
    # return dfu
    # dfu.columns = dfu.columns.droplevel(1)
    # dfu = dfu.T
    # dfu['score'] = dfu.index
    # dfu = dfu.reset_index()
    # df = dfu.melt(value_vars=['MLP', 'RF', 'SVC'], id_vars=['score'])
    df1 = df[(df['inner score']=='f1') | (df['inner score']=='precision') | (df['inner score']=='recall')]
    df2 = df[(df['inner score']=='hss') | (df['inner score']=='tss') | (df['inner score']=='roc-auc') | (df['inner score']=='accuracy')]
    if plot:
        sns.set(font_scale = 1.5)
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        if splitfigs:
            g = sns.catplot(x="outer score", y="value", hue='features',
                            col="inner score", ci='sd',row='model',
                            data=df1, kind="bar", capsize=0.15,
                            height=4, aspect=1.5,errwidth=0.8)
            g.set_xticklabels(rotation=45)
            filename = 'ML_scores_models_{}_1.png'.format('_'.join(feats))
            plt.savefig(savefig_path / filename, bbox_inches='tight')
            g = sns.catplot(x="outer score", y="value", hue='features',
                            col="inner score", ci='sd',row='model',
                            data=df2, kind="bar", capsize=0.15,
                            height=4, aspect=1.5,errwidth=0.8)
            g.set_xticklabels(rotation=45)
            filename = 'ML_scores_models_{}_2.png'.format('_'.join(feats))
            plt.savefig(savefig_path / filename, bbox_inches='tight')
        else:
            g = sns.catplot(x="outer score", y="value", hue='features',
                            col="inner score", ci='sd',row='model',
                            data=df, kind="bar", capsize=0.15,
                            height=4, aspect=1.5,errwidth=0.8)
            g.set_xticklabels(rotation=45)
            filename = 'ML_scores_models_{}.png'.format('_'.join(feats))
            plt.savefig(savefig_path / filename, bbox_inches='tight')
    return df


def calculate_metrics_from_ML_dss(dss):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    import xarray as xr
    import numpy as np
    import pandas as pd
    mean_fpr = np.linspace(0, 1, 100)
#    fpr = dss['y_true'].copy(deep=False).values
#    tpr = dss['y_true'].copy(deep=False).values
#    y_true = dss['y_true'].values
#    y_prob = dss['y_prob'].values
    ok = [x for x in dss['outer_kfold'].values]
    ik = [x for x in dss['inner_kfold'].values]
    m = [x for x in dss['model'].values]
    sc = [x for x in dss['scoring'].values]
    f = [x for x in dss['features'].values]
    # r = [x for x in dss['neg_pos_ratio'].values]
    ind = pd.MultiIndex.from_product(
        [ok, ik, m, sc, f],
        names=[
            'outer_kfold',
            'inner_kfold',
            'model',
            'scoring',
            'features'])  # , 'station'])

    okn = [x for x in range(dss['outer_kfold'].size)]
    ikn = [x for x in range(dss['inner_kfold'].size)]
    mn = [x for x in range(dss['model'].size)]
    scn = [x for x in range(dss['scoring'].size)]
    fn = [x for x in range(dss['features'].size)]
    ds_list = []
    for i in okn:
        for j in ikn:
            for k in mn:
                for n in scn:
                    for m in fn:
                        ds = xr.Dataset()
                        y_true = dss['y_true'].isel(
                            outer_kfold=i, inner_kfold=j, model=k, scoring=n, features=m).reset_coords(drop=True).squeeze()
                        y_prob = dss['y_prob'].isel(
                            outer_kfold=i, inner_kfold=j, model=k, scoring=n, features=m).reset_coords(drop=True).squeeze()
                        y_true = y_true.dropna('sample')
                        y_prob = y_prob.dropna('sample')
                        if y_prob.size == 0:
                            # in case of NaNs in the results:
                            fpr_da = xr.DataArray(
                                np.nan*np.ones((1)), dims=['sample'])
                            fpr_da['sample'] = [
                                x for x in range(fpr_da.size)]
                            tpr_da = xr.DataArray(
                                np.nan*np.ones((1)), dims=['sample'])
                            tpr_da['sample'] = [
                                x for x in range(tpr_da.size)]
                            prn_da = xr.DataArray(
                                np.nan*np.ones((1)), dims=['sample'])
                            prn_da['sample'] = [
                                x for x in range(prn_da.size)]
                            rcll_da = xr.DataArray(
                                np.nan*np.ones((1)), dims=['sample'])
                            rcll_da['sample'] = [
                                x for x in range(rcll_da.size)]
                            tpr_fpr = xr.DataArray(
                                np.nan*np.ones((100)), dims=['FPR'])
                            tpr_fpr['FPR'] = mean_fpr
                            prn_rcll = xr.DataArray(
                                np.nan*np.ones((100)), dims=['RCLL'])
                            prn_rcll['RCLL'] = mean_fpr
                            pr_auc_da = xr.DataArray(np.nan)
                            roc_auc_da = xr.DataArray(np.nan)
                            no_skill_da = xr.DataArray(np.nan)
                        else:
                            no_skill = len(
                                y_true[y_true == 1]) / len(y_true)
                            no_skill_da = xr.DataArray(no_skill)
                            fpr, tpr, _ = roc_curve(y_true, y_prob)
                            interp_tpr = np.interp(mean_fpr, fpr, tpr)
                            interp_tpr[0] = 0.0
                            roc_auc = roc_auc_score(y_true, y_prob)
                            prn, rcll, _ = precision_recall_curve(
                                y_true, y_prob)
                            interp_prn = np.interp(
                                mean_fpr, rcll[::-1], prn[::-1])
                            interp_prn[0] = 1.0
                            pr_auc_score = auc(rcll, prn)
                            roc_auc_da = xr.DataArray(roc_auc)
                            pr_auc_da = xr.DataArray(pr_auc_score)
                            prn_da = xr.DataArray(prn, dims=['sample'])
                            prn_da['sample'] = [x for x in range(len(prn))]
                            rcll_da = xr.DataArray(rcll, dims=['sample'])
                            rcll_da['sample'] = [
                                x for x in range(len(rcll))]
                            fpr_da = xr.DataArray(fpr, dims=['sample'])
                            fpr_da['sample'] = [x for x in range(len(fpr))]
                            tpr_da = xr.DataArray(tpr, dims=['sample'])
                            tpr_da['sample'] = [x for x in range(len(tpr))]
                            tpr_fpr = xr.DataArray(
                                interp_tpr, dims=['FPR'])
                            tpr_fpr['FPR'] = mean_fpr
                            prn_rcll = xr.DataArray(
                                interp_prn, dims=['RCLL'])
                            prn_rcll['RCLL'] = mean_fpr
                        ds['fpr'] = fpr_da
                        ds['tpr'] = tpr_da
                        ds['roc-auc'] = roc_auc_da
                        ds['pr-auc'] = pr_auc_da
                        ds['prn'] = prn_da
                        ds['rcll'] = rcll_da
                        ds['TPR'] = tpr_fpr
                        ds['PRN'] = prn_rcll
                        ds['no_skill'] = no_skill_da
                        ds_list.append(ds)
    ds = xr.concat(ds_list, 'dim_0')
    ds['dim_0'] = ind
    ds = ds.unstack()
    ds.attrs = dss.attrs
    ds['fpr'].attrs['long_name'] = 'False positive rate'
    ds['tpr'].attrs['long_name'] = 'True positive rate'
    ds['prn'].attrs['long_name'] = 'Precision'
    ds['rcll'].attrs['long_name'] = 'Recall'
    ds['roc-auc'].attrs['long_name'] = 'ROC or FPR-TPR Area under curve'
    ds['pr-auc'].attrs['long_name'] = 'Precition-Recall Area under curve'
    ds['PRN'].attrs['long_name'] = 'Precision-Recall'
    ds['TPR'].attrs['long_name'] = 'TPR-FPR (ROC)'
    dss = xr.merge([dss, ds], combine_attrs='no_conflicts')
    return dss

#
# def load_ML_models(path=hydro_ml_path, station='drag', prefix='CVM', suffix='.pkl'):
#    from aux_gps import path_glob
#    import joblib
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    import xarray as xr
#    import pandas as pd
#    model_files = path_glob(path, '{}_*{}'.format(prefix, suffix))
#    model_files = sorted(model_files)
#    model_files = [x for x in model_files if station in x.as_posix()]
#    m_list = [joblib.load(x) for x in model_files]
#    model_files = [x.as_posix().split('/')[-1].split('.')[0] for x in model_files]
#    # fix roc-auc:
#    model_files = [x.replace('roc_auc', 'roc-auc') for x in model_files]
#    print('loading {} station only.'.format(station))
#    model_names = [x.split('_')[3] for x in model_files]
##    model_pw_stations = [x.split('_')[1] for x in model_files]
##    model_hydro_stations = [x.split('_')[2] for x in model_files]
#    model_nsplits = [x.split('_')[6] for x in model_files]
#    model_scores = [x.split('_')[5] for x in model_files]
#    model_features = [x.split('_')[4] for x in model_files]
#    model_test_sizes = []
#    for file in model_files:
#        try:
#            model_test_sizes.append(int(file.split('_')[7]))
#        except IndexError:
#            model_test_sizes.append(20)
##    model_pwv_hs_id = list(zip(model_pw_stations, model_hydro_stations))
##    model_pwv_hs_id = ['_'.join(x) for     filename = 'CVR_{}_{}_{}_{}_{}.nc'.format(
#         name, features, refitted_scorer, ikfolds, okfolds)
# x in model_pwv_hs_id]
#    # transform model_dict to dataarray:
#    tups = [tuple(x) for x in zip(model_names, model_scores, model_nsplits, model_features, model_test_sizes)] #, model_pwv_hs_id)]
#    ind = pd.MultiIndex.from_tuples((tups), names=['model', 'scoring', 'splits', 'feature', 'test_size']) #, 'station'])
#    da = xr.DataArray(m_list, dims='dim_0')
#    da['dim_0'] = ind
#    da = da.unstack('dim_0')
#    da['splits'] = da['splits'].astype(int)
#    da['test_size'].attrs['units'] = '%'
#    return da


def plot_heatmaps_for_all_models_and_scorings(dss, var='roc-auc'):  # , save=True):
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    # assert station == dss.attrs['pwv_id']
    cmaps = {'roc-auc': sns.color_palette("Blues", as_cmap=True),
             'pr-auc': sns.color_palette("Greens", as_cmap=True)}
    fg = xr.plot.FacetGrid(
        dss,
        col='model',
        row='scoring',
        sharex=True,
        sharey=True, figsize=(10, 20))
    dss = dss.mean('inner_kfold', keep_attrs=True)
    vmin, vmax = dss[var].min(), 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(fg.axes.shape[0]):  # i is rows
        for j in range(fg.axes.shape[1]):  # j is cols
            ax = fg.axes[i, j]
            modelname = dss['model'].isel(model=j).item()
            scoring = dss['scoring'].isel(scoring=i).item()
            model = dss[var].isel(
                {'model': j, 'scoring': i}).reset_coords(drop=True)
            df = model.to_dataframe()
            title = '{} model ({})'.format(modelname, scoring)
            df = df.unstack()
            mean = df.mean()
            mean.name = 'mean'
            df = df.append(mean).T.droplevel(0)
            ax = sns.heatmap(df, annot=True, cmap=cmaps[var], cbar=False,
                             ax=ax, norm=norm)
            ax.set_title(title)
            ax.vlines([4], 0, 10, color='r', linewidth=2)
            if j > 0:
                ax.set_ylabel('')
            if i < 2:
                ax.set_xlabel('')
    cax = fg.fig.add_axes([0.1, 0.025, .8, .015])
    fg.fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    fg.fig.suptitle('{}'.format(
        dss.attrs[var].upper()), fontweight='bold')
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.937,
                           bottom=0.099,
                           left=0.169,
                           right=0.993,
                           hspace=0.173,
                           wspace=0.051)
    # if save:
    #     filename = 'hydro_models_heatmaps_on_{}_{}_{}.png'.format(
    #         station, dss['outer_kfold'].size, var)
    #     plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_feature_importances_from_dss(
        dss,
        feat_dim='features', outer_dim='outer_split',
        features='pwv+pressure+doy', fix_xticklabels=True,
        axes=None, save=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set_palette('Dark2', 6)
    sns.set_style('whitegrid')
    sns.set_style('ticks')
    # use dss.sel(model='RF') first as input
    dss['feature'] = dss['feature'].str.replace('DOY', 'doy')
    dss = dss.sel({feat_dim: features})
    # tests_ds = dss['test_score']
    # tests_ds = tests_ds.sel(scorer=scorer)
    # max_score_split = int(tests_ds.idxmax(outer_dim).item())
    # use mean outer split:
    dss = dss.mean(outer_dim)
    feats = features.split('+')
    fn = len(feats)
    if fn == 1:
        gr_spec = None
        fix_xticklabels = False
    elif fn == 2:
        gr_spec = [1, 1]
    elif fn == 3:
        gr_spec = [2, 5, 5]
    if axes is None:
        fig, axes = plt.subplots(1, fn, sharey=True, figsize=(17, 5), gridspec_kw={'width_ratios': gr_spec})
        try:
            axes.flatten()
        except AttributeError:
            axes = [axes]
    for i, f in enumerate(sorted(feats)):
        fe = [x for x in dss['feature'].values if f in x]
        print(fe)
        # dsf = dss['feature_importances'].sel(
        #     feature=fe).sel({outer_dim: max_score_split}).reset_coords(
        #     drop=True)
        dsf = dss['feature_importances'].sel(
            feature=fe).reset_coords(
            drop=True)
        dsf = dsf.to_dataset('scorer').to_dataframe(
        ).reset_index(drop=True) * 100
        title = '{}'.format(f.upper())
        dsf.plot.bar(ax=axes[i], title=title, rot=0, legend=False, zorder=20,
                     width=.8)
        dsf_sum = dsf.sum().tolist()
        handles, labels = axes[i].get_legend_handles_labels()
        labels = [
            '{} ({:.1f} %)'.format(
                x, y) for x, y in zip(
                labels, dsf_sum)]
        axes[i].legend(handles=handles, labels=labels, prop={'size': 10}, loc='upper left')
        axes[i].set_ylabel('Feature importance [%]')
        axes[i].grid(axis='y', zorder=1)
    if fix_xticklabels:
        n = sum(['pwv' in x for x in dss.feature.values])
        axes[0].xaxis.set_ticklabels('')
        hrs = np.arange(-24, -24+n)
        axes[1].set_xticklabels(hrs, rotation=30, ha="center", fontsize=12)
        axes[2].set_xticklabels(hrs, rotation=30, ha="center", fontsize=12)
        axes[1].set_xlabel('Hours prior to flood')
        axes[2].set_xlabel('Hours prior to flood')
        fig.tight_layout()
    if save:
        filename = 'RF_feature_importances_all_scorers_{}.png'.format(features)
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_feature_importances(
        dss,
        feat_dim='features',
        features='pwv+pressure+doy',
        scoring='f1', fix_xticklabels=True,
        axes=None, save=True):
    # use dss.sel(model='RF') first as input
    import matplotlib.pyplot as plt
    import numpy as np
    dss = dss.sel({feat_dim: features})
    tests_ds = dss[[x for x in dss if 'test' in x]]
    tests_ds = tests_ds.sel(scoring=scoring)
    score_ds = tests_ds['test_{}'.format(scoring)]
    max_score = score_ds.idxmax('outer_kfold').values
    feats = features.split('+')
    fn = len(feats)
    if axes is None:
        fig, axes = plt.subplots(1, fn, sharey=True, figsize=(17, 5), gridspec_kw={'width_ratios': [1, 4, 4]})
        try:
            axes.flatten()
        except AttributeError:
            axes = [axes]
    for i, f in enumerate(feats):
        fe = [x for x in dss['feature'].values if f in x]
        dsf = dss['feature_importances'].sel(
            feature=fe,
            outer_kfold=max_score).reset_coords(
            drop=True)
        dsf = dsf.to_dataset('scoring').to_dataframe(
        ).reset_index(drop=True) * 100
        title = '{} ({})'.format(f.upper(), scoring)
        dsf.plot.bar(ax=axes[i], title=title, rot=0, legend=False, zorder=20,
                     width=.8)
        dsf_sum = dsf.sum().tolist()
        handles, labels = axes[i].get_legend_handles_labels()
        labels = [
            '{} ({:.1f} %)'.format(
                x, y) for x, y in zip(
                labels, dsf_sum)]
        axes[i].legend(handles=handles, labels=labels, prop={'size': 8})
        axes[i].set_ylabel('Feature importance [%]')
        axes[i].grid(axis='y', zorder=1)
    if fix_xticklabels:
        axes[0].xaxis.set_ticklabels('')
        hrs = np.arange(-24,0)
        axes[1].set_xticklabels(hrs, rotation = 30, ha="center", fontsize=12)
        axes[2].set_xticklabels(hrs, rotation = 30, ha="center", fontsize=12)
        axes[1].set_xlabel('Hours prior to flood')
        axes[2].set_xlabel('Hours prior to flood')
    if save:
        fig.tight_layout()
        filename = 'RF_feature_importances_{}.png'.format(scoring)
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return


def plot_feature_importances_for_all_scorings(dss,
                                              features='doy+pwv+pressure',
                                              model='RF', splitfigs=True):
    import matplotlib.pyplot as plt
    # station = dss.attrs['pwv_id'].upper()
    dss = dss.sel(model=model).reset_coords(drop=True)
    fns = len(features.split('+'))
    scores = dss['scoring'].values
    scores1 = ['f1', 'precision', 'recall']
    scores2 = ['hss', 'tss', 'accuracy','roc-auc']
    if splitfigs:
        fig, axes = plt.subplots(len(scores1), fns, sharey=True, figsize=(15, 20))
        for i, score in enumerate(scores1):
            plot_feature_importances(
                dss, features=features, scoring=score, axes=axes[i, :])
        fig.suptitle(
            'feature importances of {} model'.format(model))
        fig.tight_layout()
        fig.subplots_adjust(top=0.935,
                            bottom=0.034,
                            left=0.039,
                            right=0.989,
                            hspace=0.19,
                            wspace=0.027)
        filename = 'RF_feature_importances_1.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
        fig, axes = plt.subplots(len(scores2), fns, sharey=True, figsize=(15, 20))
        for i, score in enumerate(scores2):
            plot_feature_importances(
                dss, features=features, scoring=score, axes=axes[i, :])
        fig.suptitle(
            'feature importances of {} model'.format(model))
        fig.tight_layout()
        fig.subplots_adjust(top=0.935,
                            bottom=0.034,
                            left=0.039,
                            right=0.989,
                            hspace=0.19,
                            wspace=0.027)
        filename = 'RF_feature_importances_2.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    else:
        fig, axes = plt.subplots(len(scores), fns, sharey=True, figsize=(15, 20))
        for i, score in enumerate(scores):
            plot_feature_importances(
                dss, features=features, scoring=score, axes=axes[i, :])
        fig.suptitle(
            'feature importances of {} model'.format(model))
        fig.tight_layout()
        fig.subplots_adjust(top=0.935,
                            bottom=0.034,
                            left=0.039,
                            right=0.989,
                            hspace=0.19,
                            wspace=0.027)
        filename = 'RF_feature_importances.png'
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return dss


def plot_ROC_curve_from_dss_nested_CV(dss, outer_dim='outer_split',
                                      plot_chance=True, color='tab:blue',
                                      fontsize=14, plot_legend=True,
                                      title=None,
                                      ax=None, main_label=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = "Receiver operating characteristic"
    mean_fpr = dss['FPR'].values
    mean_tpr = dss['TPR'].mean(outer_dim).values
    mean_auc = dss['roc_auc_score'].mean().item()
    if np.isnan(mean_auc):
        return ValueError
    std_auc = dss['roc_auc_score'].std().item()
    field = 'TPR'
    xlabel = 'False Positive Rate'
    ylabel = 'True Positive Rate'
    if main_label is None:
        main_label = r'Mean ROC (AUC={:.2f}$\pm${:.2f})'.format(mean_auc, std_auc)
    textstr = '\n'.join(['{}'.format(
            main_label), r'(AUC={:.2f}$\pm${:.2f})'.format(mean_auc, std_auc)])
    main_label = textstr
    ax.plot(mean_fpr, mean_tpr, color=color,
            lw=3, alpha=.8, label=main_label)
    std_tpr = dss[field].std(outer_dim).values
    n = dss[outer_dim].size
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plot Chance line:
    if plot_chance:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8, zorder=206)
    stdlabel = r'$\pm$ 1 Std. dev.'
    stdstr = '\n'.join(['{}'.format(stdlabel), r'({} outer splits)'.format(n)])
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color='grey',
        alpha=.2, label=stdstr)
    ax.grid()
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    # ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    return ax


def plot_ROC_PR_curve_from_dss(
        dss,
        outer_dim='outer_kfold',
        inner_dim='inner_kfold',
        plot_chance=True,
        ax=None,
        color='b',
        title=None,
        std_on='inner',
        main_label=None,
        fontsize=14,
        plot_type='ROC',
        plot_std_legend=True):
    """plot classifier metrics, plot_type=ROC or PR"""
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = "Receiver operating characteristic"
    if plot_type == 'ROC':
        mean_fpr = dss['FPR'].values
        mean_tpr = dss['TPR'].mean(outer_dim).mean(inner_dim).values
        mean_auc = dss['roc-auc'].mean().item()
        if np.isnan(mean_auc):
            return ValueError
        std_auc = dss['roc-auc'].std().item()
        field = 'TPR'
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
    elif plot_type == 'PR':
        mean_fpr = dss['RCLL'].values
        mean_tpr = dss['PRN'].mean(outer_dim).mean(inner_dim).values
        mean_auc = dss['pr-auc'].mean().item()
        if np.isnan(mean_auc):
            return ValueError
        std_auc = dss['pr-auc'].std().item()
        no_skill = dss['no_skill'].mean(outer_dim).mean(inner_dim).item()
        field = 'PRN'
        xlabel = 'Recall'
        ylabel = 'Precision'
    # plot mean ROC:
    if main_label is None:
        main_label = r'Mean {} (AUC={:.2f}$\pm${:.2f})'.format(
            plot_type, mean_auc, std_auc)
    else:
        textstr = '\n'.join(['Mean ROC {}'.format(
            main_label), r'(AUC={:.2f}$\pm${:.2f})'.format(mean_auc, std_auc)])
        main_label = textstr
    ax.plot(mean_fpr, mean_tpr, color=color,
            lw=2, alpha=.8, label=main_label)
    if std_on == 'inner':
        std_tpr = dss[field].mean(outer_dim).std(inner_dim).values
        n = dss[inner_dim].size
    elif std_on == 'outer':
        std_tpr = dss[field].mean(inner_dim).std(outer_dim).values
        n = dss[outer_dim].size
    elif std_on == 'all':
        std_tpr = dss[field].stack(
            dumm=[inner_dim, outer_dim]).std('dumm').values
        n = dss[outer_dim].size * dss[inner_dim].size
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plot Chance line:
    if plot_chance:
        if plot_type == 'ROC':
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)
        elif plot_type == 'PR':
            ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='r',
                    lw=2, label='No Skill', alpha=.8)
    # plot ROC STD range:
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color='grey',
        alpha=.2, label=r'$\pm$ 1 std. dev. ({} {} splits)'.format(n, std_on))
    ax.grid()
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # handles, labels = ax.get_legend_handles_labels()
    # if not plot_std_legend:
    #     if len(handles) == 7:
    #         handles = handles[:-2]
    #         labels = labels[:-2]
    #     else:
    #         handles = handles[:-1]
    #         labels = labels[:-1]
    # ax.legend(handles=handles, labels=labels, loc="lower right",
    #           fontsize=fontsize)
    return ax


def plot_many_ROC_curves(model, X, y, name='', color='b', ax=None,
                         plot_chance=True, title=None, n_splits=None):
    from sklearn.metrics import plot_roc_curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = "Receiver operating characteristic"
    # just plot the ROC curve for X, y, no nsplits and stats:
    if n_splits is None:
        viz = plot_roc_curve(model, X, y, color=color, ax=ax, name=name)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (train, val) in enumerate(cv.split(X, y)):
            model.fit(X[train], y[train])
#            y_score = model.fit(X[train], y[train]).predict_proba(X[val])[:, 1]
            y_pred = model.predict(X[val])
            fpr, tpr, _ = roc_curve(y[val], y_pred)
#            viz = plot_roc_curve(model, X[val], y[val],
#                             name='ROC fold {}'.format(i),
#                             alpha=0.3, lw=1, ax=ax)
#            fpr = viz.fpr
#            tpr = viz.tpr
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y[val], y_pred))
#            scores.append(f1_score(y[val], y_pred))
#        scores = np.array(scores)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=color,
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (
                    mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
    if plot_chance:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=title)
    ax.legend(loc="lower right")
    return ax


def HP_tuning(X, y, model_name='SVC', val_size=0.18, n_splits=None,
              test_size=None,
              best_score='f1', seed=42, savepath=None):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    """ do HP tuning with ML_Classfier_Switcher object and return a DataSet of
    results. note that the X, y are already after split to val/test"""
    # first get the features from X:
    features = list(set(['_'.join(x.split('_')[0:2])
                         for x in X['feature'].values]))
    ml = ML_Classifier_Switcher()
    sk_model = ml.pick_model(model_name)
    param_grid = ml.param_grid
    if n_splits is None and val_size is not None:
        n_splits = int((1 // val_size) - 1)
    elif val_size is not None and n_splits is not None:
        raise('Both val_size and n_splits are defined, choose either...')
    print('StratifiedKfolds of {}.'.format(n_splits))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    gr = GridSearchCV(estimator=sk_model, param_grid=param_grid, cv=cv,
                      n_jobs=-1, scoring=['f1', 'roc_auc', 'accuracy'], verbose=1,
                      refit=best_score, return_train_score=True)
    gr.fit(X, y)
    if best_score is not None:
        ds, best_model = process_gridsearch_results(gr, model_name,
                                                    features=features, pwv_id=X.attrs['pwv_id'], hs_id=y.attrs['hydro_station_id'], test_size=test_size)
    else:
        ds = process_gridsearch_results(gr, model_name, features=features,
                                        pwv_id=X.attrs['pwv_id'], hs_id=y.attrs['hydro_station_id'], test_size=test_size)
        best_model = None
    if savepath is not None:
        save_cv_results(ds, best_model=best_model, savepath=savepath)
    return ds, best_model


def save_gridsearchcv_object(GridSearchCV, savepath, filename):
    import joblib
    print('{} was saved to {}'.format(filename, savepath))
    joblib.dump(GridSearchCV, savepath / filename)
    return


def run_RF_feature_importance_on_all_features(path=hydro_path, gr_path=hydro_ml_path/'holdout'):
    import xarray as xr
    from aux_gps import get_all_possible_combinations_from_list
    feats = get_all_possible_combinations_from_list(
        ['pwv', 'pressure', 'doy'], reduce_single_list=True, combine_by_sep='+')
    feat_list = []
    for feat in feats:
        da = holdout_test(model_name='RF', return_RF_FI=True, features=feat)
        feat_list.append(da)
    daa = xr.concat(feat_list, 'features')
    daa['features'] = feats
    return daa


def load_nested_CV_test_results_from_all_models(path=hydro_path, load_hyper=False):
    from aux_gps import path_glob
    import xarray as xr
    files = path_glob(path, 'nested_CV_test_results_*_all_features_with_hyper.nc')
    models = [x.as_posix().split('/')[-1].split('_')[4] for x in files]
    if not load_hyper:
        print('loading CV test results only for {} models'.format(', '.join(models)))
        dsl = [xr.load_dataset(x) for x in files]
        dsl = [x[['mean_score', 'std_score', 'test_score', 'roc_auc_score', 'TPR']] for x in dsl]
        dss = xr.concat(dsl, 'model')
        dss['model'] = models
    return dss


def run_CV_nested_tests_on_all_features(path=hydro_path, gr_path=hydro_ml_path/'nested4',
                                        verbose=False, model_name='SVC',
                                        savepath=None, drop_hours=None, PI=30, Ptest=None):
    """returns the nested CV test results for all scorers, features and models,
    if model is chosen, i.e., model='MLP', returns just this model results
    and its hyper-parameters per each outer split"""
    import xarray as xr
    from aux_gps import get_all_possible_combinations_from_list
    from aux_gps import save_ncfile
    feats = get_all_possible_combinations_from_list(
        ['pwv', 'pressure', 'doy'], reduce_single_list=True, combine_by_sep='+')
    feat_list = []
    for feat in feats:
        print('Running CV on feature {}'.format(feat))
        ds = CV_test_after_GridSearchCV(path=path, gr_path=gr_path,
                                        model_name=model_name,
                                        features=feat, PI=PI, Ptest=Ptest,
                                        verbose=verbose, drop_hours=drop_hours)
        feat_list.append(ds)
    dsf = xr.concat(feat_list, 'features')
    dsf['features'] = feats
    dss = dsf
    dss.attrs['model'] = model_name
    if Ptest is not None:
        filename = 'nested_CV_test_results_{}_all_features_with_hyper_permutation_tests.nc'.format(model_name)
    else:
        filename = 'nested_CV_test_results_{}_all_features_with_hyper.nc'.format(model_name)
    if savepath is not None:
        save_ncfile(dss, savepath, filename)
    return dss


def run_holdout_test_on_all_models_and_features(path=hydro_path, gr_path=hydro_ml_path/'holdout'):
    import xarray as xr
    from aux_gps import get_all_possible_combinations_from_list
    feats = get_all_possible_combinations_from_list(
        ['pwv', 'pressure', 'doy'], reduce_single_list=True, combine_by_sep='+')
    models = ['MLP', 'SVC', 'RF']
    model_list = []
    model_list2 = []
    for model in models:
        feat_list = []
        feat_list2 = []
        for feat in feats:
            best, roc = holdout_test(path=path, gr_path=gr_path,
                                     model_name=model, features=feat)
            best.index.name = 'scorer'
            ds = best[['mean_score', 'std_score', 'holdout_test_scores']].to_xarray()
            roc.index.name = 'FPR'
            roc_da = roc.to_xarray().to_array('scorer')
            feat_list.append(ds)
            feat_list2.append(roc_da)
        dsf = xr.concat(feat_list, 'features')
        dsf2 = xr.concat(feat_list2, 'features')
        dsf['features'] = feats
        dsf2['features'] = feats
        model_list.append(dsf)
        model_list2.append(dsf2)
    dss = xr.concat(model_list, 'model')
    rocs = xr.concat(model_list2, 'model')
    dss['model'] = models
    rocs['model'] = models
    dss['roc'] = rocs
    return dss


def prepare_X_y_for_holdout_test(features='pwv+doy', model_name='SVC',
                                 path=hydro_path, drop_hours=None):
    # combine X,y and split them according to test ratio and seed:
    X, y = combine_pos_neg_from_nc_file(path)
    # re arange X features according to model:
    feats = features.split('+')
    if model_name == 'RF' and 'doy' in feats:
        if isinstance(feats, list):
            feats.remove('doy')
            feats.append('DOY')
        elif isinstance(feats, str):
            feats = 'DOY'
    elif model_name != 'RF' and 'doy' in feats:
        if isinstance(feats, list):
            feats.remove('doy')
            feats.append('doy_sin')
            feats.append('doy_cos')
        elif isinstance(feats, str):
            feats = ['doy_sin']
            feats.append('doy_cos')
    X = select_features_from_X(X, feats)
    if drop_hours is not None:
        X = drop_hours_in_pwv_pressure_features(X, drop_hours, verbose=True)
    return X, y


def CV_test_after_GridSearchCV(path=hydro_path, gr_path=hydro_ml_path/'nested4',
                               model_name='SVC', features='pwv',
                               verbose=False, drop_hours=None, PI=None, Ptest=None):
    """do cross_validate with all scorers on all gridsearchcv folds,
    reads the nested outer splits CV file in gr_path"""
    import xarray as xr
    import numpy as np
    cv = read_cv_params_and_instantiate(gr_path/'CV_outer.csv')
    if verbose:
        print(cv)
    param_df_dict = load_one_gridsearchcv_object(path=gr_path,
                                                 cv_type='nested',
                                                 features=features,
                                                 model_name=model_name,
                                                 verbose=verbose)
    X, y = prepare_X_y_for_holdout_test(features, model_name, path,
                                        drop_hours=drop_hours)
    if Ptest is not None:
        ds = run_permutation_classifier_test(X, y, cv, param_df_dict, Ptest=Ptest,
                                             model_name=model_name, verbose=verbose)
        return ds
    outer_bests = []
    outer_rocs = []
    fis = []
    pi_means = []
    pi_stds = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        outer_split = '{}-{}'.format(i+1, cv.n_splits)
        best_params_df = param_df_dict.get(outer_split)
        if model_name == 'RF':
            bdf, roc, fi, pi_mean, pi_std = run_test_on_CV_split(X_train, y_train, X_test, y_test,
                                                                 best_params_df, PI=PI, Ptest=Ptest,
                                                                 model_name=model_name, verbose=verbose)
            fis.append(fi)
        else:
            bdf, roc, pi_mean, pi_std = run_test_on_CV_split(X_train, y_train, X_test, y_test,
                                                             best_params_df, PI=PI,
                                                             model_name=model_name, verbose=verbose)
        pi_means.append(pi_mean)
        pi_stds.append(pi_std)
        bdf.index.name = 'scorer'
        roc.index.name = 'FPR'
        if 'hidden_layer_sizes' in bdf.columns:
            bdf['hidden_layer_sizes'] = bdf['hidden_layer_sizes'].astype(str)
        bdf_da = bdf.to_xarray()
        roc_da = roc.to_xarray().to_array('scorer')
        roc_da.name = 'TPR'
        outer_bests.append(bdf_da)
        outer_rocs.append(roc_da)
    best_da = xr.concat(outer_bests, 'outer_split')
    roc_da = xr.concat(outer_rocs, 'outer_split')
    best = xr.merge([best_da, roc_da])
    best['outer_split'] = np.arange(1, cv.n_splits + 1)
    if model_name == 'RF':
        fi_da = xr.concat(fis, 'outer_split')
        best['feature_importances'] = fi_da
    pi_mean_da = xr.concat(pi_means, 'outer_split')
    pi_std_da = xr.concat(pi_stds, 'outer_split')
    best['PI_mean'] = pi_mean_da
    best['PI_std'] = pi_std_da
    return best


def run_permutation_classifier_test(X, y, cv, best_params_df, Ptest=100,
                                    model_name='SVC', verbose=False):
    from sklearn.model_selection import permutation_test_score
    import xarray as xr
    import numpy as np
    ml = ML_Classifier_Switcher()
    if verbose:
        print('Picking {} model with best params'.format(model_name))
    splits = []
    for i, df in enumerate(best_params_df.values()):
        if verbose:
            print('running on split #{}'.format(i+1))
        true_scores = []
        pvals = []
        perm_scores = []
        for scorer in df.index:
            sk_model = ml.pick_model(model_name)
            # get best params (drop two last cols since they are not params):
            params = df.T[scorer][:-2].to_dict()
            if verbose:
                print('{} scorer, params:{}'.format(scorer, params))
            true, perm_scrs, pval = permutation_test_score(sk_model, X, y,
                                                           cv=cv,
                                                           n_permutations=Ptest,
                                                           scoring=scorers(scorer),
                                                           random_state=0,
                                                           n_jobs=-1)
            true_scores.append(true)
            pvals.append(pval)
            perm_scores.append(perm_scrs)
        true_da = xr.DataArray(true_scores, dims=['scorer'])
        true_da['scorer'] = [x for x in df.index.values]
        true_da.name = 'true_score'
        pval_da = xr.DataArray(pvals, dims=['scorer'])
        pval_da['scorer'] = [x for x in df.index.values]
        pval_da.name = 'pvalue'
        perm_da = xr.DataArray(perm_scores, dims=['scorer', 'permutations'])
        perm_da['scorer'] = [x for x in df.index.values]
        perm_da['permutations'] = np.arange(1, Ptest+1)
        perm_da.name = 'permutation_score'
        ds = xr.merge([true_da, pval_da, perm_da])
        splits.append(ds)
    dss = xr.concat(splits, dim='outer_split')
    dss['outer_split'] = np.arange(1, len(best_params_df)+ 1)
    return dss


def run_test_on_CV_split(X_train, y_train, X_test, y_test, param_df,
                         model_name='SVC', verbose=False, PI=None,
                         Ptest=None):
    import numpy as np
    import xarray as xr
    import pandas as pd
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance
    best_df = param_df.copy()
    ml = ML_Classifier_Switcher()
    if verbose:
        print('Picking {} model with best params'.format(model_name))
    # print('Features are: {}'.format(features))
    test_scores = []
    fi_list = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    roc_aucs = []
    pi_mean_list = []
    pi_std_list = []
    for scorer in best_df.index:
        sk_model = ml.pick_model(model_name)
        # get best params (drop two last cols since they are not params):
        params = best_df.T[scorer][:-2].to_dict()
        if verbose:
            print('{} scorer, params:{}'.format(scorer, params))
        sk_model.set_params(**params)
        sk_model.fit(X_train, y_train)
        if hasattr(sk_model, 'feature_importances_'):
            FI = xr.DataArray(sk_model.feature_importances_, dims=['feature'])
            FI['feature'] = X_train['feature']
            fi_list.append(FI)
        y_pred = sk_model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_auc = roc_auc_score(y_test, y_pred)
        roc_aucs.append(roc_auc)
        tprs.append(interp_tpr)
        score = scorer_function(scorer, y_test, y_pred)
        test_scores.append(score)
        pi = permutation_importance(sk_model, X_test, y_test,
                                    n_repeats=PI,
                                    scoring=scorers(scorer),
                                    random_state=0, n_jobs=-1)
        pi_mean = xr.DataArray(pi['importances_mean'], dims='feature')
        pi_std = xr.DataArray(pi['importances_std'], dims='feature')
        pi_mean.name = 'PI_mean'
        pi_std.name = 'PI_std'
        pi_mean['feature'] = X_train['feature']
        pi_std['feature'] = X_train['feature']
        pi_mean_list.append(pi_mean)
        pi_std_list.append(pi_std)
    pi_mean_da = xr.concat(pi_mean_list, 'scorer')
    pi_std_da = xr.concat(pi_std_list, 'scorer')
    pi_mean_da['scorer'] = [x for x in best_df.index.values]
    pi_std_da['scorer'] = [x for x in best_df.index.values]
    roc_df = pd.DataFrame(tprs).T
    roc_df.columns = [x for x in best_df.index]
    roc_df.index = mean_fpr
    best_df['test_score'] = test_scores
    best_df['roc_auc_score'] = roc_aucs
    if hasattr(sk_model, 'feature_importances_'):
        fi = xr.concat(fi_list, 'scorer')
        fi['scorer'] = [x for x in best_df.index.values]
        return best_df, roc_df, fi, pi_mean_da, pi_std_da
    elif PI is not None:
        return best_df, roc_df, pi_mean_da, pi_std_da
    else:
        return best_df, roc_df


def holdout_test(path=hydro_path, gr_path=hydro_ml_path/'holdout',
                 model_name='SVC', features='pwv', return_RF_FI=False,
                 verbose=False):
    """do a holdout test with best model from gridsearchcv
    with all scorers"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import xarray as xr
    import pandas as pd
    import numpy as np
    # process gridsearchcv results:
    best_df, test_ratio, seed = load_one_gridsearchcv_object(path=gr_path,
                                                             cv_type='holdout',
                                                             features=features,
                                                             model_name=model_name,
                                                             verbose=False)
    print('Using random seed of {} and {}% test ratio'.format(seed, test_ratio))
    ts = int(test_ratio) / 100
    X, y = prepare_X_y_for_holdout_test(features, model_name, path)
    # split using test_size and seed:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=ts,
                                                        random_state=int(seed),
                                                        stratify=y)
    if verbose:
        print('y train pos/neg:{}, {}'.format((y_train==1).sum().item(),(y_train==0).sum().item()))
        print('y test pos/neg:{}, {}'.format((y_test==1).sum().item(),(y_test==0).sum().item()))
    # pick model and set the params to best from gridsearchcv:
    ml = ML_Classifier_Switcher()
    print('Picking {} model with best params'.format(model_name))
    print('Features are: {}'.format(features))
    test_scores = []
    fi_list = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    roc_aucs = []
    for scorer in best_df.index:
        sk_model = ml.pick_model(model_name)
        # get best params (drop two last cols since they are not params):
        params = best_df.T[scorer][:-2].to_dict()
        if verbose:
            print('{} scorer, params:{}'.format(scorer, params))
        sk_model.set_params(**params)
        sk_model.fit(X_train, y_train)
        if hasattr(sk_model, 'feature_importances_'):
            FI = xr.DataArray(sk_model.feature_importances_, dims=['feature'])
            FI['feature'] = X_train['feature']
            fi_list.append(FI)
        y_pred = sk_model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_auc = roc_auc_score(y_test, y_pred)
        roc_aucs.append(roc_auc)
        tprs.append(interp_tpr)
        score = scorer_function(scorer, y_test, y_pred)
        test_scores.append(score)
    roc_df = pd.DataFrame(tprs).T
    roc_df.columns = [x for x in best_df.index]
    roc_df.index = mean_fpr
    best_df['holdout_test_scores'] = test_scores
    best_df['roc_auc_score'] = roc_aucs
    if fi_list and return_RF_FI:
        da = xr.concat(fi_list, 'scorer')
        da['scorer'] = best_df.index.values
        da.name = 'RF_feature_importances'
        return da
    return best_df, roc_df


def load_one_gridsearchcv_object(path=hydro_ml_path, cv_type='holdout', features='pwv',
                                 model_name='SVC', verbose=True):
    """load one gridsearchcv obj with model_name and features and run read_one_gridsearchcv_object"""
    from aux_gps import path_glob
    import joblib
    # first filter for model name:
    if verbose:
        print('loading GridsearchCVs results for {} model with {} cv type'.format(model_name, cv_type))
    model_files = path_glob(path, 'GRSRCHCV_{}_*.pkl'.format(cv_type))
    model_files = [x for x in model_files if model_name in x.as_posix()]
    # now select features:
    if verbose:
        print('loading GridsearchCVs results with {} features'.format(features))
    model_features = [x.as_posix().split('/')[-1].split('_')[3] for x in model_files]
    feat_ind = get_feature_set_from_list(model_features, features)
    # also get the test ratio and seed number:
    if len(feat_ind) > 1:
        if verbose:
            print('found {} GR objects.'.format(len(feat_ind)))
        files = sorted([model_files[x] for x in feat_ind])
        outer_splits = [x.as_posix().split('/')[-1].split('.')[0].split('_')[-3] for x in files]
        grs = [joblib.load(x) for x in files]
        best_dfs = [read_one_gridsearchcv_object(x) for x in grs]
        di = dict(zip(outer_splits, best_dfs))
        return di
    else:
        file = model_files[feat_ind]
        seed = file.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
        outer_splits = file.as_posix().split('/')[-1].split('.')[0].split('_')[-3]
    # load and produce best_df:
        gr = joblib.load(file)
        best_df = read_one_gridsearchcv_object(gr)
        return best_df, outer_splits, seed


def get_feature_set_from_list(model_features_list, features, sep='+'):
    """select features from model_features_list,
    return the index in the model_features_list and the entry itself"""
    # first find if features is a single or multiple features:
    if isinstance(features, str) and sep not in features:
        try:
            ind = [i for i, e in enumerate(model_features_list) if e == features]
            # ind = model_features_list.index(features)
        except ValueError:
            raise ValueError('{} is not in {}'.format(features, ', '.join(model_features_list)))
    elif isinstance(features, str) and sep in features:
        features_split = features.split(sep)
        mf = [x.split(sep) for x in model_features_list]
        bool_list = [set(features_split) == (set(x)) for x in mf]
        ind = [i for i, x in enumerate(bool_list) if x]
        # print(len(ind))
        # ind = ind[0]
        # feat = model_features_list[ind]
    # feat = model_features_list[ind]
    return ind


def read_one_gridsearchcv_object(gr):
    """read one gridsearchcv multimetric object and
    get the best params, best mean/std scores"""
    import pandas as pd
    # param grid dict:
    params = gr.param_grid
    # scorer names:
    scoring = [x for x in gr.scoring.keys()]
    # df:
    df = pd.DataFrame().from_dict(gr.cv_results_)
    # produce multiindex from param_grid dict:
    param_names = [x for x in params.keys()]
    # unpack param_grid vals to list of lists:
    pro = [[y for y in x] for x in params.values()]
    ind = pd.MultiIndex.from_product((pro), names=param_names)
    df.index = ind
    best_params = []
    best_mean_scores = []
    best_std_scores = []
    for scorer in scoring:
        best_params.append(df[df['rank_test_{}'.format(scorer)]==1]['mean_test_{}'.format(scorer)].index[0])
        best_mean_scores.append(df[df['rank_test_{}'.format(scorer)]==1]['mean_test_{}'.format(scorer)].iloc[0])
        best_std_scores.append(df[df['rank_test_{}'.format(scorer)]==1]['std_test_{}'.format(scorer)].iloc[0])
    best_df = pd.DataFrame(best_params, index=scoring, columns=param_names)
    best_df['mean_score'] = best_mean_scores
    best_df['std_score'] = best_std_scores
    return best_df


def process_gridsearch_results(GridSearchCV, model_name,
                               split_dim='inner_kfold', features=None,
                               pwv_id=None, hs_id=None, test_size=None):
    import xarray as xr
    import pandas as pd
    import numpy as np
    # finish getting best results from all scorers togather
    """takes GridSreachCV object with cv_results and xarray it into dataarray"""
    params = GridSearchCV.param_grid
    scoring = GridSearchCV.scoring
    results = GridSearchCV.cv_results_
    # for scorer in scoring:
    #     for sample in ['train', 'test']:
    #         sample_score_mean = results['mean_{}_{}'.format(sample, scorer)]
    #         sample_score_std = results['std_{}_{}'.format(sample, scorer)]
    #     best_index = np.nonzero(results['rank_test_{}'.format(scorer)] == 1)[0][0]
    #     best_score = results['mean_test_{}'.format(scorer)][best_index]
    names = [x for x in params.keys()]

    # unpack param_grid vals to list of lists:
    pro = [[y for y in x] for x in params.values()]
    ind = pd.MultiIndex.from_product((pro), names=names)
#        result_names = [x for x in GridSearchCV.cv_results_.keys() if 'split'
#                        not in x and 'time' not in x and 'param' not in x and
#                        'rank' not in x]
    result_names = [
        x for x in results.keys() if 'param' not in x]
    ds = xr.Dataset()
    for da_name in result_names:
        da = xr.DataArray(results[da_name])
        ds[da_name] = da
    ds = ds.assign(dim_0=ind).unstack('dim_0')
    for dim in ds.dims:
        if ds[dim].dtype == 'O':
            try:
                ds[dim] = ds[dim].astype(str)
            except ValueError:
                ds = ds.assign_coords({dim: [str(x) for x in ds[dim].values]})
        if ('True' in ds[dim]) and ('False' in ds[dim]):
            ds[dim] = ds[dim] == 'True'
    # get all splits data and concat them along number of splits:
    all_splits = [x for x in ds.data_vars if 'split' in x]
    train_splits = [x for x in all_splits if 'train' in x]
    test_splits = [x for x in all_splits if 'test' in x]
    # loop over scorers:
    trains = []
    tests = []
    for scorer in scoring:
        train_splits_scorer = [x for x in train_splits if scorer in x]
        trains.append(xr.concat([ds[x]
                                 for x in train_splits_scorer], split_dim))
        test_splits_scorer = [x for x in test_splits if scorer in x]
        tests.append(xr.concat([ds[x] for x in test_splits_scorer], split_dim))
        splits_scorer = np.arange(1, len(train_splits_scorer) + 1)
    train_splits = xr.concat(trains, 'scoring')
    test_splits = xr.concat(tests, 'scoring')
#    splits = [x for x in range(len(train_splits))]
#    train_splits = xr.concat([ds[x] for x in train_splits], 'split')
#    test_splits = xr.concat([ds[x] for x in test_splits], 'split')
    # replace splits data vars with newly dataarrays:
    ds = ds[[x for x in ds.data_vars if x not in all_splits]]
    ds['split_train_score'] = train_splits
    ds['split_test_score'] = test_splits
    ds[split_dim] = splits_scorer
    if isinstance(scoring, list):
        ds['scoring'] = scoring
    elif isinstance(scoring, dict):
        ds['scoring'] = [x for x in scoring.keys()]
    ds.attrs['name'] = 'CV_results'
    ds.attrs['param_names'] = names
    ds.attrs['model_name'] = model_name
    ds.attrs['{}_splits'.format(split_dim)] = ds[split_dim].size
    if GridSearchCV.refit:
        if hasattr(GridSearchCV.best_estimator_, 'feature_importances_'):
            f_import = xr.DataArray(
                GridSearchCV.best_estimator_.feature_importances_,
                dims=['feature'])
            f_import['feature'] = features
            ds['feature_importances'] = f_import
        ds['best_score'] = GridSearchCV.best_score_
#        ds['best_model'] = GridSearchCV.best_estimator_
        ds.attrs['refitted_scorer'] = GridSearchCV.refit
        for name in names:
            if isinstance(GridSearchCV.best_params_[name], tuple):
                GridSearchCV.best_params_[name] = ','.join(
                    map(str, GridSearchCV.best_params_[name]))
            ds['best_{}'.format(name)] = GridSearchCV.best_params_[name]
        return ds, GridSearchCV.best_estimator_
    else:
        return ds, None


def save_cv_results(cvr, savepath=hydro_path):
    from aux_gps import save_ncfile
    features = '+'.join(cvr.attrs['features'])
    # pwv_id = cvr.attrs['pwv_id']
    # hs_id = cvr.attrs['hs_id']
    # neg_pos_ratio = cvr.attrs['neg_pos_ratio']
    ikfolds = cvr.attrs['inner_kfold_splits']
    okfolds = cvr.attrs['outer_kfold_splits']
    name = cvr.attrs['model_name']
    refitted_scorer = cvr.attrs['refitted_scorer'].replace('_', '-')
    # filename = 'CVR_{}_{}_{}_{}_{}_{}_{}_{}.nc'.format(pwv_id, hs_id,
    #                                                    name, features, refitted_scorer, ikfolds, okfolds, neg_pos_ratio)
    filename = 'CVR_{}_{}_{}_{}_{}.nc'.format(
        name, features, refitted_scorer, ikfolds, okfolds)
    save_ncfile(cvr, savepath, filename)
    return


def scikit_fit_predict(X, y, seed=42, with_pressure=True, n_splits=7,
                       plot=True):
    # step1: CV for train/val (80% from 80-20 test). display results with
    # model and scores(AUC, f1), use StratifiedKFold
    # step 2: use validated model with test (20%) and build ROC curve
    # step 3: add features (pressure) but check for correlation
    # check permutations with scikit learn
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.metrics import f1_score
    from sklearn.metrics import plot_roc_curve
    from sklearn.svm import SVC
    from numpy import interp
    from sklearn.metrics import auc
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    if not with_pressure:
        just_pw = [x for x in X.feature.values if 'pressure' not in x]
        X = X.sel(feature=just_pw)
    X_tt, X_test, y_tt, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#    cv = LeaveOneOut()
    classifier = SVC(kernel='rbf', probability=False,
                     random_state=seed)
#    classifier = LinearDiscriminantAnalysis()
    # clf = QuadraticDiscriminantAnalysis()
    scores = []
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, val) in enumerate(cv.split(X_tt, y_tt)):
        #    for i in range(100):
        #        X_train, X_val, y_train, y_val = train_test_split(
        #            X_tt, y_tt, shuffle=True, test_size=0.5, random_state=i)
        #        clf.fit(X_train, y_train)
        classifier.fit(X_tt[train], y_tt[train])
#        viz = plot_roc_curve(clf, X_val, y_val,
#                             name='ROC run {}'.format(i),
#                             alpha=0.3, lw=1, ax=ax)
        viz = plot_roc_curve(classifier, X_tt[val], y_tt[val],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
#        aucs.append(viz.roc_auc)
#        y_pred = clf.predict(X_val)
        y_pred = classifier.predict(X_tt[val])
        aucs.append(roc_auc_score(y_tt[val], y_pred))
        # scores.append(clf.score(X_val, y_val))
        scores.append(f1_score(y_tt[val], y_pred))
    scores = np.array(scores)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    ax.set_title(
        'ROC curve for KFold={}, with pressure anomalies.'.format(n_splits))
    if not with_pressure:
        ax.set_title(
            'ROC curve for KFold={}, without pressure anomalies.'.format(n_splits))
    y_test_predict = classifier.predict(X_test)
    print('final test predict score:')
    print(f1_score(y_test, y_test_predict))
    if plot:
        plt.figure()
        plt.hist(scores, bins=15, edgecolor='k')
    return scores
    # clf.fit(X,y)


def produce_X_y_from_list(pw_stations=['drag', 'dsea', 'elat'],
                          hs_ids=[48125, 48199, 60170],
                          pressure_station='bet-dagan', max_flow=0,
                          window=25, neg_pos_ratio=1, path=work_yuval,
                          ims_path=ims_path, hydro_path=hydro_path,
                          concat_Xy=False):
    if isinstance(hs_ids, int):
        hs_ids = [hs_ids for x in range(len(pw_stations))]
    kwargs = locals()
    [kwargs.pop(x) for x in ['pw_stations', 'hs_ids', 'concat_Xy']]
    Xs = []
    ys = []
    for pw_station, hs_id in list(zip(pw_stations, hs_ids)):
        X, y = produce_X_y(pw_station, hs_id, **kwargs)
        Xs.append(X)
        ys.append(y)
    if concat_Xy:
        print('concatenating pwv stations {}, with hydro_ids {}.'.format(
            pw_stations, hs_ids))
        X, y = concat_X_y(Xs, ys)
        return X, y
    else:
        return Xs, ys


def concat_X_y(Xs, ys):
    import xarray as xr
    import pandas as pd
    X_attrs = [x.attrs for x in Xs]
    X_com_attrs = dict(zip(pd.DataFrame(X_attrs).T.index.values,
                           pd.DataFrame(X_attrs).T.values.tolist()))
    y_attrs = [x.attrs for x in ys]
    y_com_attrs = dict(zip(pd.DataFrame(y_attrs).T.index.values,
                           pd.DataFrame(y_attrs).T.values.tolist()))
    for X in Xs:
        feat = [x.replace('_' + X.attrs['pwv_id'], '')
                for x in X.feature.values]
        X['feature'] = feat
    X = xr.concat(Xs, 'sample')
    X.attrs = X_com_attrs
    y = xr.concat(ys, 'sample')
    y.attrs = y_com_attrs
    return X, y


def produce_X_y(pw_station='drag', hs_id=48125, pressure_station='bet-dagan',
                window=25, seed=42,
                max_flow=0, neg_pos_ratio=1, path=work_yuval,
                ims_path=ims_path, hydro_path=hydro_path):
    import xarray as xr
    from aux_gps import anomalize_xr
    from PW_stations import produce_geo_gnss_solved_stations
    import numpy as np
    # call preprocess_hydro_station
    hdf, y_meta = preprocess_hydro_station(
        hs_id, hydro_path, max_flow=max_flow)
    # load PWV and other features and combine them to fdf:
    pw = xr.open_dataset(path / 'GNSS_PW_anom_hourly_50_hour_dayofyear.nc')
    fdf = pw[pw_station].to_dataframe(name='pwv_{}'.format(pw_station))
    # add Day of year to fdf:
    doy = fdf.index.dayofyear
    # scale doy to cyclic with amp ~1:
    fdf['doy_sin'] = np.sin(doy * np.pi / 183)
    fdf['doy_cos'] = np.cos(doy * np.pi / 183)
    if pressure_station is not None:
        p = xr.load_dataset(
            ims_path /
            'IMS_BD_hourly_ps_1964-2020.nc')[pressure_station]
        p_attrs = p.attrs
        p_attrs = {'pressure_{}'.format(
            key): val for key, val in p_attrs.items()}
        p = p.sel(time=slice('1996', None))
        p = anomalize_xr(p, freq='MS')
        fdf['pressure_{}'.format(pressure_station)] = p.to_dataframe()
    # check the the last date of hdf is bigger than the first date of fdf,
    # i.e., there is at least one overlapping event in the data:
    if hdf.index[-1] < fdf.index[0]:
        raise KeyError('Data not overlapping, hdf for {} stops at {} and fdf starts at {}'.format(
            hs_id, hdf.index[-1], fdf.index[0]))
    # finally, call add_features_and_produce_X_y
    X, y = add_features_and_produce_X_y(hdf, fdf, window_size=window,
                                        seed=seed,
                                        neg_pos_ratio=neg_pos_ratio)
    # add meta data:
    gps = produce_geo_gnss_solved_stations(plot=False)
    pwv_attrs = gps.loc[pw_station, :][['lat', 'lon', 'alt', 'name']].to_dict()
    pwv_attrs = {'pwv_{}'.format(key): val for key, val in pwv_attrs.items()}
    X.attrs = pwv_attrs
    if pressure_station is not None:
        X.attrs.update(p_attrs)
    y.attrs = y_meta
    y.attrs['hydro_station_id'] = hs_id
    y.attrs['neg_pos_ratio'] = neg_pos_ratio
    # calculate distance to hydro station:
    lat1 = X.attrs['pwv_lat']
    lon1 = X.attrs['pwv_lon']
    lat2 = y.attrs['lat']
    lon2 = y.attrs['lon']
    y.attrs['max_flow'] = max_flow
    distance = calculate_distance_between_two_latlons_israel(
        lat1, lon1, lat2, lon2)
    X.attrs['distance_to_hydro_station_in_km'] = distance / 1000.0
    y.attrs['distance_to_pwv_station_in_km'] = distance / 1000.0
    X.attrs['pwv_id'] = pw_station
    return X, y

# def produce_X_y(station='drag', hs_id=48125, lag=25, anoms=True,
#                neg_pos_ratio=2, add_pressure=False,
#                path=work_yuval, hydro_path=hydro_path, with_ends=False,
#                seed=42,
#                verbose=True, return_xarray=False, pressure_anoms=None):
#    import pandas as pd
#    import numpy as np
#    import xarray as xr
#
#    def produce_da_from_list(event_list, feature='pwv'):
#        X_da = xr.DataArray(event_list, dims=['sample', 'feature'])
#        X_da['feature'] = ['{}_{}'.format(feature, x) for x in np.arange(0, 24, 1)]
#        X_df = pd.concat(event_list)
#        X_da['sample'] = [x for x in X_df.index[::24]]
#        return X_da
#
#    df = preprocess_hydro_pw(
#        pw_station=station,
#        hs_id=hs_id,
#        path=path,
#        hydro_path=hydro_path,
#        with_tide_ends=with_ends, anoms=anoms,
#        pressure_anoms=pressure_anoms,
#        add_pressure=add_pressure)
#    if pressure_anoms is not None:
#        station = pressure_anoms.name
#    # first produce all the positives:
#    # get the tides datetimes:
#    y_pos = df[df['tides'] == 1]['tides']
#    # get the datetimes of 24 hours before tide event (not inclusive):
#    y_lag_pos = y_pos.index - pd.Timedelta(lag, unit='H')
#    masks = [(df.index > start) & (df.index < end)
#             for start, end in zip(y_lag_pos, y_pos.index)]
#    # also drop event if less than 24 hour before available:
#    pw_pos_list = []
#    pressure_pos_list = []
#    ind = []
#    bad_ind = []
#    for i, tide in enumerate(masks):
#        if len(df['tides'][tide]) == (lag - 1):
#            pw_pos_list.append(df[station][tide])
#            pressure_pos_list.append(df['pressure'][tide])
#            ind.append(i)
#        else:
#            bad_ind.append(i)
#    # get the indices of the dropped events:
#    # ind = [x[0] for x in pw_pos_list]
#    if bad_ind:
#        if verbose:
#            print('{} are without full 24 hours before record.'.format(
#                ','.join([x for x in df.iloc[bad_ind].index.strftime('%Y-%m-%d:%H:00:00')])))
#    # drop the events in y so len(y) == in each x from tides_list:
#    y_pos_arr = y_pos.iloc[ind].values
#    # now get the negative y's with neg_pos_ratio (set to 1 if the same pos=neg):
#    y_neg_arr = np.zeros(y_pos_arr.shape[0] * neg_pos_ratio)
#    cnt = 0
#    pw_neg_list = []
#    pressure_neg_list = []
#    np.random.seed(seed)
#    while cnt < len(y_neg_arr):
#        # get a random date from df:
#        r = np.random.randint(low=0, high=len(df))
#        # slice -24 to 24 range with t=0 being the random date:
#        # update: extend the range to -72 hours to 72 hours:
#        lag_factor = 72 / lag
#        slice_range = int(lag * lag_factor)
#        sliced = df.iloc[r - slice_range:r + slice_range]
#        # if tides inside this date range, continue:
#        if y_pos.iloc[ind].index in sliced.index:
#            if verbose:
#                print('found positive tide in randomly sliced 48 window')
#            continue
#        # now if no 24 items exist, also continue:
#        negative = df.iloc[r - lag:r - 1][station]
#        if len(negative) != (lag-1):
#            if verbose:
#                print('didnt find full {} hours sliced negative'.format(lag-1))
#            continue
#        # else, append to pw_neg_list and increase cnt
#        pw_neg_list.append(negative)
#        pressure_neg_list.append(df.iloc[r - lag:r - 1]['pressure'])
#        cnt += 1
#    # lastly, assemble for X, y using np.columnstack:
#    y = np.concatenate([y_pos_arr, y_neg_arr])
#    X = np.stack([[x.values for x in pw_pos_list] +
#                  [x.values for x in pw_neg_list]])
#    X = X.squeeze()
#    pw_pos_da = produce_da_from_list(pw_pos_list, feature='pwv')
#    pw_neg_da = produce_da_from_list(pw_neg_list, feature='pwv')
#    pr_pos_da = produce_da_from_list(pressure_pos_list, feature='pressure')
#    pr_neg_da = produce_da_from_list(pressure_neg_list, feature='pressure')
#    if return_xarray:
#        y = xr.DataArray(y, dims='sample')
#        X_pwv = xr.concat([pw_pos_da, pw_neg_da], 'sample')
#        X_pressure = xr.concat([pr_pos_da, pr_neg_da], 'sample')
#        X = xr.concat([X_pwv, X_pressure], 'feature')
#        X.name = 'X'
#        y['sample'] = X['sample']
#        y.name = 'y'
#        X.attrs['PWV_station'] = station
#        X.attrs['hydro_station_id'] = hs_id
#        y.attrs = X.attrs
#        return X, y
#    else:
#        return X, y


def plot_Xpos_Xneg_mean_std(X_pos_da, X_neg_da):
    import matplotlib.pyplot as plt
    from PW_from_gps_figures import plot_field_with_fill_between
    fig, ax = plt.subplots(figsize=(8, 6))
    posln = plot_field_with_fill_between(X_pos_da, ax=ax, mean_dim='event',
                                         dim='time', color='b', marker='s')
    negln = plot_field_with_fill_between(X_neg_da, ax=ax, mean_dim='event',
                                         dim='time', color='r', marker='o')
    ax.legend(posln+negln, ['Positive tide events', 'Negative tide events'])
    ax.grid()
    return fig


def preprocess_hydro_station(hs_id=48125, hydro_path=hydro_path, max_flow=0,
                             with_tide_ends=False):
    """load hydro station tide events with max_flow and round it up to
    hourly sample rate, with_tide_ends, puts the value 2 at the datetime of
    tide end. regardless 1 is the datetime for tide event."""
    import xarray as xr
    import pandas as pd
    import numpy as np
    # first load tides data:
    all_tides = xr.open_dataset(hydro_path / 'hydro_tides.nc')
    # get all tides for specific station without nans:
    sta_slice = [x for x in all_tides.data_vars if str(hs_id) in x]
    sta_slice = [
        x for x in sta_slice if 'max_flow' in x or 'tide_end' in x or 'tide_max' in x]
    if not sta_slice:
        raise KeyError('hydro station {} not found in database'.format(hs_id))
    tides = all_tides[sta_slice].dropna('tide_start')
    max_flow_tide = tides['TS_{}_max_flow'.format(hs_id)]
    max_flow_attrs = max_flow_tide.attrs
    tide_starts = tides['tide_start'].where(
        ~tides.isnull()).where(max_flow_tide > max_flow).dropna('tide_start')['tide_start']
    tide_ends = tides['TS_{}_tide_end'.format(hs_id)].where(
        ~tides.isnull()).where(max_flow_tide > max_flow).dropna('tide_start')['TS_{}_tide_end'.format(hs_id)]
    max_flows = max_flow_tide.where(
        max_flow_tide > max_flow).dropna('tide_start')
    # round all tide_starts to hourly:
    ts = tide_starts.dt.round('1H')
    max_flows = max_flows.sel(tide_start=ts, method='nearest')
    max_flows['tide_start'] = ts
    ts_end = tide_ends.dt.round('1H')
    time_dt = pd.date_range(
        start=ts.min().values,
        end=ts_end.max().values,
        freq='1H')
    df = pd.DataFrame(data=np.zeros(time_dt.shape), index=time_dt)
    df.loc[ts.values, 0] = 1
    df.loc[ts.values, 1] = max_flows.loc[ts.values]
    df.columns = ['tides', 'max_flow']
    df = df.fillna(0)
    if with_tide_ends:
        df.loc[ts_end.values, :] = 2
    return df, max_flow_attrs


def add_features_and_produce_X_y(hdf, fdf, window_size=25, seed=42,
                                 neg_pos_ratio=1, plot=False):
    """hdf is the hydro events df and fdf is the features df in 'H' freq.
    This function checks the fdf for window-sized data and hour before
    each positive event.
    returns the combined df (hdf+fdf) the positive events labels and features.
    """
    import pandas as pd
    import numpy as np
    import xarray as xr
    # first add check_window_size of 0's to hdf:
    st = hdf.index[0] - pd.Timedelta(window_size, unit='H')
    en = hdf.index[0]
    dts = pd.date_range(st, en - pd.Timedelta(1, unit='H'), freq='H')
    mdf = pd.DataFrame(
        np.zeros(window_size),
        index=dts,
        columns=['tides'])
    hdf = pd.concat([hdf, mdf], axis=0)
    # check for hourly sample rate and concat:
    if not pd.infer_freq(fdf.index) == 'H':
        raise('pls resample fdf to hourly...')
    feature = [x for x in fdf.columns]
    df = pd.concat([hdf, fdf], axis=1)
    # get the tides(positive events) datetimes:
    y_pos = df[df['tides'] == 1]['tides']
    # get the datetimes of 24 hours before tide event (not inclusive):
    y_lag_pos = y_pos.index - pd.Timedelta(window_size, unit='H')
    masks = [(df.index > start) & (df.index < end)
             for start, end in zip(y_lag_pos, y_pos.index)]
    # first check how many full periods of data the feature has:
    avail = [window_size - 1 - df[feature][masks[x]].isnull().sum()
             for x in range(len(masks))]
    adf = pd.DataFrame(avail, index=y_pos.index, columns=feature)
    if plot:
        adf.plot(kind='bar')
    # produce the positive events datetimes for which all the features have
    # window sized data and hour before the event:
    good_dts = adf[adf.loc[:, feature] == window_size - 1].dropna().index
    # y array of positives (1's):
    y_pos_arr = y_pos.loc[good_dts].values
    # now produce the feature list itself:
    good_inds_for_masks = [adf.index.get_loc(x) for x in good_dts]
    good_masks = [masks[x] for x in good_inds_for_masks]
    feature_pos_list = [df[feature][x].values for x in good_masks]
    dts_pos_list = [df[feature][x].index[-1] +
                    pd.Timedelta(1, unit='H') for x in good_masks]
    # TODO: add diagnostic mode for how and where are missing features
    # now get the negative y's with neg_pos_ratio
    # (set to 1 if the same pos=neg):
    y_neg_arr = np.zeros(y_pos_arr.shape[0] * neg_pos_ratio)
    cnt = 0
    feature_neg_list = []
    dts_neg_list = []
    np.random.seed(seed)
    while cnt < len(y_neg_arr):
        # get a random date from df:
        r = np.random.randint(low=0, high=len(df))
        # slice -24 to 24 range with t=0 being the random date:
        # update: extend the range to -72 hours to 72 hours:
        window_factor = 72 / window_size
        slice_range = int(window_size * window_factor)
        sliced = df.iloc[r - slice_range:r + slice_range]
        # if tides inside this date range, continue:
        # try:
        if not (y_pos.loc[good_dts].index.intersection(sliced.index)).empty:
            # print('#')
            continue
        # except TypeError:
        #     return y_pos, good_dts, sliced
        # now if no 24 items exist, also continue:
        negative = df.iloc[r - window_size:r - 1][feature].dropna().values
        if len(negative) != (window_size - 1):
            # print('!')
            continue
        # get the negative datetimes (last record)
        neg_dts = df.iloc[r - window_size:r -
                          1][feature].dropna().index[-1] + pd.Timedelta(1, unit='H')
        # else, append to pw_neg_list and increase cnt
        feature_neg_list.append(negative)
        dts_neg_list.append(neg_dts)
        cnt += 1
        # print(cnt)
    # lastly, assemble for X, y using np.columnstack:
    y = np.concatenate([y_pos_arr, y_neg_arr])
    # TODO: add exception where no features exist, i.e., there is no
    # pw near flood events at all...
    Xpos_da = xr.DataArray(feature_pos_list, dims=['sample', 'window', 'feat'])
    Xpos_da['window'] = np.arange(0, window_size - 1)
    Xpos_da['feat'] = adf.columns
    Xpos_da['sample'] = dts_pos_list
    Xneg_da = xr.DataArray(feature_neg_list, dims=['sample', 'window', 'feat'])
    Xneg_da['window'] = np.arange(0, window_size - 1)
    Xneg_da['feat'] = adf.columns
    Xneg_da['sample'] = dts_neg_list
    X = xr.concat([Xpos_da, Xneg_da], 'sample')
#    if feature_pos_list[0].shape[1] > 0 and feature_neg_list[0].shape[1] > 0:
#        xpos = [x.ravel() for x in feature_pos_list]
#        xneg = [x.ravel() for x in feature_neg_list]
#    X = np.column_stack([[x for x in xpos] +
#                         [x for x in xneg]])
    y_dts = np.stack([[x for x in dts_pos_list]+[x for x in dts_neg_list]])
    y_dts = y_dts.squeeze()
    X_da = X.stack(feature=['feat', 'window'])
    feature = ['_'.join([str(x), str(y)]) for x, y in X_da.feature.values]
    X_da['feature'] = feature
    y_da = xr.DataArray(y, dims=['sample'])
    y_da['sample'] = y_dts
#    feats = []
#    for f in feature:
#        feats.append(['{}_{}'.format(f, x) for x in np.arange(0, window_size
#                                                                   - 1, 1)])
#    X_da['feature'] = [item for sublist in feats for item in sublist]
    return X_da, y_da


# def preprocess_hydro_pw(pw_station='drag', hs_id=48125, path=work_yuval,
#                        ims_path=ims_path,
#                        anoms=True, hydro_path=hydro_path, max_flow=0,
#                        with_tide_ends=False, pressure_anoms=None,
#                        add_pressure=False):
#    import xarray as xr
#    import pandas as pd
#    import numpy as np
#    from aux_gps import anomalize_xr
#    # df.columns = ['tides']
#    # now load pw:
#    if anoms:
#        pw = xr.load_dataset(path / 'GNSS_PW_anom_hourly_50_hour_dayofyear.nc')[pw_station]
#    else:
#        pw = xr.load_dataset(path / 'GNSS_PW_hourly_thresh_50.nc')[pw_station]
#    if pressure_anoms is not None:
#        pw = pressure_anoms
#    pw_df = pw.dropna('time').to_dataframe()
#    # now align the both dataframes:
#    pw_df['tides'] = df['tides']
#    pw_df['max_flow'] = df['max_flow']
#    if add_pressure:
#        pressure = xr.load_dataset(ims_path / 'IMS_BP_israeli_hourly.nc')['JERUSALEM-CENTRE']
#        pressure = anomalize_xr(pressure, freq='MS')
#        pr_df = pressure.dropna('time').to_dataframe()
#        pw_df['pressure'] = pr_df
#    pw_df = pw_df.fillna(0)
#    return pw_df


def loop_over_gnss_hydro_and_aggregate(sel_hydro, pw_anom=False,
                                       pressure_anoms=None,
                                       max_flow_thresh=None,
                                       hydro_path=hydro_path,
                                       work_yuval=work_yuval, ndays=5,
                                       ndays_forward=1,
                                       plot=True, plot_all=False):
    import xarray as xr
    import matplotlib.pyplot as plt
    from aux_gps import path_glob
    filename = 'PW_tide_sites_{}_{}.nc'.format(ndays, ndays_forward)
    if pw_anom:
        filename = 'PW_tide_sites_anom_{}_{}.nc'.format(ndays, ndays_forward)
    gnss_stations = []
    if (hydro_path / filename).is_file():
        print('loading {}...'.format(filename))
        ds = xr.load_dataset(hydro_path / filename)
    else:
        if pw_anom:
            file = path_glob(work_yuval, 'GNSS_PW_anom_*.nc')[-1]
            gnss_pw = xr.open_dataset(file)
        else:
            gnss_pw = xr.open_dataset(
                work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
        just_pw = [x for x in gnss_pw.data_vars if '_error' not in x]
        gnss_pw = gnss_pw[just_pw]
        da_list = []
        for i, gnss_sta in enumerate(just_pw):
            print('proccessing station {}'.format(gnss_sta))
            sliced = sel_hydro[~sel_hydro[gnss_sta].isnull()]
            hydro_ids = [x for x in sliced.id.values]
            if not hydro_ids:
                print(
                    'skipping {} station since no close hydro stations...'.format(gnss_sta))
                continue
            else:
                try:
                    if pressure_anoms is not None:
                        pname = pressure_anoms.name
                        dass = aggregate_get_ndays_pw_hydro(
                            pressure_anoms,
                            hydro_ids,
                            max_flow_thresh=max_flow_thresh,
                            ndays=ndays, ndays_forward=ndays_forward,
                            plot=plot_all)
                        gnss_stations.append(gnss_sta)
                        dass.name = '{}_{}'.format(pname, i)
                    else:
                        dass = aggregate_get_ndays_pw_hydro(
                            gnss_pw[gnss_sta],
                            hydro_ids,
                            max_flow_thresh=max_flow_thresh,
                            ndays=ndays, ndays_forward=ndays_forward,
                            plot=plot_all)
                    da_list.append(dass)
                except ValueError as e:
                    print('skipping {} because {}'.format(gnss_sta, e))
                    continue
        ds = xr.merge(da_list)
        ds.to_netcdf(hydro_path / filename, 'w')
    if plot:
        names = [x for x in ds.data_vars]
        fig, ax = plt.subplots()
        for name in names:
            ds.mean('station').mean('tide_start')[name].plot.line(
                marker='.', linewidth=0., ax=ax)
        if pressure_anoms is not None:
            names = [x.split('_')[0] for x in ds.data_vars]
            names = [x + ' ({})'.format(y)
                     for x, y in zip(names, gnss_stations)]
        ax.set_xlabel('Days before tide event')
        ax.grid()

        hstations = [ds[x].attrs['hydro_stations'] for x in ds.data_vars]
        events = [ds[x].attrs['total_events'] for x in ds.data_vars]
        fmt = list(zip(names, hstations, events))
        ax.legend(['{} with {} stations ({} total events)'.format(x, y, z)
                   for x, y, z in fmt])
        if pw_anom:
            title = 'Mean PWV anomalies for tide stations near all GNSS stations'
            ylabel = 'PWV anomalies [mm]'
        else:
            title = 'Mean PWV for tide stations near all GNSS stations'
            ylabel = 'PWV [mm]'
        if max_flow_thresh is not None:
            title += ' (max_flow > {} m^3/sec)'.format(max_flow_thresh)
        if pressure_anoms is not None:
            ylabel = 'Surface pressure anomalies [hPa]'
            title = 'Mean surface pressure anomaly in {} for all tide stations near GNSS stations'.format(
                pname)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    return ds


def aggregate_get_ndays_pw_hydro(pw_da, hs_ids, max_flow_thresh=None,
                                 hydro_path=hydro_path, ndays=5,
                                 ndays_forward=1, plot=True):
    import xarray as xr
    import matplotlib.pyplot as plt
    das = []
    max_flows_list = []
    pw_ndays_list = []
    if not isinstance(hs_ids, list):
        hs_ids = [int(hs_ids)]
    else:
        hs_ids = [int(x) for x in hs_ids]
    used_ids = []
    events = []
    for sid in hs_ids:
        print('proccessing hydro station {}'.format(sid))
        try:
            max_flows, pw_ndays, da = get_n_days_pw_hydro_all(pw_da, sid,
                                                              max_flow_thresh=max_flow_thresh,
                                                              hydro_path=hydro_path,
                                                              ndays=ndays, ndays_forward=ndays_forward,
                                                              return_max_flows=True,
                                                              plot=False)
            das.append(da)
            pw_ndays_list.append(pw_ndays)
            max_flows_list.append(max_flows)
            used_ids.append(sid)
            events.append(max_flows.size)
        except KeyError as e:
            print('{}, skipping...'.format(e))
            continue
        except ValueError as e:
            print('{}, skipping...'.format(e))
            continue
    pw_ndays = xr.concat(pw_ndays_list, 'time')
    dass = xr.concat(das, 'station')
    dass['station'] = used_ids
    dass.name = pw_da.name
    dass.attrs['hydro_stations'] = len(used_ids)
    dass.attrs['total_events'] = sum(events)
    if plot:
        fig, ax = plt.subplots(figsize=(20, 4))
        color = 'tab:blue'
        pw_ndays.plot.line(marker='.', linewidth=0., color=color, ax=ax)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel('PW [mm]', color=color)
        ax2 = ax.twinx()
        color = 'tab:red'
        for mf in max_flows_list:
            mf.plot.line(marker='X', linewidth=0., color=color, ax=ax2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax.grid()
        ax2.set_title(
            'PW in station {} {} days before tide events ({} total)'.format(
                pw_da.name, ndays, sum(events)))
        ax2.set_ylabel('max_flow [m^3/sec]', color=color)
        fig.tight_layout()
        fig, ax = plt.subplots()
        for sid in used_ids:
            dass.sel(
                station=sid).mean('tide_start').plot.line(
                marker='.', linewidth=0., ax=ax)
        ax.set_xlabel('Days before tide event')
        ax.set_ylabel('PW [mm]')
        ax.grid()
        fmt = list(zip(used_ids, events))
        ax.legend(['station #{} ({} events)'.format(x, y) for x, y in fmt])
        ax.set_title(
            'Mean PW for tide stations near {} station'.format(pw_da.name))
        if max_flow_thresh is not None:
            ax.set_title(
                'Mean PW for tide stations (above {} m^3/sec) near {} station'.format(
                    max_flow_thresh, pw_da.name))
    return dass


def produce_pwv_days_before_tide_events(pw_da, hs_df, days_prior=1, drop_thresh=0.5,
                                        days_after=1, plot=False, verbose=0,
                                        max_gap='12H', rolling=12):
    """
    takes pwv and hydro tide dates from one station and
    rounds the hydro tides dates to 5 min
    selects the tides dates that are at least the first date of pwv available
    then if no pwv data prior to 1 day of tides date - drops
    if more than half day missing - drops
    then interpolates the missing pwv data points using spline
    returns the dataframes contains pwv 1 day before and after tides
    and pwv's 1 day prior to event and 1 day after.

    Parameters
    ----------
    pw_da : TYPE
        pwv of station.
    hs_df : TYPE
        hydro tide dataframe for one station.
    days_prior : TYPE, optional
        DESCRIPTION. The default is 1.
    drop_thresh : TYPE, optional
        DESCRIPTION. The default is 0.5.
    days_after : TYPE, optional
        DESCRIPTION. The default is 1.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    max_gap : TYPE, optional
        DESCRIPTION. The default is '12H'.
    rolling : TYPE, optional
        DESCRIPTION. The default is 12.

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    pwv_after_list : TYPE
        DESCRIPTION.
    pwv_prior_list : TYPE
        DESCRIPTION.

    """
    import pandas as pd
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    if rolling is not None:
        pw_da = pw_da.rolling(time=rolling, center=True).mean(keep_attrs=True)
    if drop_thresh is None:
        drop_thresh = 0
    # first infer time freq of pw_da:
    freq = xr.infer_freq(pw_da['time'])
    if freq == '5T':
        pts_per_day = 288
        timedelta = pd.Timedelta(5, unit='min')
    if freq == '1H' or freq == 'H':
        pts_per_day = 24
        timedelta = pd.Timedelta(1, unit='H')
    # get the minimum dt of the pwv station:
    min_dt = pw_da.dropna('time').time.min().values
    # round the hs_df to 5 mins, and find the closest min_dt:
    hs_df.index = hs_df.index.round(freq)
    hs_df = hs_df[~hs_df.index.duplicated(keep='first')]
    hs_df = hs_df.sort_index()
    min_ind = hs_df.index.get_loc(min_dt, method='nearest')
    # slice the tides data accordinaly:
    hs_df = hs_df.iloc[min_ind:].dropna()
    # loop over each tide start and grab the datetimes
    pwv_prior_list = []
    pwv_after_list = []
    # se_list = []
    tot_events = hs_df.index.size
    event_cnt = 0
    dropped_thresh = 0
    dropped_no_data = 0
    for ts in hs_df.index:
        dt_prior = ts - pd.Timedelta(days_prior, unit='d')
        dt_after = ts + pd.Timedelta(days_after, unit='d')
        after_da = pw_da.sel(time=slice(ts, dt_after))
        prior_da = pw_da.sel(time=slice(dt_prior, ts - timedelta))
        if prior_da.dropna('time').size == 0:
            if verbose == 1:
                print('{} found no prior data for PWV {} days prior'.format(
                    ts.strftime('%Y-%m-%d %H:%M'), days_prior))
            dropped_no_data += 1
            continue
        elif prior_da.dropna('time').size < pts_per_day*drop_thresh:
            if verbose == 1:
                print('{} found less than {} a day prior data for PWV {} days prior'.format(
                    ts.strftime('%Y-%m-%d %H:%M'), drop_thresh, days_prior))
            dropped_thresh += 1
            continue
        if max_gap is not None:
            prior_da = prior_da.interpolate_na(
                'time', method='spline', max_gap=max_gap, keep_attrs=True)
        event_cnt += 1
        # if rolling is not None:
        #     after_da = after_da.rolling(time=rolling, center=True, keep_attrs=True).mean(keep_attrs=True)
        #     prior_da = prior_da.rolling(time=rolling, center=True, keep_attrs=True).mean(keep_attrs=True)
        # after_da.name = pw_da.name + '_{}'.format(i)
        pwv_after_list.append(after_da)
        pwv_prior_list.append(prior_da)
        # se = da.reset_index('time', drop=True).to_dataframe()[da.name]
        # se_list.append(se)
    se_list = []
    for i, (prior, after) in enumerate(zip(pwv_prior_list, pwv_after_list)):
        # return prior, after
        # df_p = prior.to_dataframe()
        # df_a = after.to_dataframe()
        # return df_p, df_a
        da = xr.concat([prior, after], 'time')
        # print(da)
        se = da.reset_index('time', drop=True).to_dataframe()
        se.columns = [da.name + '_{}'.format(i)]
        # print(se)
        # [da.name + '_{}'.format(i)]
        se_list.append(se)
    df = pd.concat(se_list, axis=1)
    df = df.iloc[:-1]
    df.index = np.arange(-days_prior, days_after, 1/pts_per_day)
    if verbose >= 0:
        print('total events with pwv:{} , dropped due to no data: {}, dropped due to thresh:{}, left events: {}'.format(
            tot_events, dropped_no_data, dropped_thresh, event_cnt))
    if plot:
        ax = df.T.mean().plot()
        ax.grid()
        ax.axvline(color='k', linestyle='--')
        ax.set_xlabel('Days before tide event')
        ax.set_ylabel('PWV anomalies [mm]')
        ax.set_title('GNSS station: {} with {} events'.format(
            pw_da.name.upper(), event_cnt))
        better = df.copy()
        better.index = pd.to_timedelta(better.index, unit='d')
        better = better.resample('15S').interpolate(
            method='cubic').T.mean().resample('5T').mean()
        better = better.reset_index(drop=True)
        better.index = np.linspace(-days_prior, days_after, better.index.size)
        better.plot(ax=ax)
        # fig, ax = plt.subplots(figsize=(20, 7))
    #     [pwv.plot.line(ax=ax) for pwv in pwv_list]
    return df, pwv_after_list, pwv_prior_list


def get_n_days_pw_hydro_all(pw_da, hs_id, max_flow_thresh=None,
                            hydro_path=hydro_path, ndays=5, ndays_forward=1,
                            return_max_flows=False, plot=True):
    """calculate the mean of the PW ndays before all tide events in specific
    hydro station. can use max_flow_thresh to get only event with al least
    this max_flow i.e., big tide events"""
    # important, DO NOT dropna pw_da!
    import xarray as xr
    import matplotlib.pyplot as plt
    import pandas as pd

    def get_n_days_pw_hydro_one_event(pw_da, tide_start, ndays=ndays, ndays_forward=0):
        freq = pd.infer_freq(pw_da.time.values)
        # for now, work with 5 mins data:
        if freq == '5T':
            points = int(ndays) * 24 * 12
            points_forward = int(ndays_forward) * 24 * 12
        elif freq == '10T':
            points = int(ndays) * 24 * 6
            points_forward = int(ndays_forward) * 24 * 6
        elif freq == 'H':
            points = int(ndays) * 24
            points_forward = int(ndays_forward) * 24
        lag = pd.timedelta_range(end=0, periods=points, freq=freq)
        forward_lag = pd.timedelta_range(
            start=0, periods=points_forward, freq=freq)
        lag = lag.union(forward_lag)
        time_arr = pd.to_datetime(pw_da.time.values)
        tide_start = pd.to_datetime(tide_start).round(freq)
        ts_loc = time_arr.get_loc(tide_start)
        # days = pd.Timedelta(ndays, unit='D')
        # time_slice = [tide_start - days, tide_start]
        # pw = pw_da.sel(time=slice(*time_slice))
        pw = pw_da.isel(time=slice(ts_loc - points,
                                   ts_loc + points_forward - 1))
        return pw, lag

    # first load tides data:
    all_tides = xr.open_dataset(hydro_path / 'hydro_tides.nc')
    # get all tides for specific station without nans:
    sta_slice = [x for x in all_tides.data_vars if str(hs_id) in x]
    if not sta_slice:
        raise KeyError('hydro station {} not found in database'.format(hs_id))
    tides = all_tides[sta_slice].dropna('tide_start')
    tide_starts = tides['tide_start'].where(
        ~tides.isnull()).dropna('tide_start')['tide_start']
    # get max flow tides data:
    mf = [x for x in tides.data_vars if 'max_flow' in x]
    max_flows = tides[mf].dropna('tide_start').to_array('max_flow').squeeze()
    # also get tide end and tide max data:
#    te = [x for x in tides.data_vars if 'tide_end' in x]
#    tide_ends = tides[te].dropna('tide_start').to_array('tide_end').squeeze()
#    tm = [x for x in tides.data_vars if 'tide_max' in x]
#    tide_maxs = tides[tm].dropna('tide_start').to_array('tide_max').squeeze()
    # slice minmum time for convenience:
    min_pw_time = pw_da.dropna('time').time.min().values
    tide_starts = tide_starts.sel(tide_start=slice(min_pw_time, None))
    max_flows = max_flows.sel(tide_start=slice(min_pw_time, None))
    # filter if hydro station data ends before gnss pw:
    if tide_starts.size == 0:
        raise ValueError('tides data end before gnss data begin')
    if max_flow_thresh is not None:
        # pick only big events:
        max_flows = max_flows.where(
            max_flows > max_flow_thresh).dropna('tide_start')
        tide_starts = tide_starts.where(
            max_flows > max_flow_thresh).dropna('tide_start')
    pw_list = []
    for ts in tide_starts.values:
        #        te = tide_ends.sel(tide_start=ts).values
        #        tm = tide_maxs.sel(tide_start=ts).values
        pw, lag = get_n_days_pw_hydro_one_event(
            pw_da, ts, ndays=ndays, ndays_forward=ndays_forward)
        pw.attrs['ts'] = ts
        pw_list.append(pw)
    # filter events that no PW exists:
    pw_list = [x for x in pw_list if x.dropna('time').size > 0]
    da = xr.DataArray([x.values for x in pw_list], dims=['tide_start', 'lag'])
    da['tide_start'] = [x.attrs['ts'] for x in pw_list]  # tide_starts
    da['lag'] = lag
    # da.name = pw_da.name + '_tide_events'
    da.attrs = pw_da.attrs
    if max_flow_thresh is not None:
        da.attrs['max_flow_minimum'] = max_flow_thresh
    pw_ndays = xr.concat(pw_list, 'time')
    if plot:
        fig, ax = plt.subplots(figsize=(20, 4))
        color = 'tab:blue'
        pw_ndays.plot.line(marker='.', linewidth=0., color=color, ax=ax)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel('PW [mm]', color=color)
        ax2 = ax.twinx()
        color = 'tab:red'
        max_flows.plot.line(marker='X', linewidth=0., color=color, ax=ax2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax.grid()
        ax2.set_title(
            'PW in station {} {} days before tide events'.format(
                pw_da.name, ndays))
        ax2.set_ylabel('max_flow [m^3/sec]', color=color)
        fig.tight_layout()
        fig, ax = plt.subplots()
        da.mean('tide_start').plot.line(marker='.', linewidth=0., ax=ax)
        ax.set_xlabel('Days before tide event')
        ax.set_ylabel('PW [mm]')
        ax.grid()
        ax.set_title(
            'Mean PW for {} tide events near {} station'.format(
                da.tide_start.size, pw_da.name))
        if max_flow_thresh is not None:
            ax.set_title(
                'Mean PW for {} tide events (above {} m^3/sec) near {} station'.format(
                    da.tide_start.size, max_flow_thresh, pw_da.name))
    if return_max_flows:
        return max_flows, pw_ndays, da
    else:
        return da


def calculate_distance_between_two_latlons_israel(lat1, lon1, lat2, lon2):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    points = np.array(([lat1, lon1], [lat2, lon2]))
    df = pd.DataFrame(points, columns=['lat', 'lon'])
    pdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                           crs={'init': 'epsg:4326'})
    pdf_meters = pdf.to_crs({'init': 'epsg:6991'})
    # distance in meters:
    distance = pdf_meters.geometry[0].distance(pdf_meters.geometry[1])
    return distance


def get_hydro_near_GNSS(radius=5, n=5, hydro_path=hydro_path,
                        gis_path=gis_path, plot=True):
    import pandas as pd
    import geopandas as gpd
    from pathlib import Path
    import xarray as xr
    import matplotlib.pyplot as plt
    df = pd.read_csv(Path().cwd() / 'israeli_gnss_coords.txt',
                     delim_whitespace=True)
    df = df[['lon', 'lat']]
    gnss = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                            crs={'init': 'epsg:4326'})
    gnss = gnss.to_crs({'init': 'epsg:2039'})
    hydro_meta = read_hydro_metadata(hydro_path, gis_path, plot=False)
    hydro_meta = hydro_meta.to_crs({'init': 'epsg:2039'})
    for index, row in gnss.iterrows():
        # hdict[index] = hydro_meta.geometry.distance(row['geometry'])
        hydro_meta[index] = hydro_meta.geometry.distance(row['geometry'])
        hydro_meta[index] = hydro_meta[index].where(
            hydro_meta[index] <= radius * 1000)
    gnss_list = [x for x in gnss.index]
    # get only stations within desired radius
    mask = ~hydro_meta.loc[:, gnss_list].isnull().all(axis=1)
    sel_hydro = hydro_meta.copy()[mask]  # pd.concat(hydro_list)
    # filter unexisting stations:
    tides = xr.load_dataset(hydro_path / 'hydro_tides.nc')
    to_remove = []
    for index, row in sel_hydro.iterrows():
        sid = row['id']
        try:
            tides['TS_{}_max_flow'.format(sid)]
        except KeyError:
            print('{} hydro station non-existant in database'.format(sid))
            to_remove.append(index)
    sel_hydro.drop(to_remove, axis=0, inplace=True)
    if plot:
        isr = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
        isr.crs = {'init': 'epsg:4326'}
        gnss = gnss.to_crs({'init': 'epsg:4326'})
        sel_hydro = sel_hydro.to_crs({'init': 'epsg:4326'})
        ax = isr.plot(figsize=(10, 16))
        sel_hydro.plot(ax=ax, color='yellow', edgecolor='black')
        gnss.plot(ax=ax, color='green', edgecolor='black', alpha=0.7)
        for x, y, label in zip(gnss.lon, gnss.lat, gnss.index):
            ax.annotate(label, xy=(x, y), xytext=(3, 3),
                        textcoords="offset points")
        plt.legend(['hydro-tide stations', 'GNSS stations'], loc='upper left')
        plt.suptitle(
            'hydro-tide stations within {} km of a GNSS station'.format(radius), fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
#        for x, y, label in zip(sel_hydro.lon, sel_hydro.lat,
#                               sel_hydro.id):
#            ax.annotate(label, xy=(x, y), xytext=(3, 3),
#                        textcoords="offset points")
    return sel_hydro


def read_hydro_metadata(path=hydro_path, gis_path=gis_path, plot=True):
    import pandas as pd
    import geopandas as gpd
    import xarray as xr
    df = pd.read_excel(hydro_path / 'hydro_stations_metadata.xlsx',
                       header=4)
    # drop last row:
    df.drop(df.tail(1).index, inplace=True)  # drop last n rows
    df.columns = [
        'id',
        'name',
        'active',
        'agency',
        'type',
        'X',
        'Y',
        'area']
    df.loc[:, 'active'][df['active'] == 'פעילה'] = 1
    df.loc[:, 'active'][df['active'] == 'לא פעילה'] = 0
    df.loc[:, 'active'][df['active'] == 'לא פעילה זמנית'] = 0
    df['active'] = df['active'].astype(float)
    df = df[~df.X.isnull()]
    df = df[~df.Y.isnull()]
    # now, geopandas part:
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                              crs={'init': 'epsg:2039'})
    # geo_df.crs = {'init': 'epsg:2039'}
    geo_df = geo_df.to_crs({'init': 'epsg:4326'})
    isr_dem = xr.open_rasterio(gis_path / 'israel_dem.tif')
    alt_list = []

    for index, row in geo_df.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        alt = isr_dem.sel(band=1, x=lon, y=lat, method='nearest').values.item()
        alt_list.append(float(alt))
    geo_df['alt'] = alt_list
    geo_df['lat'] = geo_df.geometry.y
    geo_df['lon'] = geo_df.geometry.x
    isr = gpd.read_file(gis_path / 'Israel_and_Yosh.shp')
    isr.crs = {'init': 'epsg:4326'}
    geo_df = gpd.sjoin(geo_df, isr, op='within')
    if plot:
        ax = isr.plot()
        geo_df.plot(ax=ax, edgecolor='black', legend=True)
    return geo_df


def read_tides(path=hydro_path):
    from aux_gps import path_glob
    import pandas as pd
    import xarray as xr
    from aux_gps import get_unique_index
    files = path_glob(path, 'tide_report*.xlsx')
    df_list = []
    for file in files:
        df = pd.read_excel(file, header=4)
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        df.columns = [
            'id',
            'name',
            'hydro_year',
            'tide_start_hour',
            'tide_start_date',
            'tide_end_hour',
            'tide_end_date',
            'tide_duration',
            'tide_max_hour',
            'tide_max_date',
            'max_height',
            'max_flow[m^3/sec]',
            'tide_vol[MCM]']
        df = df[~df.hydro_year.isnull()]
        df['id'] = df['id'].astype(int)
        df['tide_start'] = pd.to_datetime(
            df['tide_start_date'], dayfirst=True) + pd.to_timedelta(
            df['tide_start_hour'].add(':00'), unit='m', errors='coerce')
        # tides are in local Israeli winter clock (no DST):
        # dst = np.zeros(df['tide_start'].shape)
        # df['tide_start'] = df['tide_start'].dt.tz_localize('Asia/Jerusalem', ambiguous=dst).dt.tz_convert('UTC')
        df['tide_start'] = df['tide_start'] - pd.Timedelta(2, unit='H')
        df['tide_end'] = pd.to_datetime(
            df['tide_end_date'], dayfirst=True) + pd.to_timedelta(
            df['tide_end_hour'].add(':00'),
            unit='m',
            errors='coerce')
        # also to tide ends:
        df['tide_end'] = df['tide_end'] - pd.Timedelta(2, unit='H')
        # df['tide_end'] = df['tide_end'].dt.tz_localize('Asia/Jerusalem', ambiguous=dst).dt.tz_convert('UTC')
        df['tide_max'] = pd.to_datetime(
            df['tide_max_date'], dayfirst=True) + pd.to_timedelta(
            df['tide_max_hour'].add(':00'),
            unit='m',
            errors='coerce')
        # also to tide max:
        # df['tide_max'] = df['tide_max'].dt.tz_localize('Asia/Jerusalem', ambiguous=dst).dt.tz_convert('UTC')
        df['tide_max'] = df['tide_max'] - pd.Timedelta(2, unit='H')
        df['tide_duration'] = pd.to_timedelta(
            df['tide_duration'] + ':00', unit='m', errors='coerce')
        df.loc[:,
               'max_flow[m^3/sec]'][df['max_flow[m^3/sec]'].str.contains('<',
                                                                         na=False)] = 0
        df.loc[:, 'tide_vol[MCM]'][df['tide_vol[MCM]'].str.contains(
            '<', na=False)] = 0
        df['max_flow[m^3/sec]'] = df['max_flow[m^3/sec]'].astype(float)
        df['tide_vol[MCM]'] = df['tide_vol[MCM]'].astype(float)
        to_drop = ['tide_start_hour', 'tide_start_date', 'tide_end_hour',
                   'tide_end_date', 'tide_max_hour', 'tide_max_date']
        df = df.drop(to_drop, axis=1)
        df_list.append(df)
    df = pd.concat(df_list)
    dfs = [x for _, x in df.groupby('id')]
    ds_list = []
    meta_df = read_hydro_metadata(path, gis_path, False)
    for df in dfs:
        st_id = df['id'].iloc[0]
        st_name = df['name'].iloc[0]
        print('proccessing station number: {}, {}'.format(st_id, st_name))
        meta = meta_df[meta_df['id'] == st_id]
        ds = xr.Dataset()
        df.set_index('tide_start', inplace=True)
        attrs = {}
        attrs['station_name'] = st_name
        if not meta.empty:
            attrs['lon'] = meta.lon.values.item()
            attrs['lat'] = meta.lat.values.item()
            attrs['alt'] = meta.alt.values.item()
            attrs['drainage_basin_area'] = meta.area.values.item()
            attrs['active'] = meta.active.values.item()
        attrs['units'] = 'm'
        max_height = df['max_height'].to_xarray()
        max_height.name = 'TS_{}_max_height'.format(st_id)
        max_height.attrs = attrs
        max_flow = df['max_flow[m^3/sec]'].to_xarray()
        max_flow.name = 'TS_{}_max_flow'.format(st_id)
        attrs['units'] = 'm^3/sec'
        max_flow.attrs = attrs
        attrs['units'] = 'MCM'
        tide_vol = df['tide_vol[MCM]'].to_xarray()
        tide_vol.name = 'TS_{}_tide_vol'.format(st_id)
        tide_vol.attrs = attrs
        attrs.pop('units')
#        tide_start = df['tide_start'].to_xarray()
#        tide_start.name = 'TS_{}_tide_start'.format(st_id)
#        tide_start.attrs = attrs
        tide_end = df['tide_end'].to_xarray()
        tide_end.name = 'TS_{}_tide_end'.format(st_id)
        tide_end.attrs = attrs
        tide_max = df['tide_max'].to_xarray()
        tide_max.name = 'TS_{}_tide_max'.format(st_id)
        tide_max.attrs = attrs
        ds['{}'.format(max_height.name)] = max_height
        ds['{}'.format(max_flow.name)] = max_flow
        ds['{}'.format(tide_vol.name)] = tide_vol
#         ds['{}'.format(tide_start.name)] = tide_start
        ds['{}'.format(tide_end.name)] = tide_end
        ds['{}'.format(tide_max.name)] = tide_max
        ds_list.append(ds)
    dsu = [get_unique_index(x, dim='tide_start') for x in ds_list]
    print('merging...')
    ds = xr.merge(dsu)
    ds.attrs['time'] = 'UTC'
    filename = 'hydro_tides.nc'
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def plot_hydro_events(hs_id, path=hydro_path, field='max_flow', min_flow=10):
    import xarray as xr
    import matplotlib.pyplot as plt
    tides = xr.open_dataset(path/'hydro_tides.nc')
    sta_slice = [x for x in tides.data_vars if str(hs_id) in x]
    tide = tides[sta_slice]['TS_{}_{}'.format(hs_id, field)]
    tide = tide.dropna('tide_start')
    fig, ax = plt.subplots()
    tide.plot.line(linewidth=0., marker='x', color='r', ax=ax)
    if min_flow is not None:
        tide[tide > min_flow].plot.line(
            linewidth=0., marker='x', color='b', ax=ax)
        print('min flow of {} m^3/sec: {}'.format(min_flow,
                                                  tide[tide > min_flow].dropna('tide_start').size))
    return tide


def text_process_hydrographs(path=hydro_path, gis_path=gis_path):
    from aux_gps import path_glob
    files = path_glob(path, 'hydro_flow*.txt')
    for i, file in enumerate(files):
        print(file)
        with open(file, 'r') as f:
            big_list = f.read().splitlines()
        # for small_list in big_list:
        #     flat_list = [item for sublist in l7 for item in sublist]
        big = [x.replace(',', ' ') for x in big_list]
        big = big[6:]
        big = [x.replace('\t', ',') for x in big]
        filename = 'hydro_graph_{}.txt'.format(i)
        with open(path / filename, 'w') as fs:
            for item in big:
                fs.write('{}\n'.format(item))
        print('{} saved to {}'.format(filename, path))
    return


def read_hydrographs(path=hydro_path):
    from aux_gps import path_glob
    import pandas as pd
    import xarray as xr
    from aux_gps import get_unique_index
    files = path_glob(path, 'hydro_graph*.txt')
    df_list = []
    for file in files:
        print(file)
        df = pd.read_csv(file, header=0, sep=',')
        df.columns = [
            'id',
            'name',
            'time',
            'tide_height[m]',
            'flow[m^3/sec]',
            'data_type',
            'flow_type',
            'record_type',
            'record_code']
        # make sure the time is in UTC since database is in ISR winter clock (no DST)
        df['time'] = pd.to_datetime(df['time'], dayfirst=True) - pd.Timedelta(2, unit='H')
        df['tide_height[m]'] = df['tide_height[m]'].astype(float)
        df['flow[m^3/sec]'] = df['flow[m^3/sec]'].astype(float)
        df.loc[:, 'data_type'][df['data_type'].str.contains(
            'מדודים', na=False)] = 'measured'
        df.loc[:, 'data_type'][df['data_type'].str.contains(
            'משוחזרים', na=False)] = 'reconstructed'
        df.loc[:, 'flow_type'][df['flow_type'].str.contains(
            'תקין', na=False)] = 'normal'
        df.loc[:, 'flow_type'][df['flow_type'].str.contains(
            'גאות', na=False)] = 'tide'
        df.loc[:, 'record_type'][df['record_type'].str.contains(
            'נקודה פנימית', na=False)] = 'inner_point'
        df.loc[:, 'record_type'][df['record_type'].str.contains(
            'נקודה פנימית', na=False)] = 'inner_point'
        df.loc[:, 'record_type'][df['record_type'].str.contains(
            'התחלת קטע', na=False)] = 'section_begining'
        df.loc[:, 'record_type'][df['record_type'].str.contains(
            'סיום קטע', na=False)] = 'section_ending'
        df_list.append(df)
    df = pd.concat(df_list)
    dfs = [x for _, x in df.groupby('id')]
    ds_list = []
    meta_df = read_hydro_metadata(path, gis_path, False)
    for df in dfs:
        st_id = df['id'].iloc[0]
        st_name = df['name'].iloc[0]
        print('proccessing station number: {}, {}'.format(st_id, st_name))
        meta = meta_df[meta_df['id'] == st_id]
        ds = xr.Dataset()
        df.set_index('time', inplace=True)
        attrs = {}
        attrs['station_name'] = st_name
        if not meta.empty:
            attrs['lon'] = meta.lon.values.item()
            attrs['lat'] = meta.lat.values.item()
            attrs['alt'] = meta.alt.values.item()
            attrs['drainage_basin_area'] = meta.area.values.item()
            attrs['active'] = meta.active.values.item()
        attrs['units'] = 'm'
        tide_height = df['tide_height[m]'].to_xarray()
        tide_height.name = 'HS_{}_tide_height'.format(st_id)
        tide_height.attrs = attrs
        flow = df['flow[m^3/sec]'].to_xarray()
        flow.name = 'HS_{}_flow'.format(st_id)
        attrs['units'] = 'm^3/sec'
        flow.attrs = attrs
        ds['{}'.format(tide_height.name)] = tide_height
        ds['{}'.format(flow.name)] = flow
        ds_list.append(ds)
    dsu = [get_unique_index(x) for x in ds_list]
    print('merging...')
    ds = xr.merge(dsu)
    ds.attrs['time'] = 'UTC'
    filename = 'hydro_graphs.nc'
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


def read_station_from_tide_database(hs_id=48125, rounding='1H',
                                    hydro_path=hydro_path):
    import xarray as xr
    all_tides = xr.open_dataset(hydro_path / 'hydro_tides.nc')
    # get all tides for specific station without nans:
    sta_slice = [x for x in all_tides.data_vars if str(hs_id) in x]
    if not sta_slice:
        raise KeyError('hydro station {} not found in database'.format(hs_id))
    # tides = all_tides[sta_slice].dropna('tide_start')
    df = all_tides[sta_slice].to_dataframe()
    df.columns = ['max_height', 'max_flow', 'tide_vol', 'tide_end', 'tide_max']
    df = df[df['max_flow'] != 0]
    df['hydro_station_id'] = hs_id
    if rounding is not None:
        print('rounding to {}'.format(rounding))
        df.index = df.index.round(rounding)
    return df
    # tide_starts = tides['tide_start'].where(
    #     ~tides.isnull()).dropna('tide_start')['tide_start']


def check_if_tide_events_from_stations_are_within_time_window(df_list, rounding='H',
                                                              days=1, return_hs_list=False):
    import pandas as pd
    dfs = []
    for i, df in enumerate(df_list):
        df.dropna(inplace=True)
        if rounding is not None:
            df.index = df.index.round(rounding)
        dfs.append(df['hydro_station_id'])
    df = pd.concat(dfs, axis=0).to_frame()
    df['time'] = df.index
    df = df.sort_index()
    # filter co-tide events:
    df = df.loc[~df.index.duplicated()]
    print('found {} co-tide events'.format(df.index.duplicated().sum()))
    # secondly check for events that are within days period of each other and filter:
    dif = df['time'].diff()
    mask = abs(dif) <= pd.Timedelta(days, unit='D')
    dupes = dif[mask].index
    print('found {} tide events that are within {} of each other.'.format(
        dupes.size, days))
    print(df.loc[dupes, 'hydro_station_id'])
    df = df.loc[~mask]
    if return_hs_list:
        hs_ids = [x['hydro_station_id'].iloc[0] for x in df_list]
        df_list = [df[df['hydro_station_id'] == x] for x in hs_ids]
        return df_list
    else:
        return df


def scorers(scorer_str):
    from sklearn.metrics import make_scorer
    if scorer_str == 'tss':
        return make_scorer(tss_score)
    elif scorer_str == 'hss':
        return make_scorer(hss_score)
    else:
        return scorer_str

def scorer_function(scorer_label, y_true, y_pred):
    import sklearn.metrics as sm
    if scorer_label == 'precision':
        return sm.precision_score(y_true, y_pred)
    elif scorer_label == 'recall':
        return sm.recall_score(y_true, y_pred)
    elif scorer_label == 'f1':
        return sm.f1_score(y_true, y_pred)
    elif scorer_label == 'accuracy':
        return sm.accuracy_score(y_true, y_pred)
    elif scorer_label == 'tss':
        return tss_score(y_true, y_pred)
    elif scorer_label == 'hss':
        return hss_score(y_true, y_pred)
    else:
        raise('{} is not implemented yet'.format(scorer_label))


def acc_score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def tss_score(y, y_pred, **kwargs):
    from sklearn.metrics import confusion_matrix
    # if y == y_pred:
    #     raise ValueError('y_true == y_pred either 0 or 1')
    # else:
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # print('TN: {}'.format(tn))
    # print('FP: {}'.format(fp))
    # print('FN: {}'.format(fn))
    # print('TP: {}'.format(tp))
    tss = tp / (tp + fn) - fp / (fp + tn)
    # print('TSS: {}'.format(tss))
    return tss


def hss_score(y, y_pred, **kwargs):
    from sklearn.metrics import confusion_matrix
    # if y == y_pred:
    #     raise ValueError('y_true == y_pred either 0 or 1')
    # else:
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # print('TN: {}'.format(tn))
    # print('FP: {}'.format(fp))
    # print('FN: {}'.format(fn))
    # print('TP: {}'.format(tp))
    # if (tp+fn) == 0 or (fn+tn) == 0 :
    #     raise ValueError('TSS undefined, denom is 0!')
    hss = 2 * (tp*tn - fn*fp)/((tp+fn)*(fn+tn)+(tp+fn)*(fp+tn))
    # print('HSS: {}'.format(hss))
    return hss


def order_of_mag(minimal=-5, maximal=1):
    import numpy as np
    return [10**float(x) for x in np.arange(minimal, maximal + 1)]


class ML_Classifier_Switcher(object):

    def pick_model(self, model_name, pgrid='normal'):
        """Dispatch method"""
        # from sklearn.model_selection import GridSearchCV
        self.param_grid = None
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
#        if gridsearch:
#            return(GridSearchCV(method(), self.param_grid, n_jobs=-1,
#                                return_train_score=True))
#        else:
        # Call the method as we return it
        # whether to select lighter param grid, e.g., for testing purposes.
        self.pgrid = pgrid
        return method()

    def SVC(self):
        from sklearn.svm import SVC
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
                               'C': [0.1, 100],
                               'gamma': [0.0001, 1],
                               'degree': [1, 2, 5],
                               'coef0': [0, 1, 4]}
        elif self.pgrid == 'normal':
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
                               'C': order_of_mag(-1, 2),
                               'gamma': order_of_mag(-5, 0),
                               'degree': [1, 2, 3, 4, 5],
                               'coef0': [0, 1, 2, 3, 4]}
        elif self.pgrid == 'dense':
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
                               'C': order_of_mag(-2, 2),
                               'gamma': order_of_mag(-5, 0),
                               'degree': [1, 2, 3, 4, 5],
                               'coef0': [0, 1, 2, 3, 4]}
        return SVC(random_state=42, class_weight=None)

    def MLP(self):
        import numpy as np
        from sklearn.neural_network import MLPClassifier
        if self.pgrid == 'light':
            self.param_grid = {
                'activation': [
                    'identity',
                    'relu'],
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50)]}
        elif self.pgrid == 'normal':
            self.param_grid = {'alpha': order_of_mag(-5, 1),
                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
                               'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                               'learning_rate': ['constant', 'adaptive'],
                               'solver': ['adam', 'lbfgs', 'sgd']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': order_of_mag(-5, 1),
                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
                               'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                               'learning_rate': ['constant', 'adaptive'],
                               'solver': ['adam', 'lbfgs', 'sgd']}
        return MLPClassifier(random_state=42, max_iter=2000)

    def RF(self):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'max_features': ['auto', 'sqrt']}
        elif self.pgrid == 'normal':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10],
                               'min_samples_split': [2, 5, 15, 50],
                               'n_estimators': [100, 300, 700, 1200]
                               }
        elif self.pgrid == 'dense':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100, 150, 250],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10, 15, 25],
                               'min_samples_split': [2, 5, 15, 30, 50, 70, 100],
                               'n_estimators': [100, 200, 300, 500, 700, 1000, 1300, 1500]
                               }
        return RandomForestClassifier(random_state=42, n_jobs=-1,
                                      class_weight=None)
