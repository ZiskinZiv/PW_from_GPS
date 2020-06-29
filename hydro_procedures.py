#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:08:43 2019

@author: ziskin
"""

from PW_paths import work_yuval
hydro_path = work_yuval / 'hydro'
gis_path = work_yuval / 'gis'

# TODO: scan for seasons in the tide events and remove summer
# TODO: prepare pw_hourly_50 so no homogenized data will enter
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
    fg = xr.plot.FacetGrid(da, col='model', col_wrap=4, sharex=False, sharey=False)
    for model_str, ax in zip(da['model'].values, fg.axes.flatten()):
        model = model_str.split('-')[0]
        method = model_str.split('-')[-1]
        if model == method:
            method = None
        ax = scikit_decompose(X, y, model=model, n=n, method=method, ax=ax)
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
        X2.flat[::X.shape[1] + 1] += 0.01
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
                              color='b',marker='s', alpha=0.3,
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
            color='r',marker='x',
            label='0',
            s=50)
    elif X_decomp.shape[1] == 2:
        if ax is not None:
            df_1.plot.scatter(ax=ax,
                              x='{}_1'.format(model),
                              y='{}_2'.format(model),
                              color='b',marker='s', alpha=0.3,
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
    if not cv:
        clf = SVC(kernel='rbf')
        clf = LinearDiscriminantAnalysis()
        # cv = StratifiedKFold(2, shuffle=True)
        cv = KFold(2, shuffle=True)
        n_classes = 2
        score, permutation_scores, pvalue = permutation_test_score(
            clf, X, y, scoring="f1", cv=cv, n_permutations=1000, n_jobs=1)

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
            X, y, test_size=0.3, shuffle=True, random_state=42)
        param_grid = {
            'C': [
                0.1, 1, 10, 100], 'gamma': [
                1, 0.1, 0.01, 0.001], 'kernel': [
                'rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(X_test)
        print(confusion_matrix(y_test, grid_predictions))
        print(classification_report(y_test, grid_predictions))
    return


def scikit_fit_predict(X, y, seed=42, plot=True):
    # check permutations with scikit learn
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.metrics import f1_score
    from sklearn.metrics import plot_roc_curve
    from sklearn.svm import SVC
    from scipy import interp
    from sklearn.metrics import auc
    import numpy as np
    import matplotlib.pyplot as plt
    X_tt, X_test, y_tt, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=seed)
    # clf = SVC(gamma='auto')
    clf = LinearDiscriminantAnalysis()
    # clf = QuadraticDiscriminantAnalysis()
    scores = []
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(100):
        clf = LinearDiscriminantAnalysis()
        X_train, X_val, y_train, y_val = train_test_split(
            X_tt, y_tt, shuffle=True, test_size=0.5, random_state=i)
        clf.fit(X_train, y_train)
        viz = plot_roc_curve(clf, X_val, y_val,
                             name='ROC run {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        y_pred = clf.predict(X_val)
        # scores.append(clf.score(X_val, y_val))
        scores.append(f1_score(y_val, y_pred))
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
    if plot:
        plt.figure()
        plt.hist(scores, bins=15, edgecolor='k')
    return scores
    # clf.fit(X,y)


def produce_X_y(station='drag', hs_id=48125, lag=25, anoms=True,
                neg_pos_ratio=2,
                path=work_yuval, hydro_path=hydro_path, with_ends=False, seed=42,
                verbose=True):
    import pandas as pd
    import numpy as np
    df = preprocess_hydro_pw(
        pw_station=station,
        hs_id=hs_id,
        path=path,
        hydro_path=hydro_path,
        with_ends=with_ends, anoms=anoms)
    # first produce all the positives:
    # get the tides datetimes:
    y_pos = df[df['tides'] == 1]['tides']
    # get the datetimes of 24 hours before tide event (not inclusive):
    y_lag_pos = y_pos.index - pd.Timedelta(lag, unit='H')
    masks = [(df.index > start) & (df.index < end)
             for start, end in zip(y_lag_pos, y_pos.index)]
    # also drop event if less than 24 hour before available:
    pw_pos_list = []
    ind = []
    bad_ind = []
    for i, tide in enumerate(masks):
        if len(df['tides'][tide]) == (lag - 1):
            pw_pos_list.append(df[station][tide])
            ind.append(i)
        else:
            bad_ind.append(i)
    # get the indices of the dropped events:
    # ind = [x[0] for x in pw_pos_list]
    if bad_ind:
        if verbose:
            print('{} are without full 24 hours before record.'.format(
                ','.join([x for x in df.iloc[bad_ind].index.strftime('%Y-%m-%d:%H:00:00')])))
    # drop the events in y so len(y) == in each x from tides_list:
    y_pos_arr = y_pos.iloc[ind].values
    # now get the negative y's with neg_pos_ratio (set to 1 if the same pos=neg):
    y_neg_arr = np.zeros(y_pos_arr.shape[0] * neg_pos_ratio)
    cnt = 0
    pw_neg_list = []
    np.random.seed(seed)
    while cnt < len(y_neg_arr):
        # get a random date from df:
        r = np.random.randint(low=0, high=len(df))
        # slice -24 to 24 range with t=0 being the random date:
        sliced = df.iloc[r - lag:r + lag]
        # if tides inside this date range, continue:
        if y_pos.iloc[ind].index in sliced.index:
            if verbose:
                print('found positive tide in randomly sliced 48 window')
            continue
        # now if no 24 items exist, also continue:
        negative = df.iloc[r - lag:r - 1][station]
        if len(negative) != (lag-1):
            if verbose:
                print('didnt find full {} hours sliced negative'.format(lag-1))
            continue
        # else, append to pw_neg_list and increase cnt
        pw_neg_list.append(negative)
        cnt += 1
    # lastly, assemble for X, y using np.columnstack:
    y = np.concatenate([y_pos_arr, y_neg_arr])
    X = np.stack([[x.values for x in pw_pos_list] + 
                        [x.values for x in pw_neg_list]])
    X = X.squeeze()
    return X, y


def preprocess_hydro_pw(pw_station='drag', hs_id=48125, path=work_yuval,
                        anoms=True, hydro_path=hydro_path, with_ends=False):
    import xarray as xr
    import pandas as pd
    import numpy as np
    # first load tides data:
    all_tides = xr.open_dataset(hydro_path / 'hydro_tides.nc')
    # get all tides for specific station without nans:
    sta_slice = [x for x in all_tides.data_vars if str(hs_id) in x]
    if not sta_slice:
        raise KeyError('hydro station {} not found in database'.format(hs_id))
    tides = all_tides[sta_slice].dropna('tide_start')
    tide_starts = tides['tide_start'].where(
        ~tides.isnull()).dropna('tide_start')['tide_start']
    tide_ends = tides['TS_{}_tide_end'.format(hs_id)].where(
        ~tides.isnull()).dropna('tide_start')['TS_{}_tide_end'.format(hs_id)]
    # round all tide_starts to hourly:
    ts = tide_starts.dt.round('1H')
    ts_end = tide_ends.dt.round('1H')
    time_dt = pd.date_range(
        start=ts.min().values,
        end=ts_end.max().values,
        freq='1H')
    df = pd.DataFrame(data=np.zeros(time_dt.shape), index=time_dt)
    df.loc[ts.values, :] = 1
    if with_ends:
        df.loc[ts_end.values, :] = 2
    df.columns = ['tides']
    # now load pw:
    if anoms:
        pw = xr.load_dataset(path / 'GNSS_PW_anom_hourly_50_hour_dayofyear.nc')[pw_station]
    else:
        pw = xr.load_dataset(path / 'GNSS_PW_hourly_thresh_50.nc')[pw_station]
    pw_df = pw.dropna('time').to_dataframe()
    # now align the both dataframes:
    pw_df['tides'] = df['tides']
    pw_df['tides'] = pw_df['tides'].fillna(0)
    return pw_df


def loop_over_gnss_hydro_and_aggregate(sel_hydro, pw_anom=False,
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
    if (hydro_path / filename).is_file():
        print('loading {}...'.format(filename))
        ds = xr.load_dataset(hydro_path / filename)
    else:
        if pw_anom:
            file = path_glob(work_yuval, 'GNSS_PW_anom_*.nc')[-1]
            gnss_pw = xr.open_dataset(file)
        else:
            gnss_pw = xr.open_dataset(work_yuval / 'GNSS_PW_thresh_50_homogenized.nc')
        just_pw = [x for x in gnss_pw.data_vars if '_error' not in x]
        gnss_pw = gnss_pw[just_pw]
        da_list = []
        for gnss_sta in just_pw:
            print('proccessing station {}'.format(gnss_sta))
            sliced = sel_hydro[~sel_hydro[gnss_sta].isnull()]
            hydro_ids = [x for x in sliced.id.values]
            if not hydro_ids:
                print(
                    'skipping {} station since no close hydro stations...'.format(gnss_sta))
                continue
            else:
                try:
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
        ax.set_xlabel('Days before tide event')
        ax.set_ylabel('PW [mm]')
        ax.grid()
        hstations = [ds[x].attrs['hydro_stations'] for x in ds.data_vars]
        events = [ds[x].attrs['total_events'] for x in ds.data_vars]
        fmt = list(zip(names, hstations, events))
        ax.legend(['{} with {} stations ({} total events)'.format(x, y, z)
                   for x, y, z in fmt])
        if pw_anom:
            title = 'Mean PW anomalies for tide stations near all GNSS stations'
        else:
            title = 'Mean PW for tide stations near all GNSS stations'
        if max_flow_thresh is not None:
            title += ' (max_flow > {} m^3/sec)'.format(max_flow_thresh)
        ax.set_title(title)
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
        elif freq == 'H':
            points = int(ndays) * 24
            points_forward = int(ndays_forward) * 24
        lag = pd.timedelta_range(end=0, periods=points, freq=freq)
        forward_lag = pd.timedelta_range(start=0, periods=points_forward, freq=freq)
        lag = lag.union(forward_lag)
        time_arr = pd.to_datetime(pw_da.time.values)
        tide_start = pd.to_datetime(tide_start).round(freq)
        ts_loc = time_arr.get_loc(tide_start)
        # days = pd.Timedelta(ndays, unit='D')
        # time_slice = [tide_start - days, tide_start]
        # pw = pw_da.sel(time=slice(*time_slice))
        pw = pw_da.isel(time=slice(ts_loc - points, ts_loc + points_forward -1))
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
        pw, lag = get_n_days_pw_hydro_one_event(pw_da, ts, ndays=ndays, ndays_forward=ndays_forward)
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
        plt.suptitle('hydro-tide stations within {} km of a GNSS station'.format(radius), fontsize=14)
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
        df['tide_end'] = pd.to_datetime(
            df['tide_end_date'], dayfirst=True) + pd.to_timedelta(
            df['tide_end_hour'].add(':00'),
            unit='m',
            errors='coerce')
        df['tide_max'] = pd.to_datetime(
            df['tide_max_date'], dayfirst=True) + pd.to_timedelta(
            df['tide_max_hour'].add(':00'),
            unit='m',
            errors='coerce')
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
    filename = 'hydro_tides.nc'
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done!')
    return ds


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
        df['time'] = pd.to_datetime(df['time'], dayfirst=True)
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
    filename = 'hydro_graphs.nc'
    print('saving {} to {}'.format(filename, path))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(path / filename, 'w', encoding=encoding)
    print('Done!')
    return ds
