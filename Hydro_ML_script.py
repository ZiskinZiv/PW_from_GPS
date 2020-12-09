#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 08:31:03 2020

@author: shlomi
"""
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = (
        'ignore::UserWarning,ignore::RuntimeWarning')  # Also affect subprocesses


def check_station_name(name):
    # import os
    if isinstance(name, list):
        name = [str(x).lower() for x in name]
        for nm in name:
            if len(nm) != 4:
                raise argparse.ArgumentTypeError(
                    '{} should be 4 letters...'.format(nm))
        return name
    else:
        name = str(name).lower()
        if len(name) != 4:
            raise argparse.ArgumentTypeError(name + ' should be 4 letters...')
        return name


# def check_loopover():
#    return

# def check_hydro_id(num):
#    return

# def check_features(feat):
#    return

def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def main_hydro_ML(args):
    from hydro_procedures import produce_X_y
    from hydro_procedures import produce_X_y_from_list
    from hydro_procedures import nested_cross_validation_procedure
    from aux_gps import get_all_possible_combinations_from_list
    if args.verbose is None:
        verbose=0
    else:
        verbose = args.verbose
    if args.n_jobs is None:
        n_jobs = -1
    else:
        n_jobs = args.n_jobs
    if args.max_flow is None:
        max_flow = 0
    else:
        max_flow = args.max_flow
    if args.neg_pos_ratio is not None:
        neg_pos_ratio = args.neg_pos_ratio
    else:
        neg_pos_ratio = 1
    logger.info('max flow {} threshold m^3/sec selected.'.format(max_flow))
    logger.info('negative to positive ratio {} selected.'.format(neg_pos_ratio))
    if len(args.pw_station) > 1:
        X, y = produce_X_y_from_list(pw_stations=args.pw_station,
                                     hs_ids=args.hydro_id,
                                     pressure_station='bet-dagan', window=25,
                                     max_flow=max_flow,
                                     neg_pos_ratio=neg_pos_ratio,
                                     concat_Xy=True)
    else:
        X, y = produce_X_y(pw_station=args.pw_station[0], hs_id=args.hydro_id[0],
                           pressure_station='bet-dagan', window=25,
                           max_flow=max_flow,
                           neg_pos_ratio=neg_pos_ratio)
    scorers = ['roc_auc', 'f1', 'accuracy']
#    splits = [2, 3, 4]
    f = ['pwv', 'pressure', 'doy']
    features = get_all_possible_combinations_from_list(
        f, reduce_single_list=True)
    if args.inner_splits is not None:
        inner_splits = args.inner_splits
    else:
        inner_splits = 2
    if args.outer_splits is not None:
        outer_splits = args.outer_splits
    else:
        outer_splits = 4
#    if args.test_size is not None:
#        test_size = args.test_size
#    else:
#        test_size = 0.2
    if args.savepath is not None:
        savepath = args.savepath
    else:
        savepath = hydro_path
#    if args.model is not None:
    model_name = args.model
    for scorer in scorers:
        if model_name != 'RF':
            for feature in features:
                logger.info(
                    'Running {} model with {} test scorer and {},{} (inner, outer) nsplits, features={}'.format(
                        model_name, scorer, inner_splits, outer_splits, feature))
                model = nested_cross_validation_procedure(
                    X,
                    y,
                    model_name=model_name,
                    features=feature,
                    inner_splits=inner_splits,
                    outer_splits=outer_splits,
                    refit_scorer=scorer,
                    verbose=verbose,
                    diagnostic=False,
                    savepath=savepath, n_jobs=n_jobs)
        else:
            logger.info(
                    'Running {} model with {} test scorer and {},{} (inner, outer) nsplits, features={}'.format(
                        model_name, scorer, inner_splits, outer_splits, f))
            model = nested_cross_validation_procedure(
                X,
                y,
                model_name=model_name,
                features=f,
                inner_splits=inner_splits,
                outer_splits=outer_splits,
                refit_scorer=scorer,
                verbose=verbose,
                diagnostic=False,
                savepath=savepath, n_jobs=n_jobs)
#    else:
#        logger.info('Running with all three models:')
#        models = ['SVC', 'RF', 'MLP']
#        for model_name in models:
#            for scorer in scorers:
#                for feature in features:
#                    logger.info(
#                        'Running {} model with {} test scorer and {},{} (inner, outer) nsplits, features={}'.format(
#                            model_name, scorer, inner_splits, outer_splits, feature))
#                    model = nested_cross_validation_procedure(
#                        X,
#                        y,
#                        model_name=model_name,
#                        features=feature,
#                        inner_splits=inner_splits,
#                        outer_splits=outer_splits,
#                        refit_scorer=scorer,
#                        verbose=0,
#                        diagnostic=False,
#                        savepath=savepath)


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from aux_gps import configure_logger
    from PW_paths import work_yuval
    hydro_path = work_yuval / 'hydro'
    logger = configure_logger('Hydro_ML')
    savepath = Path(hydro_path)
    parser = argparse.ArgumentParser(
        description='a command line tool for running the ML models tuning for hydro-PWV.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument(
        '--pw_station',
        help="GNSS 4 letter station", nargs='+',
        type=check_station_name)
    required.add_argument(
        '--hydro_id',
        help="5 integer hydro station", nargs='+',
        type=int)  # check_hydro_id)
#    optional.add_argument('--loop_over', help='select which params to loop over',
#                          type=check_loopover, nargs='+')
    optional.add_argument(
        '--savepath',
        help="a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins",
        type=check_path)
    optional.add_argument(
        '--outer_splits',
        help='how many splits for the outer nested loop',
        type=int)
    optional.add_argument(
        '--inner_splits',
        help='how many splits for the inner nested loop',
        type=int)
    optional.add_argument(
        '--max_flow',
        help='slice the hydro events for minimum max flow',
        type=float)
    optional.add_argument(
        '--neg_pos_ratio',
        help='negative to positive events ratio',
        type=int)
    optional.add_argument(
        '--n_jobs',
        help='number of CPU threads to do gridsearch and cross-validate',
        type=int)
    optional.add_argument(
        '--verbose',
        help='verbosity 0, 1, 2',
        type=int)
#    optional.add_argument('--nsplits', help='select number of splits for HP tuning.', type=int)
    required.add_argument(
        '--model',
        help='select ML model.',
        choices=[
            'SVC',
            'MLP',
            'RF'])
#    optional.add_argument('--feature', help='select features for ML', type=check_features, nargs='+')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.pw_station is None:
        print('pw_station is a required argument, run with -h...')
        sys.exit()
    if args.hydro_id is None:
        print('hydro_id is a required argument, run with -h...')
        sys.exit()
    if args.model is None:
        print('model is a required argument, run with -h...')
        sys.exit()
    logger.info('Running pwv station {} with hydro station {}.'.format(args.pw_station, args.hydro_id))
    main_hydro_ML(args)
