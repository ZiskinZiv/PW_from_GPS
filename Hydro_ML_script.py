#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 08:31:03 2020

@author: shlomi
"""
import os, sys, warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::RuntimeWarning') # Also affect subprocesses

def check_station_name(name):
    # import os
    if isinstance(name, list):
        name = [str(x).lower() for x in name]
        for nm in name:
            if len(nm) != 4:
                raise argparse.ArgumentTypeError('{} should be 4 letters...'.format(nm))
        return name
    else:
        name = str(name).lower()
        if len(name) != 4:
            raise argparse.ArgumentTypeError(name + ' should be 4 letters...')
        return name


#def check_loopover():
#    return 

#def check_hydro_id(num):
#    return

#def check_features(feat):
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
    from hydro_procedures import ML_main_procedure
    X, y = produce_X_y(pw_station=args.pw_station, hs_id=args.hydro_id,
                       pressure_station='bet-dagan', window=25, max_flow=0, neg_pos_ratio=1)
    scorers = ['roc_auc', 'f1', 'accuracy']
    splits = [2, 3, 4]
    features = ['pwv', 'pressure', ['pwv', 'pressure']]
    if args.test_size is not None:
        test_size = args.test_size
    else:
        test_size = 0.2
    if args.savepath is not None:
        savepath = args.savepath
    else:
        savepath = hydro_path
    if args.model is not None:
        model_name = args.model
        for scorer in scorers:
            for n_s in splits:
                for feature in features:
                    logger.info(
                        'Running {} model with {} test scorer and {} nsplits, features={}'.format(
                            model_name, scorer, n_s, feature))
                    model = ML_main_procedure(
                        X,
                        y,
                        model_name=model_name,
                        features=feature,
                        n_splits=n_s,
                        best_score=scorer,
                        val_size=None,
                        test_size=test_size,
                        savepath=savepath,
                        plot=False)
    else:
        logger.info('Running with all three models:')
        models = ['SVC', 'RF', 'MLP']
        for model_name in models:
            for scorer in scorers:
                for n_s in splits:
                    for feature in features:
                        logger.info(
                            'Running {} model with {} test scorer and {} nsplits, features={}'.format(
                                model_name, scorer, n_s, feature))
                        model = ML_main_procedure(
                            X,
                            y,
                            model_name=model_name,
                            features=feature,
                            n_splits=n_s,
                            best_score=scorer,
                            val_size=None,
                            test_size=test_size,
                            savepath=savepath,
                            plot=False)


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from aux_gps import configure_logger
    from PW_paths import work_yuval
    hydro_path = work_yuval / 'hydro'
    logger = configure_logger('Hydro_ML')
    savepath = Path(hydro_path)
    parser = argparse.ArgumentParser(description='a command line tool for running the ML models tuning for hydro-PWV.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--pw_station', help="GNSS 4 letter station", type=check_station_name)
    required.add_argument('--hydro_id', help="5 integer hydro station", type=int)# check_hydro_id)
#    optional.add_argument('--loop_over', help='select which params to loop over',
#                          type=check_loopover, nargs='+')
    optional.add_argument('--savepath', help="a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins", type=check_path)
    optional.add_argument('--test_size', help='how much to leave for test [0-1]', type=float)
#    optional.add_argument('--nsplits', help='select number of splits for HP tuning.', type=int)
    optional.add_argument('--model', help='select ML model.', choices=['SVC', 'MLP', 'RF'])
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
    main_hydro_ML(args)