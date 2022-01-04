#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 19:52:22 2021
Not finished script!
@author: shlomi
"""

import click
from loguru import logger
from pathlib import Path

upload_path = Path('/home/axis-gps')
historic_path = Path('/home/axis-gps/historic/Axis-Gps/T02')


@click.command()
@click.option('--months', '-m', help='the number of months old T02 to look for',
              type=click.Int, default=3)


def main_program(*args, **kwargs):
    months = kwargs['months']
    move_old_T02_to_historic_folder(months)
    return


def move_old_T02_to_historic_folder(months, upload_path=upload_path, historic_path=historic_path):
    import subprocess
    import pandas as pd
    import calendar
    datetime = pd.Timestamp.today() - pd.Timedelta(months*30, unit='d')
    month = datetime.month
    month = calendar.month_abbr[month]
    year = str(datetime.year)[2:]
    year_path = historic_path / 'RefData.{}'.format(year)
    # create the year path anyway:
    if not year_path:
        year_path.mkdir(parents=True, exist_ok=True)
    mnth_path = year_path/'Month.{}'.format(month)
    if not mnth_path.is_dir():
        # if the historic folder does not exists then move the uploaded
        # folder to the historic path
        # make sure that the uploded folder exists:
        curr_month_path = upload_path/'Month.{}'.format(month)
        if not curr_month_path.is_dir():
            logger.warning('{} was not found, either already moved or not existant, quitting...')
            return
        else:
            cmd = 'mv {} {}'.format(curr_month_path, year_path)
            subprocess.call(cmd, shell=True)


    return
if __name__ == '__main__':
    main_program()
