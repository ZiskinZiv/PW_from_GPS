#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:03:47 2022

@author: shlomi
"""
import ftplib
# import subprocess
from pathlib import Path
import click
from loguru import logger

savepath = Path('/mnt/DATA/Work_Files/PW_yuval/Etkes')
ftp = ftplib.FTP("ftp.etkes.com")
ftp.login("Uni", "Uni")

# ftp.dir()


@click.command()
@click.option('--savepath', '-s', help='a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins.',
              type=click.Path(exists=True), default=savepath)


def main_program(*args, **kwargs):
    from pathlib import Path
    savepath = Path(kwargs['savepath'])
    main_dir_loop(savepath)
    return


def getDirnamesFilenamesFromFTP(ftp, switch_to=None):
    import pandas as pd
    if switch_to is not None:
        ftp.cwd('{}'.format(switch_to))
        logger.info('CWD to {}.'.format(switch_to))
    ls = []
    ftp.retrlines('MLSD', ls.append)
    d=pd.DataFrame(ls)
    d = d[0].str.split(";", n = 4, expand = True)
    ddf = d[d[1]=='type=dir'][4].str.strip().to_frame('name')
    ddf['type'] = 'dir'
    fdf = d[d[1]=='type=file'][4].str.strip().to_frame('name')
    fdf['type'] = 'file'
    df = pd.concat([ddf, fdf], axis=0)
    return df


def selectDirsOrFiles(df, pick='dir'):
    df = df[df['type']==pick]
    return df['name'].unique()


def main_dir_loop(savepath):
    station_dirs = getDirnamesFilenamesFromFTP(ftp)
    station_dirs = selectDirsOrFiles(station_dirs, 'dir')
    for station in station_dirs:
        # print(ftp.pwd())
        years = getDirnamesFilenamesFromFTP(ftp, switch_to=station)
        years = selectDirsOrFiles(years, 'dir')
        station_savepath = savepath / station
        try:
            station_savepath.mkdir()
        except FileExistsError:
            logger.info(
                '{} already exists, using that folder.'.format(station_savepath))
        for year in years:
            station_year_savepath = station_savepath / year
            try:
                station_year_savepath.mkdir()
            except FileExistsError:
                logger.info(
                    '{} already exists, using that folder.'.format(station_year_savepath))
            # print(ftp.pwd())
            months = getDirnamesFilenamesFromFTP(ftp, switch_to=year)
            months = selectDirsOrFiles(months, 'dir')
            for month in months:
                # print(ftp.pwd())
                days = getDirnamesFilenamesFromFTP(ftp, switch_to=month)
                days = selectDirsOrFiles(days, 'dir')
                for day in days:
                    file = getDirnamesFilenamesFromFTP(ftp, switch_to=day)
                    try:
                        file = selectDirsOrFiles(file, 'file')[0]
                    except IndexError:
                        logger.warning('Files not Found in {}, skipping...'.format(ftp.pwd()))
                        ftp.cwd('..')
                        continue
                    local_filename = station_year_savepath/file
                    if local_filename.is_file():
                        logger.warning('{} already exists, skipping...'.format(local_filename))
                        ftp.cwd('..')
                        continue
                    with open(local_filename, "wb") as f:
                        ftp.retrbinary(f"RETR {file}", f.write)
                    # command = 'wget -q -P {}'.format(savepath)\
                    #     + ' ftp://ftp.etkes.com/{}'.format(ftp.pwd())\
                    #     + '/{}'.format(file)\
                    #     + ' --ftp-user="Uni" --ftp-password="Uni"'
                    logger.info('{} was copied to {}.'.format(file, station_year_savepath))
                    # subprocess.run(command, shell=True, check=True)
                    ftp.cwd('..')
                ftp.cwd('..')
            ftp.cwd('..')
        ftp.cwd('..')



if __name__ == '__main__':
    main_program()
# wget ftp://ftp.etkes.com//ARAV/2021/10/14/arav2870.rnx.zip --ftp-user="Uni" --ftp-password="Uni"
# command = 'wget -q -P {}'.format(savepath)\
        # + ' http://anonymous:shlomiziskin%40gmail.com@garner.ucsd.edu'\
        # + '/pub/rinex/{}/{}/{}'.format(year, dayofyear, filename)
# try:
    # subprocess.run(command, shell=True, check=True)
    # logger.info('Downloaded {} to {}.'.format(filename, savepath))
