#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 19:52:22 2021

@author: shlomi
"""

import click
from loguru import logger
from pathlib import Path

cwd=Path().cwd()



@click.command()
@click.option('--path', '-s', help='a full path to where the credential file is and the T02 report',
              type=click.Path(exists=True), default=cwd)


def main_program(*args, **kwargs):
    from pathlib import Path
    path = Path(kwargs['path'])
    email_alert_when_no_T02_files(path)
    return


def format_df_to_string_with_breaks(df):
    df = df.reset_index()
    li = []
    li.append([x for x in df.columns])
    for i, row in df.iterrows():
        li.append(row.tolist())
    big_str = '\n'.join('{}' for _ in range(len(li))).format(*li)
    return big_str


def email_alert_when_no_T02_files(path=cwd):
    """run this file daily and check last 6 hours of 'T02_file_count.csv',
    if all empty (0) then send an email to Yuval"""
    import pandas as pd
    df = pd.read_csv(path / 'T02_file_count.csv', index_col='dt')
    # if df.iloc[-6:]['no_files'].all():
    logger.warning('No files for the last 6 hours!')
    big_str = format_df_to_string_with_breaks(df)
    msg = 'No T02 files for the last 6 hours from AXIS, see report below!'
    msg ='\n'.join([msg,big_str])
    sender_email, passwd = read_gmail_creds(path)
    # rec_mails = ['shlomiziskin@gmail.com', 'yuvalr@ariel.ac.il', 'vlf.gps@gmail.com']
    rec_mails = ['shlomiziskin@gmail.com']
    subject = 'Geophysics1: AXIS TO2 lack of data'
    for rec_mail in rec_mails:
        send_gmail(sender_email, rec_mail, passwd, subject, msg)
# else:
#     logger.info('No total lack of AXIS T02 files detected in the last 6 hours.')
    return df


def read_gmail_creds(path=cwd, filename='.ariel.geophysics1.txt'):
    with open(path/filename) as f:
        mylist = f.read().splitlines()[0]
    email = mylist.split(',')[0]
    passwd = mylist.split(',')[1]
    return email, passwd


def send_gmail(sender_email, receiver_email, passwd, subject='', msg=''):
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    # sender_email = sender_mail  # Enter your address
    # receiver_email = rec_email  # Enter receiver address
    # password = input("Type your password and press enter: ")
    # message = """\
    # Subject: Hi there
    #
    # This message is sent from Python."""
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email
    part1 = MIMEText(msg, "plain")
    message.attach(part1)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, passwd)
        server.sendmail(sender_email, receiver_email, message.as_string())
    logger.info('email sent to {} from {}.'.format(receiver_email, sender_email))
    return


if __name__ == '__main__':
    main_program()
