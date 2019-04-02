#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:12:31 2019

@author: shlomi
"""
# shell script to run gipsy:
#download clock and orbit:
# goa_prod_ftp.pl -d 2017-01-28 -s flinnR -hr

# content of run.sh:
#!/bin/csh -f  

#The following environment variables were active:
#setenv GOA /opt/goa-6.4

#You may need the following source command to set your PATH if you have changed it
#source  /opt/goa-6.4/rc_gipsy.csh
#( gd2p.pl \
#      # -e "-a 10 -LC -PC -F -ca2p_override -r 'JAVAD TRE_G3TH DELTA'" \
#      -e "-a 10 -LC -PC -F" \
#      # -orb_clk "flinnR_nf /root/hrmnforyuval/jplorb" \
#      -orb_clk "flinnR /home/shlomi/Desktop/DATA/Work_Files/PW_yuval" \
#      -no_del_shad \
#      -arp \
#      # -AntCal /root/work/2108/antcal/HRMN_antex.xyz \
#      -amb_res 1 \
#      -add_ocnld \
#      # -trop_map VMF1GRID \
#      -trop_map GPT2 \
#      #-vmf1dir /root/VMF1GRID \
#      -tides WahrK1 PolTid FreqDepLove OctTid \
#      -trop_z_rw 5e-8 \
#      -wetzgrad 5e-9 \
#      -w_elmin 7 \
#      -post_wind 2.5e-3 2.5e-5 \
#      -pb_min_slip 10e-5 \
#      -flag_qm \
#      -flag_brks_qm \
#      -edtpnt_max 10 \
#      -type s \
#      -r 300 \
#      -stacov \
#      -sta_info sta_info \
#      -p 4470.2581511  3084.5897721  3332.9527759  \
#      -i /home/shlomi/Desktop/DATA/Work_Files/PW_yuval/alon0280.17o \
#      -n ALON \
#      -d 2017-01-28 \
#      # -tdp_in /root/hrmnforyuval/9/trop/tdp \
# > gd2p.log ) |&  sed '/^Skipping namelist/d' > gd2p.err 
