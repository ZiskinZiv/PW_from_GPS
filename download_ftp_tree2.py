import ftplib
from ftptool import FTPHost
ftp = ftplib.FTP("garner.ucsd.edu", 'anonymous', "shlomiziskin@gmail.com")
a_host=FTPHost(ftp)
a_host.mirror_to_local('/pub/solutions/gipsy/trop', '/home/shlomi/Desktop/DATA/Work_Files/PW_yuval/garner/trop')

