"""
test script

"""

import glob
import os
from smap.contrib import dtutil

dir = "/Users/hdz_1989/Downloads/SDB/Todai/"
list = glob.glob(dir + '/*.dat')
f = open(dir + 'sample/' + 'ts_checking.txt', 'w')

for i in list:
	fp = open(i, 'r')
	j = 0
	while j<3:
		rd = fp.readline()
		ts = float(rd.strip('\n').split()[0])
		time = dtutil.strftime_tz(dtutil.ts2dt(ts), "%m-%d-%Y %H:%M:%S")
		f.write("%s, " %time)
		j += 1
	rd = fp.readline()
	ts = float(rd.strip('\n').split()[0])
	time = dtutil.strftime_tz(dtutil.ts2dt(ts), "%m-%d-%Y %H:%M:%S")
	f.write("%s\n" %time)
	fp.close()

f.close()
