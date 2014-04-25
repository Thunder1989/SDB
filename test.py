"""
sample a snippet from the original data set and output to a new csv file

"""
import glob
import os
from smap.contrib import dtutil

path = "/Users/hdz_1989/Downloads/SDB/102B1"
folder = 'sample'
list = glob.glob(dir + '/*.txt')
data_length = 7
for i in list:
	raw_rd = [line.strip('\n') for line in open(i, 'r').readlines()]
	[filepath, filename] = os.path.split(i)
	(name, extension) = os.path.splitext(filename)
	f = open(path+'/'+folder+name+'.csv', 'w')
	#the interval btw timestamp in the data file is 20s, and read data of data_length days
	tmp = 0.0
	for k in range(60*60*24*data_length/60):
		# ts = float(raw_rd[k].split()[0])
		# time = dtutil.strftime_tz(dtutil.ts2dt(ts), "%m-%d-%Y %H:%M:%S")
		rd = raw_rd[k].split()[2]
		if rd != '?':
			f.write("%.2f\n" %float(rd))
			tmp = float(rd)
		else:
			f.write("%.2f\n" %tmp)
	f.close()
