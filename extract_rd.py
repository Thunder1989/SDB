"""
sample a snippet from the original data set and output to a new csv file

"""
import glob
import os
from smap.contrib import dtutil

dir = "/Users/hdz_1989/Downloads/SDB/Todai"
list = glob.glob(dir + '/*.dat')
data_length = 7
for i in list:
	raw_rd = [line.strip('\n') for line in open(i, 'r').readlines()]
	[filepath, filename] = os.path.split(i)
	(name, extension) = os.path.splitext(filename)
	f = open(dir + '/sample/' + name + '.csv', 'w')
	#the interval btw timestamp in the data file is 20s, and read data of data_length days 
	f.writelines(["%s\n" %(l.split()[1]) for l in raw_rd])
	f.close()

