import urllib2
import numpy as np
import matplotlib.pyplot as plt
import csv
from smap.archiver.client import SmapClient
from smap.contrib import dtutil

def get_temp(start, end):
	print "================================================"
	print "retrieving temp from %s to %s..." %(start, end)
	start = dtutil.dt2ts(dtutil.strptime_tz("%s" %start, "%m-%d-%Y %H:%M")) * 1000
	end = dtutil.dt2ts(dtutil.strptime_tz("%s" %end, "%m-%d-%Y %H:%M")) * 1000
	clause = "select data in (%f, %f) limit -1 streamlimit 10000 \
	where uuid = '395005af-a42c-587f-9c46-860f3061ef0d'" %(start, end)
	# make a client
	c = SmapClient("http://new.openbms.org/backend")
	result = c.query(clause) #the result is a list
	reading = result[0]['Readings']

	# output readings to file
	fw = open('temp.csv', 'w')
	fw.writelines("%s,%s\n" %(rd[0],rd[1]) for rd in reading)
	fw.close()
	data = np.asarray(reading)
	return data

def plot_temp(data):
	# plot the readings
	# data = np.asarray(reading)
	# data = data[:,1]
	plt.figure()
	plt.title("Outside Air Temperature")
	plt.plot(data)
	plt.show()

