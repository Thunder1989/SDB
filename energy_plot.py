"""
sum up the feeds of energy consumption from a building and apply EMD to the sum
@author: Dezhi
"""
import urllib2
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from smap.archiver.client import SmapClient
from smap.contrib import dtutil
from EMDpython import EMD
from temp import get_temp

# make a client
c = SmapClient("http://new.openbms.org/backend")

# get the components for query
bldg_list = [i.strip('\n') for i in open('bldg_list.txt', 'r').readlines()]
index = range(1,len(bldg_list))
bldg_dict = dict(zip(index, bldg_list))
print "================================================"
print "\n".join(["%s-%s" %(k,v) for k,v in bldg_dict.items()])

num = raw_input("choose a # from above to query: ")
bldg = bldg_dict[int(num)]
# start = raw_input("start time (\"%m-%d-%Y %H:%M\" or "-d" for default): ")
# end = raw_input("end time (\"%m-%d-%Y %H:%M\") or "-d" for default: ")
start = "6-23-2013 00:00"
end = "6-30-2013 23:59"
# get the outside air temperature during the period specified
# get_temp(start, end)

if start == '-d':
	print "computing the IMFs in %s from 7 days backwards till now..." %bldg
else:
	print "computing the IMFs in %s from %s to %s..." %(bldg, start, end)
if end == '-d':
	end = int(time.time()*1000)
else:
	end = dtutil.dt2ts(dtutil.strptime_tz("%s" %end, "%m-%d-%Y %H:%M")) * 1000
if start == '-d':
	start = end - 7*24*60*60*1000
else:
	start = dtutil.dt2ts(dtutil.strptime_tz("%s" %start, "%m-%d-%Y %H:%M")) * 1000
applySum = "apply nansum(axis=1) < paste < window(first, field='minute', width=15) < units to data in (%f, %f) limit -1 streamlimit 10000 \
	where (Metadata/Extra/System = 'total' or Metadata/Extra/System = 'electric') \
	and (Properties/UnitofMeasure = 'kW' or Properties/UnitofMeasure = 'Watts' or Properties/UnitofMeasure = 'W') \
	and Metadata/Location/Building like '%s%%' and not Metadata/Extra/Operator like 'sum%%' \
	and not Path like '%%demand' and not Path like '/Cory_Hall/Electric_5A7/ABC/real_power' and not Path like '/Cory_Hall/Electric_5B7/ABC/real_power'" \
	%(start, end, bldg)

result = c.query(applySum) #the result is a list
reading = result[0]['Readings']

# output readings to file
fw = open('energy.csv', 'w')
fw.writelines("%s,%s\n" %(rd[0],rd[1]) for rd in reading)
fw.close()

# adujst to data to an array of even length
data = np.asarray(reading)
data = data[:,1]
if len(data)%2==0:
	data = data
else:
	data = data[:len(data)-1]
# print len(data)

# EMD
res = EMD.emd(data)
col = res.shape[1]
print "got ", col, "IMFs"
# output IMFs to file
with file('imfs.csv', 'w') as outfile:
	fw = csv.writer(outfile, delimiter=',')
	fw.writerows(res[i,:] for i in range(1,res.shape[0]))
outfile.close()

#plot figures
plt.figure()
plt.title("Power Consumption of %s (kWh)"%bldg)
for i in range(col):
	plt.subplot(col+1, 1, i+1)
	# tmp = np.zeros([len(res),1])
	# for t in range(i+1):
	# 	tmp[:,0] += res[:,t]
	# plt.plot(tmp, 'b-', data, 'r--')
	plt.plot(res[:,i])

plt.subplot(col+1,1,col+1)
plt.plot(data)
plt.show()
