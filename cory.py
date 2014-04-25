from smap.archiver.client import SmapClient
from smap.contrib import dtutil

from matplotlib import pyplot
from matplotlib import dates
import os
# make a client
c = SmapClient("http://new.openbms.org/backend")

# start and end values are Unix timestamps
t_start = "1-1-2012 0:00"
t_end = "1-1-2013 0:00"
start = 1000*dtutil.dt2ts(dtutil.strptime_tz(t_start, "%m-%d-%Y %H:%M"))
end   = 1000*dtutil.dt2ts(dtutil.strptime_tz(t_end, "%m-%d-%Y %H:%M"))

stnc = "select distinct Path where Metadata/SourceName='Cory Hall Dent Meters' and Path ~ '(?i)power$' and not Path ~ '.*ABC.*'"
pathlist = c.query(stnc) #the result is a list

pathlist = pathlist[275:] 
for path in pathlist:
	print "==========Fetching streams in path %s=========="%path
	for s in path:
		# fetch the metadata of path wanted
		tags = c.tags("Path='%s'"%path)

		# mkdir for each path
		path1 = "/Users/hdz_1989/Downloads/SDB/Cory"
		# folder = path
		# if not os.path.exists(path1+'/'+folder):
		# 	os.makedirs(path1+'/'+folder)

		# ft = open(path+'/'+folder+'/' + 'date.txt', 'w')
		# ft.write(t_start + ' ~ ' + t_end)
		# ft.close()

		for timeseries in tags:
			uuid = timeseries['uuid']
			# filename = timeseries['Path'].split('/')[-1]
			filename = timeseries['Path'][1:].replace('/','_')
			clause = "select data in (%.0f, %.0f) limit -1 where uuid = '%s'" \
			%(start, end, uuid)
			result = c.query(clause) #the result is a list
			d = result[0]['Readings']
			# f = open(path1+'/'+folder+'/'+filename + '.csv', 'w')
			f = open(path1+'/'+filename + '.csv', 'w')
			# f.writelines(["%.0f, %.2f\n"%(float(i[0])/1000, float(i[1])) for i in d])
			f.writelines(["%.2f\n"%(float(i[1])) for i in d])
			f.close()
