from smap.archiver.client import SmapClient
from smap.contrib import dtutil

from matplotlib import pyplot
from matplotlib import dates
import os
# make a client
c = SmapClient("http://new.openbms.org/backend")

# start and end values are Unix timestamps
t_start = "11-4-2012 00:00"
t_end = "12-1-2012 23:59"
start = 1000*dtutil.dt2ts(dtutil.strptime_tz(t_start, "%m-%d-%Y %H:%M"))
end   = 1000*dtutil.dt2ts(dtutil.strptime_tz(t_end, "%m-%d-%Y %H:%M"))

# download the metadata of path wanted
tags = c.tags("Path like '/sdh_co2/13%'")

# make a dict mapping uuids to data vectors
path = "/Users/hdz_1989/Downloads/SDB/SDH"
folder = tags[0]['Path'].split('/')[-2]
if not os.path.exists(path+'/'+folder):
	os.makedirs(path+'/'+folder)

ft = open(path+'/'+folder+'/' + 'date.txt', 'w')
ft.write(t_start + ' ~ ' + t_end)
ft.close()

for timeseries in tags:
	uuid = timeseries['uuid']
	filename = timeseries['Path'].split('/')[-1]
	clause = "select data in (%.0f, %.0f) limit -1 where uuid = '%s'" \
	%(start, end, uuid)
	result = c.query(clause) #the result is a list
	d = result[0]['Readings']
	f = open(path+'/'+folder+'/'+filename + '.csv', 'w')
	f.writelines(["%.0f, %.2f\n"%(float(i[0])/1000, float(i[1])) for i in d])
	# f.writelines(["%.2f\n"%(float(i[1])) for i in d])
	f.close()

# # plot all the data
# for timeseries in tags:
#   d = data_map[timeseries['uuid']]
#   # since we have the tags, we can add some metadata
#   label = "%s (%s)" % (timeseries['Path'],
#                        timeseries['Properties/UnitofMeasure'])
#   # we can plot all of the series in their appropriate time zones
#   pyplot.plot_date(dates.epoch2num(d[:, 0] / 1000), d[:, 1], '-', 
#                    label=label,
#                    tz=timeseries['Properties/Timezone'])

# pyplot.legend(loc="upper center")
# pyplot.show()
