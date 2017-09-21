from smap.archiver.client import SmapClient
from smap.contrib import dtutil

from matplotlib import pyplot
from matplotlib import dates
import os
# make a client
c = SmapClient("http://new.openbms.org/backend")

# start and end values are Unix timestamps
t_start = "6-12-2013 8:00"
t_end = "6-19-2013 8:00"
start = 1000*dtutil.dt2ts(dtutil.strptime_tz(t_start, "%m-%d-%Y %H:%M"))
end   = 1000*dtutil.dt2ts(dtutil.strptime_tz(t_end, "%m-%d-%Y %H:%M"))

stnc = "select distinct Metadata/Location/RoomNumber where Metadata/SourceName='KETI Motes'"
roomlist = c.query(stnc) #the result is a list

#roomlist = roomlist[16:]
#roomlist = ['621A','621B','621C','621D','621E']
for room in roomlist:
	print "==========Fetching streams in Room %s=========="%room
	stnc = "select Path where Metadata/Location/RoomNumber='%s' and not Path ~ '.*pir.*'" %room
	streams = c.query(stnc)
	if len(streams)>0:
		# print "----%d streams in Room %s----"%(len(streams), room)

		for s in streams:
			# fetch the metadata of path wanted
			tags = c.tags("Path='%s'"%s['Path'])

			# mkdir for each room
			path = "/Users/hdz_1989/Documents/Dropbox/SDB/KETI_tmp"
			folder = room
			if not os.path.exists(path+'/'+folder):
				os.makedirs(path+'/'+folder)

			# ft = open(path+'/'+folder+'/' + 'date.txt', 'w')
			# ft.write(t_start + ' ~ ' + t_end)
			# ft.close()

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
