from smap.archiver.client import SmapClient
from smap.contrib import dtutil
from datetime import datetime
import os

# make a client
c = SmapClient("http://new.openbms.org/backend")

start =  datetime.now()
f = open('uuid','r')
uuid = [i.strip('\n') for i in open('uuid', 'r').readlines()]
# ts = range(len(uuid))
# table = dict(zip(uuid,ts))
f.close()

# for u in uuid:
stnc = "select data before now streamlimit -1 where Metadata/SourceName ~ '.*'"
out = c.query(stnc)

for rd in out:
	table[rd['uuid']] = (int)(rd['Readings'][0][0]/1000)

f = open('freshTable','w')
f.writelines("%s:%s\n"%(k,v) for k,v in table.items())
f.close()

print "Time elapsed:", str(datetime.now()-start)