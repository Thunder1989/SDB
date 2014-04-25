from smap.archiver.client import SmapClient
from smap.contrib import dtutil

from matplotlib import pyplot
from matplotlib import dates
import os
import re
# make a client
c = SmapClient("http://new.openbms.org/backend")

# get tag list
tag = [i.strip('\n') for i in open('tagList.txt', 'r').readlines()]
# stnc = "select distinct"
# tag = c.query(stnc)

#prompt user to input
input = raw_input("keyword to query: ")

# regular expression search goes here...
# kw = input.split(' ')
# regex = re.compile("(?i).*(name).*")
# entry = [m.group(0) for l in tag for m in [regex.search(l)] if m]
# for e in entry:
for e in tag:
	# stnc = "select distinct Path where %s like '%%%s%%'"%(e, input)
	stnc = "select distinct Path where %s ~ '(?i).*%s.*'"%(e, input)
	print ">>>>>>>>>>querying thru entry %s>>>>>>>>>>"%e
	list = c.query(stnc) #the result is a list
	if list:
		print "the entry getting matches is", e
		print '\n'.join(i for i in list)
