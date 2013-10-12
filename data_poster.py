
# -*- coding: utf8 -*-
##########################################################
# load data -> JSON format -> Post to the server
##########################################################
import time
import datetime
import urllib
import json
import httplib
import uuid
# import simplejson
# import pickle

dataQueue = [100, 200]
uid = uuid.uuid1()

host = "new.openbms.org"
url = "/add"
key = "FRt4munyuxxOK8zmyYrNwiUI2w4HN2Ikpf0R"
port = 8079
headers = {"Content-Type":"application/json", "Connection":"Keep-Alive", "Referer":host} 

try:
    print 'now post the testing data ...'
    # set the needed parameter
    path = 'Dezhi/test'
    City = 'Berkeley'
    Building = 'Soda Hall'
    timestamp = int(time.time()*1000)
   
    print '---------------------------------------'
    print 'uuid: ' + str(uid)
    print 'City: ' + City
    print 'Building: ' + Building
    print 'timestamp: ' + str(timestamp)
    print 'data: ' + str(dataQueue)
    print '---------------------------------------\n'

    dictdata = {
        path:{
        'Metadata':{
            'SourceName': 'Post Testing Data',
            'Instrument': 'OfficalWeatherStation',
            'Location':{
                'City': City, 
                'Building': Building,
            }
        },
        'Properties':{
            'TimeZone':'US/Los_Angeles',
            'UnitofMeasure':'kW',
            'ReadingType':'int'
        },
        'Readings' : [],
        'uuid' : str(uid)
        }
    }
    for rd in dataQueue:
        dictdata[path]['Readings'].append([timestamp, rd])
        timestamp += 1000

    # print dicdata

    jdata = json.dumps(dictdata)

    # now post the data to the archiver server
    print jdata

    conn = httplib.HTTPConnection(host, port)
    conn.request(method="POST", url= url + '/' + key, body=jdata, headers=headers)
    response = conn.getresponse()
    print '#############'
    print "status:", response.status
    print '#############'

    #time.sleep(10)

    conn.close()
    
except Exception, e:
     print "Error: %s" % str(e)

# try to post the data to sMAP archiver


