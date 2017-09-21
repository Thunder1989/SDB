from search import *
from smap.archiver.client import SmapClient
from smap.contrib import dtutil
from sklearn.tree import DecisionTreeClassifier as DT

import numpy as np
import os

def get_feature(data):
	mean = np.mean(data)
	median = np.median(data)
	std = np.std(data)
	q1 = np.percentile(data,25)	
	q3 = np.percentile(data,75)
	vector = [mean, median, std, q1, q3]
	return vector

c = SmapClient("http://new.openbms.org/backend")
lines = open('MetadataDump').readlines()
sdh_path = [i.strip('\n') for i in open('sdh_path','r').readlines()]

search_path = []
search_res = []
res = keywordSearch(lines, "sdh temp", 2000)
water = keywordSearch(lines, "sdh hw temp", 1000)
stp = keywordSearch(lines, "sdh temp stp", 1000)
search = keywordSearch(lines, "sdh room temp", 2000)
add = keywordSearch(lines, "sdh rm temp", 1000)
for i in water:
	res.remove(i)
for i in stp:
	res.remove(i)
for r in res:
	search_path.append(r['Path'])
for i in add:
	search.append(i)
for i in search:
	search_res.append(i['Path'])

train_start = "6-12-2013 8:00"
train_end = "6-12-2013 8:10"
start1 = 1000*dtutil.dt2ts(dtutil.strptime_tz(train_start, "%m-%d-%Y %H:%M"))
end1 = 1000*dtutil.dt2ts(dtutil.strptime_tz(train_end, "%m-%d-%Y %H:%M"))

test_start = "6-13-2013 10:10"
test_end = "6-13-2013 10:20"
start2 = 1000*dtutil.dt2ts(dtutil.strptime_tz(test_start, "%m-%d-%Y %H:%M"))
end2 = 1000*dtutil.dt2ts(dtutil.strptime_tz(test_end, "%m-%d-%Y %H:%M"))

anomaly_path1 = []
anomaly_path2 = []
train_data = []
test_data = []
search_train = []
# anomaly_train = []
search_test = []
# anomaly_test = []
count = 0
valid = []

for i in sdh_path:
	count += 1
	if count%50==0:
		print '%.2f%% done...'%(float(count)/len(sdh_path)*100)
	q = "select data in (%.0f, %.0f) limit -1 where Path = '%s'" \
						%(start1, end1, i)
	rd = c.query(q)
	if rd:
		if rd[0]['Readings']:
			if rd[0]['Readings'][0][1] and len(rd[0]['Readings'])>1:
				flag = 0
				data = rd[0]['Readings']
				train_data.append(get_feature(data))
				
				if i in search_path:
					# for d in data:
					# 	if d[1]>72 or d[1]<60:
					# 		anomaly_path.append(i)
					# 		count1 += 1
					# 		flag = 1;
					# 		break;
				
					search_train.append(1)
					# anomaly_train.append(flag)
			
				else:
					search_train.append(0)
					# anomaly_train.append(0)

	q = "select data in (%.0f, %.0f) limit -1 where Path = '%s'" \
					%(start2, end2, i)
	rd = c.query(q)
	if rd:
		if rd[0]['Readings']:
			if rd[0]['Readings'][0][1] and len(rd[0]['Readings'])>1:
				flag = 0;
				data = rd[0]['Readings']
				test_data.append(get_feature(data))	
				valid.append(i)
			
				if i in search_path:
					# for d in data:
					# 	if d[1]>72 or d[1]<60:
					# 		anomaly_path.append(i)
					# 		count2 += 1
					# 		flag = 1;
					# 		break;
				
					search_test.append(1)
					# anomaly_test.append(flag)
				else:
					search_test.append(0)
					# anomaly_test.append(0)

print len(train_data), "traces valid in training"
print len(test_data), "traces valid in testing"
# print len(search_path), 'anomaly_path related to search'
# print len(anomaly_path), 'target_path once out of the comfort range' 
# print 'valid rate (traces have data in the specified period)', float(valid)/len(search_path) 
# print 'anomaly rate (#anomaly/#valid)', float(count)/valid 

print '==========='
print 'training...'
print '==========='
clf = DT()
clf = clf.fit(train_data, search_train)
print '=========='
print 'testing...'
print '=========='
predict_res = clf.predict(test_data)
print '==============='
print 'stats coming...'
print '==============='

acc = 0
tp = 0
fp = 0
tn = 0
fn = 0
bonus = 0
for i,j,k in zip(predict_res, search_test, valid):
	if i==j:
		acc+=1
		if i==1:
			tp+=1
			if k in search_path:
				bonus+=1
		else:
			tn+=1
	else:
		if i==1:
			fp+=1
		else:
			fn+=1
print '# got back:', sum(search_test)
print '# bonus from classifer', bonus 
print 'ACC:', float(acc)/len(search_test)
print 'TPR:', float(tp)/sum(search_test)
print 'FPR:', float(fp)/(len(search_test)-sum(search_test))
print 'PPV:', float(tp)/(tp+fp)
print 'NPV:', float(tn)/(tn+fn)

# f = open('api_list','w')
# f.write("-------anomaly list of training case-------\n")
# f.writelines(["%s\n"%i for i in anomaly_path])	
# f.write("-------anomaly list of test case-------\n")
# f.writelines(["%s\n"%i for i in anomaly_path])	
# f.close()

# print len(search_path), 'anomaly_path related to search'
# print len(anomaly_path), 'target_path once out of the comfort range' 
# print 'valid rate (traces have data in the specified period)', float(valid)/len(search_path) 
# print 'anomaly rate (#anomaly/#valid)', float(count)/valid 
