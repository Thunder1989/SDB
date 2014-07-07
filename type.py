from sklearn.cross_validation import LeaveOneOut
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
import numpy as np
import math
import pylab as pl
import logging

input = np.genfromtxt('sdh_bsln', delimiter=',')
data = input[:,0:-1]
label = input[:,-1]

ctr = 0
preds = []
idx = LeaveOneOut(len(data))
#clf = DT(criterion='entropy', random_state=5)
clf = RFC(n_estimators=50, criterion='entropy')
#log = open('log_','w')
for train, test in idx:
    train_data = data[train]
    train_label = label[train]
    test_data = data[test]
    test_label = label[test]
    clf.fit(train_data, train_label)
    #out = tree.export_graphviz(clf, out_file='tree33.dot')
    pred = clf.predict(test_data)
    pr = clf.predict_proba(test_data)
    preds.append(pred)
    #entropy = np.sum(-p*math.log(p,6) for p in pr[0] if p!=0)
    #print 'inst', test+1, '%d:%d'%(test_label,pred)#, test_data
    #log.write('[%d]-%d:%d'%(test+1,test_label,pred))
    #log.write('-%s'%clf.predict_proba(test_data))
    #log.write('-%.3f\n'%entropy)
    if pred != test_label:
        ctr += 1

#log.close()
acc = accuracy_score(label, preds)
print ctr, 'wrong out of', len(data), 'instances'
print 'err rate:', 1-acc

cm = CM(label,preds)
#print cm
cm = normalize(cm.astype(np.float), axis=1, norm='l1')
indi_acc = [cm[i][i] for i in range(6)]
print indi_acc
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)

for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f"%cm[x][y]), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center')


cls = ['co2','humidity','rmt','stpt','flow','other_t']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix')
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()
