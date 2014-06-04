from sklearn.cross_validation import LeaveOneOut
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
import numpy as np
import pylab as pl

input = np.genfromtxt('rice_45min', delimiter=',')
data = input[:,0:-1]
label = input[:,-1]

'''
loo = LeaveOneOut(len(data))
for train_idx, test_idx in loo:
    pass
'''

i=0
ctr = 0
preds = []
while i<len(data):
    idx = range(len(data))
    idx.remove(i)

    train_data = data[idx]
    train_label = label[idx]
    test_data = data[i]
    test_label = label[i]
    #clf = DT(criterion='entropy', random_state=0)
    clf = RFC(n_estimators=6, criterion='entropy')
    clf.fit(train_data, train_label)
    #out = tree.export_graphviz(clf, out_file='tree33.dot')
    pred = clf.predict(test_data)
    preds.append(pred)
    if pred != test_label:
        ctr += 1
        print 'inst', i+1, '%d:%d'%(test_label,pred)#, test_data
    i+=1
print '# of instances', len(data)
print '%d wrongly predicted,'%ctr, 'err rate:', float(ctr)/len(data)

cm = CM(label,preds)
#print cm
cm = normalize(cm.astype(np.float), axis=1, norm='l1')
#print cm
#cm /= cm.astype(np.float).sum(axis=1)
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
