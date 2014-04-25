from sklearn.cross_validation import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix as CM
import numpy as np
import pylab as pl

input = np.genfromtxt('rice', delimiter=',')
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
    clf = DT(criterion='entropy', random_state=0)
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    preds.append(pred)
    if pred != test_label:
        ctr += 1
        #print 'inst', i+1, 'test label:', test_label, 'predicted label:', pred
    i+=1

print '%d wrongly predicted'%ctr, 'err rate:', float(ctr)/len(data)

cm = CM(label,preds)
cm = cm/cm.astype(np.float).sum(axis=1)
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()
