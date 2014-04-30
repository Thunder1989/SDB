from sklearn.cross_validation import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
import numpy as np
import pylab as pl

input1 = np.genfromtxt('sdh', delimiter=',')
data1 = input1[:,0:-1]
label1 = input1[:,-1]
input2 = np.genfromtxt('rice', delimiter=',')
data2 = input2[:,0:-1]
label2 = input2[:,-1]

preds = []
clf = DT(criterion='entropy', random_state=0)
clf.fit(data1, label1)
preds = clf.predict(data2)

i = 0
ctr = 0
while i<len(preds):
    if preds[i] != label2[i]:
        ctr += 1
        #print 'inst', i+1, 'test label:', test_label, 'predicted label:', pred
    i+=1

print '%d training instancs'%len(data1)
print '%d testing instancs'%len(data2)
print '%d wrongly predictedi at err rate %.3f'%(ctr, float(ctr)/len(data2))

cm = CM(label2,preds)
print cm
cm = normalize(cm.astype(np.float), axis=1, norm='l1')
print cm
#cm /= cm.astype(np.float).sum(axis=1)
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)

for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.2f"%cm[x][y]), xy=(y,x),
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

