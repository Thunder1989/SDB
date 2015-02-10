from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
import numpy as np
import math
import pylab as pl

#input1 = np.genfromtxt('rice_45min_raw_sliding', delimiter=',')
input1 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
input2 = [i.strip().split('\\')[-1][:-4] for i in open('rice_pt_forsdh').readlines()]
#data1 = input1[:,[0,1,2,3,5,6,7,9,10,11]]
fd = input1[:,[0,1,2,3,5,6,7]]
#data1 = input1[:,0:-1]
#index = data1.shape[1]*2/3
#data1 = input1[:,0:index]
label = input1[:,-1]
input3 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input4 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_forrice').readlines()]
fd1 = input3[:,[0,1,2,3,5,6,7]]
label1 = input3[:,-1]

fold = 2
clx = 9
skf = StratifiedKFold(label, n_folds=fold)
acc_sum = []
indi_acc =[[] for i in range(clx)]
clf1 = RFC(n_estimators=100, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
clf2 = SVC(kernel='linear')
vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
fn = vc.fit_transform(input4).toarray()

loop = 0
run = fold
importance = np.zeros(fd.shape[1])
a_r = 0
a_w = 0
d_md_r = 0
d_mn_r = 0
while loop<run/fold:
    for train_idx, test_idx in skf:
        '''
        because we want to do inverse k-fold XV
        aka, use 1 fold to train, k-1 folds to test
        so the indexing is inversed
        '''
        train_fd = fd[test_idx]
        train_fn = fn[test_idx]
        train_label = label[test_idx]
        test_fd = fd[train_idx]
        test_fn = fn[train_idx]
        test_label = label[train_idx]
        #test_data = data2
        #test_label = label2
        clf1.fit(train_fd, train_label)
        clf2.fit(train_fn, train_label)
        #print clf.classes_
        #print clf.feature_importances_
        preds_fd = clf1.predict(test_fd)
        preds_fn = clf2.predict(test_fn)
        acc = accuracy_score(test_label, preds_fd)
        acc_sum.append(acc)
        #importance += clf.feature_importances_
        #print acc

        #compute the statistics per model on rights and wrongs
        for i,j,k in zip(preds_fd, preds_fn, test_label):
            if i==j:
                if i==k:
                    a_r += 1
                else:
                    a_w += 1
            else:
                if i==k:
                    d_md_r += 1
                else:
                    d_mn_r += 1

        '''
        cm_ = CM(test_label,preds)
        cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
        k=0
        while k<clx:
            indi_acc[k].append(cm[k,k])
            k += 1

        #debug co2 instances
        print '===================='
        importance += clf.feature_importances_
        for i,j in zip(train_label, test_idx):
            if i==1:
                    print 'train id:', j+1
        #for i,j,k in zip(test_label, preds, train_idx):
        for i,j,k in zip(test_label, preds, range(len(test_label))):
            if i==1 and i!=j:
                print '%d-%d'%(k+1,j)

        fig = pl.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)

        for x in xrange(len(cm)):
            for y in xrange(len(cm)):
                ax.annotate(str("%.3f(%d)"%(cm[x][y],cm_[x][y])), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center')


        cls = ['co2','humidity','rmt','stpt','flow','other_t']
        pl.xticks(range(len(cm)),cls)
        pl.yticks(range(len(cm)),cls)
        pl.title('Confusion matrix (%.3f)'%acc)
        pl.ylabel('True label')
        pl.xlabel('Predicted label')
        pl.show()
        '''

    loop+=1

num = len(test_label)
print float(a_r)/(run*num)
print float(a_w)/(run*num)
print float(d_md_r)/(run*num)
print float(d_mn_r)/(run*num)

#for i,j,k in zip(test_label, preds, train_idx):
#    if j==4 and i!=j:
#        print '%d-%d'%(k+1,i)

#print importance/run
#indi_ave_acc = [np.mean(i) for i in indi_acc]
#indi_ave_acc_std = [np.std(i) for i in indi_acc]
#print 'ave acc/type:', indi_ave_acc
#print 'acc std/type:', indi_ave_acc_std
print 'ave acc:', np.mean(acc_sum)
#print 'std:', np.std(acc_sum)

'''
cm_ = CM(test_label,preds)
#print cm
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
#print cm
#cm /= cm.astype(np.float).sum(axis=1)
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)

for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f(%d)"%(cm[x][y],cm_[x][y])), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10)

mapping = {1:'co2',2:'humidity',3:'pressure',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',14:'pressure',15:'timer',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
cls_id =np.unique(test_label)
cls = []
for c in cls_id:
    cls.append(mapping[c])
#cls = ['co2','humidity','pressure','rmt','status','stpt','flow','other T','occu']
#cls = ['rmt','pos','stpt','flow','other_t','ctrl','spd','sta','pressure','tmr'] #soda
#cls = ['rmt','pos','stpt','flow','other_t','pwr','ctrl','occu','spd','sta'] #sdh
#cls = ['co2','humidity','rmt','status','stpt','flow','HW sup','HW ret','CW sup','CW ret','SAT','RAT','MAT','C enter','C leave','occu']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Confusion matrix (%.3f)'%acc)
#pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()
'''
