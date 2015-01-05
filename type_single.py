from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
from collections import defaultdict
import numpy as np
import math
import random
import pylab as pl

# input1 = [i.strip().split('\\')[-1][:-4] for i in open('sdh_pt_new_forrice').readlines()]
#input1 = np.genfromtxt('sdh_45min_new', delimiter=',')
input1 = np.genfromtxt('rice_45min', delimiter=',')
# input2 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
#input2 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_all').readlines()]
input2 = [i.strip().split('\\')[-1][:-4] for i in open('rice_pt').readlines()]
# input2 = np.genfromtxt('sdh_45min_new', delimiter=',')
# input1 = [i.strip().split('_')[-1][:-4] for i in open('soda_pt_part').readlines()]
# input2 = np.genfromtxt('soda_45min_part', delimiter=',')
data1 = input1[:,[0,1,2,3,5,6,7]]
label1 = input1[:,-1]
# label2 = input4[:,-1]

'''
first, split the examples into several folds
'''
fold = 5
#skf = StratifiedKFold(label1, n_folds=fold)
kf = KFold(len(label1), n_folds=fold, shuffle=True)
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1

iteration = 100
clx = 9 #number of classes

acc_sum = [[] for i in range(iteration)] #log overall acc over iterations
acc_type = [[] for i in range(clx)]
for fd in range(fold):
    print 'running AL on new bldg - fold', fd
    train = np.hstack((folds[(fd+x)%fold] for x in range(1)))
    validate = np.hstack((folds[(fd+x)%fold] for x in range(1,fold/2)))
    # validate = np.append(validate,train[2:])
    # train = train[:2]
    test = np.hstack((folds[(fd+x)%fold] for x in range(fold/2,fold)))

    '''
    second, run the data model to predict labels
    '''
    test_ = np.append(test,validate)
    train_data = data1[train]
    train_label = label1[train]
    test_data = data1[test_]
    test_label = label1[test_]
    clf = RFC(n_estimators=50, criterion='entropy')
    clf.fit(train_data, train_label)
    #print 'training class in data model:\n', clf.classes_
    preds = clf.predict(test_data)
    md_acc = clf.score(test_data, test_label)
    print 'acc of data model', clf.score(test_data, test_label)

    #compute confidence for each example in the data model
    label_pr = np.sort(clf.predict_proba(test_data)) #sort each prob vector in ascending order
    cfdn_d = defaultdict(list) #confidence list
    for h,i,pr in zip(test_,preds,label_pr):
        # entropy = np.sum(-p*math.log(p,2) for p in pr if p!=0)
        if len(pr)<2:
            margin = 1
        else:
            margin = pr[-1]-pr[-2]
        cfdn_d[h].append([i,margin])

    '''
    third, run active learning on string model of the same bldg
    '''
    vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
    #vc = CV(token_pattern='[a-z]{2,}')
    data2 = vc.fit_transform(input2).toarray() #feature vector of string model
    label2 = label1
    # ex = []

    # train = np.hstack((folds[(fd+x)%fold] for x in range(1)))
    # validate = np.hstack((folds[(fd+x)%fold] for x in range(1,fold/2)))
    # validate = np.append(validate,train[1:])
    train = validate[:1] #training set starts from size 1
    validate = validate[1:]

    # test = np.hstack((folds[(fd+x)%fold] for x in range(fold/2,fold)))
    test_data = data2
    test_label = label2

    for itr in range(iteration):
        train_data = data2[train]
        train_label = label2[train]
        validate_data = data2[validate]
        validate_label = label2[validate]

        clf.fit(train_data, train_label)
        # print clf.classes_
        acc = clf.score(test_data, test_label)
        acc_sum[itr].append(acc)

        preds = clf.predict(test_data)
        cm = CM(test_label,preds)
        cm = normalize(cm.astype(np.float), axis=1, norm='l1')
        k=0
        while k<clx:
            acc_type[k].append(cm[k,k])
            k += 1

        #example selection
        #compute entropy for each instance and rank
        label_pr = np.sort(clf.predict_proba(validate_data)) #sort in ascending order, same as data model
        preds = clf.predict(validate_data)
        res = [] #ranking list
        for h,i,pr in zip(validate,preds,label_pr):
            # entropy = np.sum(-p*math.log(p,clx) for p in pr if p!=0)
            if len(pr)<2:
                margin = 1
            else:
                margin = pr[-1]-pr[-2]
            cfdn = cfdn_d[h][0][-1] #confidence of the same example from data model
            #res.append([h,i,cfdn/(margin+1)])
            res.append([h,i,margin])

        #res = sorted(res, key=lambda x: x[-1], reverse=True)
        res = sorted(res, key=lambda x: x[-1])
        # pick the first example on the list, which data model is most confident while str model least confident
        idx = 0

        # randomly pick one example
        #idx = random.randint(0,len(res)-1)

        elmt = res[idx][0] # get example id
        # ex.extend([itr+1, elmt, label1[elmt], label_gt[elmt]])
        train = np.append(train, elmt)
        validate = validate[validate!=elmt]

ave_acc = [np.mean(acc) for acc in acc_sum]
acc_std = [np.std(acc) for acc in acc_sum]

print 'overall acc:', repr(ave_acc)
#print 'acc std:', repr(acc_std)
#print 'acc by type', repr(acc_type)
# f = open('pipe_out','w')
# f.writelines('%s;\n'%repr(i) for i in acc_type)
# f.write('ex in each itr:'+repr(ex)+'\n')
# f.write(repr(np.unique(test_label)))
# f.close()
#for i in acc_type:
    #print 'a = ', repr(i), '; plot(a\');'
#print repr(ex)

#plot confusion matrice
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
preds = clf.predict(test_data)
cm_ = CM(test_label,preds)
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
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
cls_id = np.unique(test_label)
cls = []
for c in cls_id:
    cls.append(mapping[c])
#cls = ['co2','humidity','rmt','stpt','flow','other_t']
#cls = ['rmt','pos','stpt','flow','other_t','ctrl','spd','sta']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Mn Confusion matrix (%.3f)'%clf.score(test_data, test_label))
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()

cm_ = CM(test_label, label1)
cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
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
#cls = ['co2','humidity','rmt','stpt','flow','other_t']
#cls = ['rmt','pos','stpt','flow','other_t','ctrl','spd','sta']
pl.xticks(range(len(cm)),cls)
pl.yticks(range(len(cm)),cls)
pl.title('Md Confusion matrix (%.3f)'%md_acc)
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()
