from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize
from collections import defaultdict as dd
import numpy as np
import math
import random
import pylab as pl

'''
input1 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
data1 = input1[:,[0,1,2,3,5,6,7]]
label1 = input1[:,-1]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
data2 = input2[:,[0,1,2,3,5,6,7]]
label2 = input2[:,-1]
#label = [1,2,4,6,7,8]
'''
input1 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-4] for i in open('rice_pt_forsdh').readlines()]
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
label1 = input2[:,-1]
label = input4[:,-1]

iteration = 100
fold = 5
#loo = LeaveOneOut(len(data))
#skf = StratifiedKFold(label1, n_folds=fold)
kf = KFold(len(label), n_folds=fold, shuffle=True)
folds = [[] for i in range(fold)]
i = 0
for train, test in kf:
    folds[i] = test
    i+=1

acc_sum = [[] for i in range(iteration)]
#acc_type = [[[] for i in range(iteration)] for i in range(6)]
#precision_type = [[[] for i in range(iteration)] for i in range(6)]
#recall_type = [[[] for i in range(iteration)] for i in range(6)]
#clf = ETC(n_estimators=10, criterion='entropy')
clf = RFC(n_estimators=50, criterion='entropy')
#clf = DT(criterion='entropy', random_state=0)
#clf = Ada(n_estimators=100)
#clf = SVC(kernel='linear')

vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
fn = vc.fit_transform(input3).toarray()
for fd in range(fold):
    train = np.hstack((folds[(fd+x)%fold] for x in range(1)))
    validate = np.hstack((folds[(fd+x)%fold] for x in range(1,fold/2)))
    validate = np.hstack((train,validate))
    '''
    ex_dict = dd(list)
    train_label = label[validate]
    for i,j in zip(train_label,validate):
        ex_dict[i].append(j)
    train = []
    #get one ex per class as initial train
    for v in ex_dict.values():
        random.shuffle(v)
        train.append(v[0])
        #remove train ex id from validate, the rest is the new validate
        validate = validate[validate!=v[0]]
    '''
    #cut train to one example
    validate = np.append(validate,train[2:])
    train = train[:2]

    #print len(train)
    test = np.hstack((folds[(fd+x)%fold] for x in range(fold/2,fold)))
    test_data = fn[test]
    test_label = label[test]

    for itr in range(iteration):
        #if itr%10==0:
        #    print 'running fold %d iter %d'%(fd, itr)
        train_data = fn[train]
        train_label = label[train]
        validate_data = fn[validate]
        #validate_label = label[validate]

        clf.fit(train_data, train_label)
        #print clf.classes_
        preds = clf.predict(test_data)
        acc = clf.score(test_data, test_label)
        acc_sum[itr].append(acc)

        '''
        cm_ = CM(test_label,preds)
        cm = normalize(cm_.astype(np.float), axis=1, norm='l1')

        #plot confusion matrix, for debugging
        if itr<20 and itr%2==0 or itr==iteration-1:
            print itr
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

            mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
            cls_id = np.unique(test_label)
            cls = []
            for c in cls_id:
                cls.append(mapping[c])
            pl.xticks(range(len(cm)),cls)
            pl.yticks(range(len(cm)),cls)
            pl.title('Confusion matrix (%.3f)'%acc)
            pl.ylabel('True label')
            pl.xlabel('Predicted label')
            pl.show()

        #statistics by type
        pre = precision_score(test_label, preds, average=None)
        rec = recall_score(test_label, preds, average=None)
        k=0
        while k<6:
            acc_type[k][itr].append(cm[k,k])
            precision_type[k][itr].append(pre[k])
            recall_type[k][itr].append(rec[k])
            k += 1
        '''

        #entropy based example selection block
        #compute entropy for each instance and rank
        label_pr = np.sort(clf.predict_proba(validate_data)) #sort in ascending order
        preds = clf.predict(validate_data)
        res = []
        for h,i,pr in zip(validate,preds,label_pr):
            #entropy = np.sum(-p*math.log(p,2) for p in pr if p!=0)
            if len(pr)<2:
                margin = 1
            else:
                margin = pr[-1]-pr[-2]
            res.append([h,i,margin])
        #print 'iter', itr, 'wrong #', len(wrong)

        '''
        #Entropy-based, sort and pick the one with largest H
        res = sorted(res, key=lambda x: x[-2], reverse=True)
        idx = 0


        #Margin-based, sort and pick the one with least margin
        res = sorted(res, key=lambda x: x[-1])
        idx = 0


        #least confidence based
        tmp = sorted(label_pr, key=lambda x: x[-1])
        idx = 0


        #Expectation-based, pick the one with H most close to 0.5
        for i in res:
            i[-2] = abs(i[-2]-0.5)
        res = sorted(res, key=lambda x: x[3])
        idx = 0

        '''
        #randomly pick one
        idx = random.randint(0,len(res)-1)


        elmt = res[idx][0]
        #print 'running fold %d iter %d'%(fd, itr)
        #print label1[elmt]

        '''
        #minimal future expected error
        loss = []
        label_pr = clf.predict_proba(validate_data)
        clx = clf.classes_
        for i, pr in zip(validate, label_pr):
            #print 'validate ex#', i
            #print 'pr vector', pr
            new_train = np.append(train,i)
            new_validate = validate[validate!=i]

            new_train_data = data1[new_train]
            new_train_label = label1[new_train]
            new_validate_data = data1[new_validate]
            new_validate_label = label1[new_validate]

            err = 0
            for j in range(len(pr)):
                #compute the sum of confidence for the rest of examples in validate set
                #on the new re-trained model after each possbile labeling (x,y_i) of i is added to the train set
                if pr[j]==0:
                    continue

                new_train_label[-1] = clx[j]
                clf.fit(new_train_data, new_train_label)
                confidence_sum = np.sum(1-np.sort(clf.predict_proba(new_validate_data))[:,-1])
                err += pr[j]*confidence_sum

            loss.append([i,err])

        loss = sorted(loss, key=lambda x: x[-1])
        #print loss

        elmt = loss[0][0]
        '''

        #remove the item from validate set
        #add it to train set
        train = np.append(train, elmt)
        validate = validate[validate!=elmt]
        #train_idx.append(elmt)
        #test_idx.remove(elmt)

ave_acc = [np.mean(acc) for acc in acc_sum]
acc_std = [np.std(acc) for acc in acc_sum]
'''
ave_acc_type = [[] for i in range(6)]
ave_pre = [[] for i in range(6)]
ave_rec = [[] for i in range(6)]
for i in range(6):
    ave_acc_type[i] = [np.mean(a) for a in acc_type[i]]
    ave_pre[i] = [np.mean(p) for p in precision_type[i]]
    ave_rec[i] = [np.mean(r) for r in recall_type[i] ]
'''
print 'overall acc:', repr(ave_acc)
#print 'acc std:', repr(acc_std)
'''
print '=================================='
print 'acc by type:', repr(ave_acc_type)
print '=================================='
print 'precision by type:', repr(ave_pre)
print '=================================='
print 'recall by type:', repr(ave_rec)
'''
