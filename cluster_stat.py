import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
from scikits.statsmodels.tools.tools import ECDF
from scipy import stats
from scipy.optimize import curve_fit
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import silhouette_score as SS
from sklearn.metrics import confusion_matrix as CM
from sklearn import tree
from sklearn.preprocessing import normalize

input1 = [i.strip().split('+')[-1][:-5] for i in open('sdh_pt_new_forrice').readlines()]
input2 = np.genfromtxt('sdh_45min_forrice', delimiter=',')
input3 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_forsdh').readlines()]
input4 = np.genfromtxt('rice_45min_forsdh', delimiter=',')
input5 = [i.strip().split('_')[-1][:-5] for i in open('soda_pt_new').readlines()]
input6 = np.genfromtxt('soda_45min_new', delimiter=',')
label1 = input2[:,-1]
label = input4[:,-1]
label1 = input6[:,-1]
#input3 = input1 #for quick run of the code using other building
#input3, label = shuffle(input3, label)
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))

cv = CV(analyzer='char_wb', ngram_range=(3,4))
#tv = TV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()
#fn = cv.fit_transform(input1).toarray()
fd = input4[:,[0,1,2,3,5,6,7]]
#print 'class count of true labels of all ex:\n', ct(label)
#n_class = len(np.unique(label))
#print n_class
#print np.unique(label)
#print 'class count from groud truth labels:\n',ct(label)
#kmer = cv.get_feature_names()
#idf = zip(kmer, cv._tfidf.idf_)
#idf = sorted(idf, key=lambda x: x[-1], reverse=True)
#print idf[:20]
#print idf[-20:]
#print cv.get_feature_names()

clf = LinearSVC()
#clf = SVC(kernel='linear')
#clf = RFC(n_estimators=100, criterion='entropy')
clf.fit(fn, label)
coef = abs(clf.coef_)
weight = np.max(coef, axis=0)
#weight = np.mean(coef,axis=0)
feature_rank = []
for i,j in zip(weight, xrange(len(weight))):
    feature_rank.append([i,j])
feature_rank = sorted(feature_rank,key=lambda x: x[0],reverse=True)
feature_idx=[]
for i in feature_rank:
    if i[0]>=0.05:
        feature_idx.append(i[1])
#print 'feature num', len(feature_idx)
fn = fn[:, feature_idx]

#fn = fd
#fn = StandardScaler().fit_transform(fn)

'''
#explained variance v.s. # of clusters
Y = np.mean(fn, axis=0)
V = np.sum((fn-Y)**2, axis=0)/(len(fn)-1)
var_expln = []
ss = []
kk = range(2,50)
for n in kk:
    #print 'running', n
    c = KMeans(init='k-means++', n_clusters=n, n_init=10)
    c.fit(fn)
    ss.append(SS(fn,c.labels_))

    ex = dd(list) #distance to centroid
    for i,j in zip(c.labels_, xrange(len(fn))):
        ex[i].append(j)

    centers = c.cluster_centers_
    win = 0
    btw = 0
    for k,v in ex.items():
        win += np.sum((fn[v]-centers[k])**2)
        btw += len(v)*np.sum((centers[k]-Y)**2)
    ratio = 100*btw/(win+btw)
    #ratio = btw/(n-1)/(win/(len(fn)-n))
    var_expln.append(np.mean(ratio))
diff = np.diff(var_expln)
#elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(kk, var_expln, 'b*-')
ax.set_ylabel('Percentage of variance explained (%)')
ax.set_ylim((0,100))
ax2 = ax.twinx()
#ax2.plot(kk[1:], diff, 'r*-')
ax2.plot(kk, ss, 'r*-')
ax2.set_ylabel('Diff from previous')
#ax.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12,
#    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
#ax.set_ylim((0,1))
plt.grid(True)
plt.xlabel('Number of clusters')
plt.title('Elbow for KMeans clustering')
plt.show()
'''


n_class = 16*2
c = KMeans(init='k-means++', n_clusters=2*n_class, n_init=10)
c.fit(fn)

dist = np.sort(c.transform(fn))
ex = dd(list) #distance to centroid
d_c = dd(list) #distance to centroid
for i,j,k in zip(c.labels_, xrange(len(fn)), dist):
        ex[i].append(k[0])
        d_c[i].append([j,k[0]])

'''
#distance to centroid v.s. label
s2center = []
d2center = []
for k,v in d_c.items():
    d_c[k] = sorted(v,key=lambda x: x[-1])
for v in d_c.values():
    l = label[v[0][0]]
    for vv in v:
        if label[vv[0]] == l:
            s2center.append(vv[-1])
        else:
            d2center.append(vv[-1])
            break

src = s2center
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'r--', label='same as centroid')
src = d2center
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'k--', label='1st diff than centroid')
plt.legend(loc='lower right')
plt.xlabel('L2 distance')
plt.title('dist. to centroid distribution')
plt.grid(axis='y')
plt.show()
'''

'''
#distance matrix, diagonal is cluster radius, other cell is inter cluster centroid dist
cm = np.zeros((32,32))
center = c.cluster_centers_
for i in xrange(32):
    cm[i][i] = np.max(ex[i])
    for j in xrange(i):
        cm[i][j] = np.linalg.norm(center[i]-center[j])
def printMatrix(M):
    print '  ',
    for i in range(len(M[1])):  # Make it work with non square matrices.
        print i,'  ',
    print
    for i,element in enumerate(M):
        print '%d(%d)'%(i,len(ex[i])),' '.join([str("%.3f"%e) for e in element])
printMatrix(cm)
count = 0
for i in xrange(32):
    for j in xrange(i):
        if cm[i][j] < cm[i][i] + cm[j][j]:
            count += 1
print count
'''

'''
fig = pl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
for x in xrange(len(cm)):
    for y in xrange(len(cm)):
        ax.annotate(str("%.3f"%cm[x][y]), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10)
cls = []
#pl.yticks(range(len(cls)), cls)
#pl.ylabel('True label')
#pl.xticks(range(len(cls)), cls)
#pl.xlabel('Predicted label')
pl.title('Cluster Distance Matrix')
pl.show()
'''


#generating CDF
p_same = []
p_diff = []
p_all = []#all pairs
intra_same = []
intra_diff = []
intra_all = []#only intra cluster pairs
inter_same = []
inter_diff = []
inter_all = []#only inter cluster pairs
for i in xrange(len(fn)):
    for j in xrange(0,i):
        d = np.linalg.norm(fn[i]-fn[j])
#for v in ex_id.values():
#    pair = list(itertools.combinations(v,2))
#    for p in pair:
#        d = np.linalg.norm(fn[p[0]]-fn[p[1]])
        p_all.append(d)
        if c.labels_[i] == c.labels_[j]:
            intra_all.append(d)
            if label[i] == label[j]:
            #if label[p[0]] == label[p[1]]:
                intra_same.append(d)
                p_same.append(d)
            else:
                intra_diff.append(d)
                p_diff.append(d)
        else:
            inter_all.append(d)
            if label[i] == label[j]:
                inter_same.append(d)
                p_same.append(d)
            else:
                inter_diff.append(d)
                p_diff.append(d)
#print len(p_all)
#t,p = stats.ttest_ind(same, diff, equal_var=False)
#print t,p
#y1 = np.mean(same)
#y2 = np.mean(diff)
#s1 = np.var(same)
#s2 = np.var(diff)
#n1 = len(same)
#n2 = len(diff)
#T = (y1-y2)/np.sqrt(s1/n1 + s2/n2)
#print T

src = p_same
ecdf = ECDF(src)
#plt.plot(ecdf.x, ecdf.y, 'b--', label='within')
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'r--', label='within')
src = p_diff
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
print 'true tao', min(src)
y = ecdf(x)
plt.plot(x, y, 'b--', label='across')
src = p_all
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'k--', label='all')
plt.legend(loc='lower right')
plt.xlabel('L2 distance')
plt.title('pairwise dist. distribution on all pairs')
plt.grid(axis='y')
plt.show()
s = raw_input()

src = intra_same
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'r--', label='within')
src = intra_diff
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'b--', label='across')
src = intra_all
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'k--', label='all')
plt.legend(loc='lower right')
plt.xlabel('L2 distance')
plt.title('pairwise dist. distribution on intra-cluster pairs')
plt.grid(axis='y')
plt.show()
s = raw_input()

src = inter_same
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'r--', label='within')
src = inter_diff
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'b--', label='across')
src = inter_all
ecdf = ECDF(src)
x = np.linspace(min(src), max(src), int((max(src)-min(src))/0.01))
y = ecdf(x)
plt.plot(x, y, 'k--', label='all')
plt.legend(loc='lower right')
plt.xlabel('L2 distance')
plt.title('pairwise dist. distribution on inter-cluster pairs')
plt.grid(axis='y')
plt.show()
s = raw_input()

