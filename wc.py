from sklearn.feature_extraction.text import CountVectorizer as CV
from collections import defaultdict
import numpy as np
import operator

#input1 = [i.strip().split('\\')[-1][:-4] for i in open('rice_pt').readlines()]
#input2 = np.genfromtxt('rice_45min_woOT', delimiter=',')
input1 = [i.strip().split('+')[-1][:-4] for i in open('sdh_pt_new_all').readlines()]
input2 = np.genfromtxt('sdh_45min_new', delimiter=',')
label = input2[:,-1]
vc = CV(token_pattern='[a-z]{2,}')
#vc = CV(analyzer='char_wb', ngram_range=(2,4), min_df=1, token_pattern='[a-z]{2,}')
data = defaultdict(list)
for i,j in zip(label,input1):
    data[i].append(j)
out = open('wc_out','w')
for i,j in data.items():
    v = vc.fit_transform(j).toarray()
    print v
    f = v.sum(axis=0)
    print f
    t = vc.get_feature_names()
    count = zip(t, f)
    count = sorted(count, key=operator.itemgetter(1))
    #print '=== %s (%s) ==='%(i, data.shape[0])
    out.write('=== %s (%s) ===\n'%(i, v.shape[0]))
    out.writelines('%s\n'%repr(c) for c in count)
    #for c in count:
        #print c
    #d = sorted(d.items(), key=operator.itemgetter(1))
    #print i,'===', d
out.close()
