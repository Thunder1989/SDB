lines = [i.strip('\n').split('-') for i in open('log','r').readlines()]
correct = []
wrong = []

for line in lines:
    res = line[1].split(':')
    if res[0]==res[1]:
        correct.append(float(line[-1]))
    else:
        wrong.append(float(line[-1]))

f = open('array','w')
f.write('%s\n'%correct)
f.write('%s\n'%wrong)
f.close()

