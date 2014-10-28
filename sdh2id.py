lines = [i.strip().split('+')[-2].strip() for i in open('sdh').readlines()]
list = []
tmp = lines[0]
id = 1
for line in lines:
    if line==tmp:
        list.append(id)
    else:
        tmp = line
        id+=1
        list.append(id)
f = open('id2','w')
f.writelines('%s\n'%i for i in list)
f.close()
