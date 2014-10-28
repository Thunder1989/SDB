lines = [i.strip().split('_')[0].strip() for i in open('soda_').readlines()]
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
f = open('id','w')
f.writelines('%s\n'%i for i in list)
f.close()
