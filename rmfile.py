import glob
import os
import re

path = '/Users/hdz_1989/Documents/Dropbox/SDB/KETI'
foler = glob.glob(path + '/*')
p = re.compile(".*win's.*")

for fd in foler:
	file = glob.glob(fd + '/*')
	for f in file:
		if p.match(f):
			os.remove(f)
