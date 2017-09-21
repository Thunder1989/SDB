import sys

print sys.argv
print len(sys.argv)

query = "grep -i " + sys.argv[1] + " MetadataDump.modified"
for string in sys.argv[2:]:
	query = query + " | grep -i " + string

print query