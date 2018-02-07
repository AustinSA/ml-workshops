import csv
import sys

def tmtosecs(t):
	h = int(t[0])
	m = int(t[2]) * 10 + int(t[3])
	s = int(t[5]) * 10 + int(t[6])
	return (h * 60 * 60) + (m * 60) + s

maxsecs = 0

with open('sf-web-data.csv', 'rb') as csvfile:
	rdr = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in rdr:
		try:
			s = tmtosecs(row[3])
			print row
		except:
			print >>sys.stderr, "ERROR SKIPPING BAD ROW: ", row

