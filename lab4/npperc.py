import numpy as np
a = np.array([[30,40,70],[80,20,10],[50,90,60]])

print "Our array is: "
print a
print "\n"

print "Applying percentile() function: "
print np.percentile(a,50)
print '\n'

print "Applying percentile() function on axis 1: "
print np. percentile(a,50,axis=1)
print '\n'

print "Applying percentile() function on axis 0: "
print np.percentile(a,50, axis=0)
print '\n'
