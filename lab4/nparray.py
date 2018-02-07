import numpy as np
a = np.arange(6).reshape(3,2)

print "Our array is: "
print a 
print '\n'

print "Modfied array: "
wt = np.array([3,5])
print np.average(a, axis=1, weights=wt)
print '\n'

print"Modified array: "
print np.average(a, axis = 1, weights=wt, returned = True)
