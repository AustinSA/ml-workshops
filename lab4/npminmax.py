import numpy as np
a = np.array([[3,7,5],[8,4,3],[2,4,9]])

print "Our array is: "
print a
print "/n"

print "Applying amin() function: "
print np.amin(a,1)
print '/n'

print "Applying amin() function again: "
print np. amin(a,0)
print '/n'

print "Applying amax() functions: "
print np.amax(a)
print '/n'

print "Applying amax function again: "
print np.amax(a, axis=0)
