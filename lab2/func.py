from matplotlib import pyplot as plt
import numpy

t = numpy.arange(0.0, 2.0, 0.01)
s = numpy.sin(2.5 * numpy.pi * t)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV')
plt.title('Sine wave')
plt.grid(True)
plt.show()
