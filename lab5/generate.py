import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

x_tr = np.linspace(0.,2,200)
f = lambda x: np.exp(3 * x)
y_tr = f(x_tr)
#print "Regression Curve Data: ", x_tr, y_tr

x = np.array([0,.1,.2,.5,.8,.9, 1])
y=f(x)+np.random.randn(len(x))
#print "Data: ", x, y

#create the model
lr = lm.LinearRegression()
#train the model
lr.fit(x[:, np.newaxis],y);
#predict new points
y_lr=lr.predict(x_tr[:, np.newaxis])

ridge=lm.RidgeCV()
plt.figure(figsize=(6,3));
plt.plot(x_tr, y_tr, "--k");
#plt.plot(x_tr, y_tr, 'ok', ms=10);

#Linear Regression code
#plt.plot(x_tr, y_lr, 'g')
#plt.xlim(0,1);
#plt.ylim(y.min()-1, y.max()+1);
#plt.title("Linear regression")

for deg, s in zip([2,5], ['-','.']):
	ridge.fit(np.vander(x, deg+1), y);
	y_ridge = ridge.predict(np.vander(x_tr, deg+1));
	plt.plot(x_tr, y_ridge, s, label='degree ' + str(deg));
	plt.legend(loc=2);
	plt.xlim(0,1.5);
	plt.ylim(-5, 80);
	print(' '.join(['%.2f' % c for c in ridge.coef_]))
plt.plot(x,y, 'ok', ms=10)
plt.title("Ridge regression")

plt.show()
