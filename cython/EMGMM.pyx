#!/usr/bin/env 

from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport exp



DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


##############################################################################
#										E-Step																									 #
##############################################################################
cdef EStep(int n_mixture, np.ndarray[DTYPE_t, ndim=2] data, 
	np.ndarray[DTYPE_t, ndim=2] means, np.ndarray[DTYPE_t, ndim=3] covars, 
	np.ndarray[DTYPE_t, ndim=2] z):

	cdef unsigned int i, j, datalen	
	datalen = data.shape[0]
	for i in xrange(n_mixture):
		for j in xrange(datalen):
			z[j, i] = veross(i, j, data, means, covars)
	z = z.T/z.sum(axis=1).T

##############################################################################
#										M-Step																									 #
##############################################################################
cdef MStep(int n_mixture, np.ndarray[DTYPE_t, ndim=2] data, 
	np.ndarray[DTYPE_t, ndim=2] means, np.ndarray[DTYPE_t, ndim=3] covars, 
	np.ndarray[DTYPE_t, ndim=2] z):

	cdef np.ndarray[DTYPE_t, ndim=2] newmi = np.zeros_like(means)
	cdef np.ndarray[DTYPE_t, ndim=3] newcov = np.zeros_like(covars)
	cdef unsigned int i, j

	for i in xrange(n_mixture):
		for j in xrange(data.shape[0]):
			newmi[i] += z[j, i] * data[j]			
			newcov[i] += z[j, i] * np.outer(data[j] - means[i],data[j] - means[i])
		means[i] = newmi[i] / z[:, i].sum()
		covars[i] = newcov[i] / z[:, i].sum()

##############################################################################
#										Veross  																								 #
##############################################################################
cdef veross(unsigned int i, unsigned int j, np.ndarray[DTYPE_t, ndim=2] data, 
	np.ndarray[DTYPE_t, ndim=2] means, np.ndarray[DTYPE_t, ndim=3] covars):

	cdef double x = data[j]
	cdef np.ndarray[DTYPE_t, ndim=1] mean = means[i]
	cdef np.ndarray[DTYPE_t, ndim=2] cov = covars[i]
	cdef np.ndarray[DTYPE_t, ndim=1] xm = np.subtract(x,means[i])

	print 'x', x
	print 'mean', mean
	print 'cov', cov
	print 'xm', xm

	return (1/(2 * np.pi * np.linalg.det(cov) ** 0.5)) * \
				exp(-.5 * np.dot(np.dot(xm, np.linalg.inv(cov)),xm))

##############################################################################
#										Fit   																									 #
##############################################################################
cdef fit(unsigned int iter, int n_mixture, np.ndarray[DTYPE_t, ndim=2] data, 
	np.ndarray[DTYPE_t, ndim=2] means, np.ndarray[DTYPE_t, ndim=3] covars, 
	np.ndarray[DTYPE_t, ndim=2] z):
	for it in xrange(iter):
		EStep(n_mixture, data, means, covars, z)
		MStep(n_mixture, data, means, covars, z)

cdef extern double atan2(double,double)
cdef extern double floor(double)
cdef extern double ceil(double)
cdef extern double hypot(double,double)
cdef extern double sqrt(double)
cdef extern double log10(double)
cdef extern double exp(double)
cdef extern double fabs(double)


##############################################################################
#										Python Object	:)																				 #
##############################################################################
class EMGMM:

	def __init__(self, unsigned int n_mixture, np.ndarray[DTYPE_t, ndim=2] data):

		self.n_mixture = n_mixture
		self.data = data
		self.dim = data.shape[1]
		self.means = np.ones((n_mixture, self.dim))
		self.covars = np.ones((n_mixture, self.dim, self.dim))
		self.covars *= np.identity(self.dim)
		self.means = kmeans(n_mixture, data)[0]
		self.z = np.zeros((len(data), self.n_mixture))

	def iterate(self, iter):
		fit(iter, self.n_mixture, self.data, self.means, self.covars, self.z)
		#except:
			#print "Singular Covariance Matrix... Restarting..."
			#self.__init__(self.n_mixture, self.data)
			#self.iterate(iter)

##############################################################################


##############################################################################
#										K means          																				 #
##############################################################################
cdef kmeans(unsigned int n_clusters, np.ndarray[DTYPE_t, ndim=2] data):

	cdef np.ndarray[DTYPE_t, ndim=2] k = np.random.randint(floor(data.min()), 
		ceil(data.max()+1), size=(n_clusters, data.shape[1]))/1

	cdef np.ndarray[DTYPE_t, ndim=2] kn = np.zeros_like(k)
	cdef np.ndarray[DTYPE_t, ndim=2] c = np.zeros((data.shape[0], k.shape[0]))

	cdef unsigned int i

	while(True):
		
		c = distance(k, data)
		c[:, 0] = c.argmin(axis=1)

		for i in xrange(len(k)):
			if (len(data[c[:,0]==i]) > 0):
				kn[i] = data[c[:,0]==i].sum(axis=0)/len(data[c[:,0]==i])

		if (len(np.where((kn==k) == False)[0]) == 0):
			break
		else:
			k = 1*kn

	return [k,c[:,0]]


##############################################################################
#										Distance        																				 #
##############################################################################
cdef distance(np.ndarray[DTYPE_t, ndim=2] clusters, 
	np.ndarray[DTYPE_t, ndim=2] data):
	
	cdef unsigned int i
	cdef np.ndarray[DTYPE_t, ndim=2] c = np.zeros((clusters.shape[0], data.shape[0]))



	for i in xrange(clusters.shape[0]):
		c[i] = np.sqrt(((data-clusters[i])**2).sum(axis=1))
	return c.T
