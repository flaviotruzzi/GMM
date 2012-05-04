#!/usr/bin/env 
# cython: boundscheck=False
# cython. wraparound=False
# cython: cdivision=True
# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


##############################################################################
# Calcula parâmetros das PDF #
##############################################################################
cdef pdf_params(np.ndarray[DTYPE_t, ndim=3] covars,
		np.ndarray[DTYPE_t, ndim=1] coefs,
		np.ndarray[DTYPE_t, ndim=3] inv_covars
		):
	cdef int dims = covars.shape[1]
	for k in range(covars.shape[0]):
		print k
		coefs[k] = 1.0/sqrt( (2 * np.pi)**dims * np.linalg.det(covars[k]) )
		inv_covars[k] = np.linalg.inv(covars[k])


##############################################################################
# Veross #
##############################################################################
cdef inline double veross(unsigned int i,unsigned int j,unsigned int dims,
			  double* x, double* means,
			  double* coefs, double* M):
	cdef double xm[5]
	x += j*dims
	M += i*dims*dims

	cdef int aj,ak 
	for ak in range(dims):
		xm[ak] = x[ak] - means[i*dims+ak]

	cdef double dotdot = 0.0
	for aj in range(dims):
		for ak in range(dims):
			dotdot += xm[aj] * xm[ak] * M[aj*dims+ak]

	return coefs[i] * exp(-.5 * dotdot)

##############################################################################
# E-Step #
##############################################################################
cdef EStep(int n_mixture,
	   np.ndarray[DTYPE_t, ndim=2] data,
	   np.ndarray[DTYPE_t, ndim=2] means,
	   np.ndarray[DTYPE_t, ndim=3] covars,
	   np.ndarray[DTYPE_t, ndim=2] z,
	   np.ndarray[DTYPE_t, ndim=1] pk,
	   np.ndarray[DTYPE_t, ndim=1] coefs,
	   np.ndarray[DTYPE_t, ndim=3] inv_covars
	   ):

	cdef unsigned int i, j, datalen	
	datalen = data.shape[0]

	cdef int dims = data.shape[1]
	cdef double nf
	for j in xrange(datalen):
		nf = 0.0
		for i in xrange(n_mixture):
			z[j, i] = pk[i] * veross(i, j, dims,
						 <double*>data.data,
						 <double*>means.data,
						 <double*>coefs.data,
						 <double*>inv_covars.data)
			nf = nf + z[j, i]
		## Normalize this line
		for i in xrange(n_mixture):
			z[j, i] = z[j, i] / nf

	## Calculate new pk values
	for i in xrange(n_mixture):
		pk[i] = 0
		for j in xrange(datalen):
			pk[i] += z[j, i]
	nf = 0.0
	for i in xrange(n_mixture):
		nf += pk[i]
	for i in xrange(n_mixture):
		pk[i] = pk[i] / nf




##############################################################################
# M-Step #
##############################################################################
cdef MStep(int n_mixture, np.ndarray[DTYPE_t, ndim=2] data,
	   np.ndarray[DTYPE_t, ndim=2] means,
	   np.ndarray[DTYPE_t, ndim=3] covars,
	   np.ndarray[DTYPE_t, ndim=2] z,
	   np.ndarray[DTYPE_t, ndim=1] pk):

	cdef unsigned int i, j,aj,ak,dd

	cdef unsigned int dims = data.shape[1]

        ## WATCH OUT assuming a maximum of 5 classes in problem
	cdef double class_sum[5]
        ## WATCH OUT assuming a maximum of 5 dimensions in problem
	cdef double xx[5]

	cdef double* d_data = <double*>data.data
	cdef double* m_data = <double*>means.data
	cdef double* z_data = <double*>z.data

	cdef double acc[25], zz


	## Calculate the sum of each column of z
	for i in xrange(n_mixture):
		class_sum[i] = 0.0
		for j in xrange(data.shape[0]):
			class_sum[i] += z_data[j*n_mixture+i]


	## Calculate new mean estimates using weighted means
	for i in xrange(n_mixture):
		## Clear means
		for dd in xrange(dims):
			m_data[i*dims+dd] = 0.0
		## Accumulate weighted coordinates
		for j in xrange(data.shape[0]):
			zz = z_data[j*n_mixture+i] ## Weight from this point
			for dd in xrange(dims):
				m_data[i*dims+dd] += zz * d_data[j*dims+dd]
		for dd in xrange(dims):
			m_data[i*dims+dd] = m_data[i*dims+dd] / class_sum[i]



	## Calculate new covariance matrices using weighted means, and new mean
	## estimates
	for i in xrange(n_mixture):
		# Clean accumulator
		for aj in xrange(5):
			for ak in xrange(5):
				acc[aj*5+ak] = 0.0

		## Iterate over the points and accumulate weighted outer
		## products
		for j in xrange(data.shape[0]):
			zz = z_data[j*n_mixture+i] ## Weight from this point
			## Calculate new vector-minus-mean
			for dd in xrange(dims):
				xx[dd] = d_data[j*dims+dd] - m_data[i*dims+dd]
			for aj in xrange(dims):
				for ak in xrange(aj,dims):					
					acc[aj*dims+ak] += zz * xx[aj] * xx[ak]
		## Divide by sum of weights from this class, and assign to
		## output array
		for aj in xrange(dims):
			for ak in xrange(aj,dims):
				zz = acc[aj*dims+ak] / class_sum[i]
				covars[i,aj,ak] = zz
				if ak > aj:
					covars[i,ak,aj] = zz
					



##############################################################################
# Fit #
##############################################################################
cdef fit(unsigned int iter, int n_mixture,
	 np.ndarray[DTYPE_t, ndim=2] data, 
	 np.ndarray[DTYPE_t, ndim=2] means,
	 np.ndarray[DTYPE_t, ndim=3] covars, 
	 np.ndarray[DTYPE_t, ndim=2] z,
	 np.ndarray[DTYPE_t, ndim=1] pk,
	 np.ndarray[DTYPE_t, ndim=1] coefs,
	 np.ndarray[DTYPE_t, ndim=3] inv_covars):
	for it in xrange(iter):
		pdf_params(covars, coefs, inv_covars)
		EStep(n_mixture, data, means, covars, z, pk, coefs, inv_covars)
		MStep(n_mixture, data, means, covars, z, pk)

cdef extern double atan2(double,double)
cdef extern double floor(double)
cdef extern double ceil(double)
cdef extern double hypot(double,double)
cdef extern double sqrt(double)
cdef extern double log10(double)
cdef extern double exp(double)
cdef extern double fabs(double)


##############################################################################
# Python class #
##############################################################################
class EMGMM:

	def __init__(self, unsigned int n_mixture, np.ndarray[DTYPE_t, ndim=2] data):

		self.n_mixture = n_mixture
		self.data = data
		self.dim = data.shape[1]
		self.means = np.ones((n_mixture, self.dim))
		self.covars = np.ones((n_mixture, self.dim, self.dim))
		self.covars *= 0.01 * np.identity(self.dim)
		self.means = kmeans(n_mixture, data)[0]
		self.z = np.zeros((len(data), self.n_mixture))

		self.pk = np.ones((n_mixture,)) / n_mixture

		self.coefs = np.zeros((n_mixture,))
		self.inv_covars = np.zeros(self.covars.shape)

	def iterate(self, iter):
		fit(iter, self.n_mixture, self.data, self.means, self.covars, self.z, self.pk,
		    self.coefs, self.inv_covars)
		#except:
			#print "Singular Covariance Matrix... Restarting..."
			#self.__init__(self.n_mixture, self.data)
			#self.iterate(iter)

##############################################################################


##############################################################################
# K means #
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
# Distance #
##############################################################################
cdef distance(np.ndarray[DTYPE_t, ndim=2] clusters, 
	np.ndarray[DTYPE_t, ndim=2] data):
	
	cdef unsigned int i
	cdef np.ndarray[DTYPE_t, ndim=2] c = np.zeros((clusters.shape[0], data.shape[0]))



	for i in xrange(clusters.shape[0]):
		c[i] = np.sqrt(((data-clusters[i])**2).sum(axis=1))
	return c.T


## Local Variables:
## fill-column: 79
## End:
