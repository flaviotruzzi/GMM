#!/usr/bin/env 
# cython: boundscheck=False
# cython. wraparound=False
# cython: cdivision=True
# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np

from scipy import ndimage

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
        #print k
        coefs[k] = 1.0/sqrt( (2 * np.pi)**dims * np.linalg.det(covars[k]) )
        inv_covars[k] = np.linalg.inv(covars[k])

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


    cdef double* data_d = <double*>data.data
    cdef double* means_d = <double*>means.data
    cdef double* coefs_d = <double*>coefs.data
    cdef double* inv_covars_d = <double*>inv_covars.data
    cdef double* pk_d = <double*>pk.data

    cdef double* z_d = <double *> z.data

    cdef int dims = data.shape[1]


    cdef double nf


    cdef double* x = <double*>data.data
    cdef double* M
    cdef double* mu

    cdef double xm[5]
    cdef double dotdot
    cdef double wrk
    cdef int aj,ak 


    for j in xrange(datalen):
        nf = 0.0
        M = inv_covars_d
        mu = means_d
        for i in xrange(n_mixture):
            dotdot = 0.0
            for ak in range(dims):
                xm[ak] = x[ak] - mu[ak]
            for aj in range(dims):
                for ak in range(dims):
                    dotdot += xm[aj] * xm[ak] * M[aj*dims+ak]
            # wrk = pk_d[i] * coefs_d[i] * exp(-.5 * dotdot)
            wrk = pk_d[i] * coefs_d[i] * fast_exp_mineiro(-.5 * dotdot)

            mu += dims
            M += dims*dims

            z_d[j*n_mixture+i] = wrk
            nf += wrk

        ## Normalize this line
        for i in xrange(n_mixture):
            z_d[j*n_mixture+i] /= nf
        x += dims

    ## Calculate new pk values
    for i in xrange(n_mixture):
        pk_d[i] = 0
        for j in xrange(datalen):
            pk_d[i] += z_d[j*n_mixture+i]
    nf = 0.0
    for i in xrange(n_mixture):
        nf += pk_d[i]
    for i in xrange(n_mixture):
        pk_d[i] /= nf

	#z = ndimage.filters.median_filter(z.reshape(1024,1360,3),(15,15,1)).reshape(-1,3)
	#z = ndimage.filters.maximum_filter(z.reshape(1024,1360,3),size=5).reshape(-1,3)


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
    cdef double class_sum[10]
        ## WATCH OUT assuming a maximum of 5 dimensions in problem
    cdef double xx[10]

    cdef double* d_data = <double*>data.data
    cdef double* m_data = <double*>means.data
    cdef double* z_data = <double*>z.data

    cdef double acc[100], zz


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
        for aj in xrange(10):
            for ak in xrange(10):
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
     np.ndarray[DTYPE_t, ndim=3] inv_covars,
     int Nlin, int Ncol):
    pdf_params(covars, coefs, inv_covars)
    cdef int cl = np.argmin(means[:,2])
    print 'Venação: ', cl
    for it in xrange(iter):
        EStep(n_mixture, data, means, covars, z, pk, coefs, inv_covars)        
        z_morph(z, cl, Nlin, Ncol)
        MStep(n_mixture, data, means, covars, z, pk)
        pdf_params(covars, coefs, inv_covars)


cdef extern double atan2(double,double)
cdef extern double floor(double)
cdef extern double ceil(double)
cdef extern double hypot(double,double)
cdef extern double sqrt(double)
cdef extern double log10(double)
cdef extern double exp(double)
cdef extern double fabs(double)

cdef extern float fast_exp_mineiro(float)
def py_fast_exp(x):
    return fast_exp_mineiro(x)







##############################################################################
# Morphological madness #
##############################################################################
cdef z_morph(np.ndarray[DTYPE_t, ndim=2] z, int cl, int Nlin, int Ncol):

    cdef int j, k, jj, kk, cc

    cdef double* z_d = <double *> z.data
    cdef double mmin,mmax,v

    cdef np.ndarray z_out = np.zeros((z.shape[0], z.shape[1]), dtype=DTYPE)
    cdef double* zo_d = <double *> z_out.data

    z_out[:] = z
    
    for j in xrange(5,Nlin-5):
        for k in xrange(5,Ncol-5):
            mmax = 0
            mmin=z_d[3*Ncol*(j)+3*(k)+cl]
            v = z_d[3*Ncol*(j+5)+3*(k)+cl]
            if v < mmin:
                mmin = v
            v = z_d[3*Ncol*(j-5)+3*(k)+cl]
            if v < mmin:
                mmin = v
            if mmin > mmax:
                mmax = mmin

            mmin=z_d[3*Ncol*(j)+3*(k)+cl]
            v = z_d[3*Ncol*(j)+3*(k+5)+cl]
            if v < mmin:
                mmin = v
            v = z_d[3*Ncol*(j)+3*(k-5)+cl]
            if v < mmin:
                mmin = v
            if mmin > mmax:
                mmax = mmin

            mmin=z_d[3*Ncol*(j)+3*(k)+cl]
            v = z_d[3*Ncol*(j+4)+3*(k+4)+cl]
            if v < mmin:
                mmin = v
            v = z_d[3*Ncol*(j-4)+3*(k-4)+cl]
            if v < mmin:
                mmin = v
            if mmin > mmax:
                mmax = mmin

            mmin=z_d[3*Ncol*(j)+3*(k)+cl]
            v = z_d[3*Ncol*(j+4)+3*(k-4)+cl]
            if v < mmin:
                mmin = v
            v = z_d[3*Ncol*(j-4)+3*(k+4)+cl]
            if v < mmin:
                mmin = v
            if mmin > mmax:
                mmax = mmin

            zo_d[3*Ncol*(j)+3*(k)+cl] = mmax

    ## Now go back to the first array, making a dilation
    for j in xrange(5,Nlin-5):
        for k in xrange(5,Ncol-5):
            mmax = 0.0
            for jj in xrange(-3,4):
                for kk in xrange(-3,4):
                    v = zo_d[3*Ncol*(j+jj)+3*(k+kk)+cl]
                    if v > mmax:
                        mmax = v
                    # mmax += zo_d[3*Ncol*(j+jj-2)+3*(k+kk-2)+cl]
            # z_d[3*Ncol*(j)+3*(k)+cl] = mmax/25.0
            z_d[3*Ncol*(j)+3*(k)+cl] = mmax
            # z_d[3*Ncol*(j)+3*(k)+cl] = zo_d[3*Ncol*(j)+3*(k)+cl]


    ## dilation on other channels
    for j in xrange(5,Nlin-5):
        for k in xrange(5,Ncol-5):
            for cc in range(3):
                if cc == cl:
                    continue
                mmax = 0.0
                for jj in xrange(-3,4):
                    for kk in xrange(-3,4):
                        v = z_d[3*Ncol*(j+jj)+3*(k+kk)+cc]
                        if v > mmax:
                            mmax = v
                zo_d[3*Ncol*(j)+3*(k)+cc] = mmax

    ## erosion on other channels
    for j in xrange(5,Nlin-5):
        for k in xrange(5,Ncol-5):
            for cc in range(3):
                if cc == cl:
                    continue
                mmax = 1000.0
                for jj in xrange(-3,4):
                    for kk in xrange(-3,4):
                        v = zo_d[3*Ncol*(j+jj)+3*(k+kk)+cc]
                        if v < mmax:
                            mmax = v
                z_d[3*Ncol*(j)+3*(k)+cc] = mmax


    ## Re-normalize z
    for j in xrange(5,Nlin-5):
        for k in xrange(5,Ncol-5):
            v = z_d[3*Ncol*(j)+3*(k)]+z_d[3*Ncol*(j)+3*(k)+1]+z_d[3*Ncol*(j)+3*(k)+2]
            z_d[3*Ncol*(j)+3*(k)] /= v
            z_d[3*Ncol*(j)+3*(k)+1] /= v
            z_d[3*Ncol*(j)+3*(k)+2] /= v
            
            ## f stuff is still in zo_d, copy it all back to z first
            # if cl != 0:
            #     zo_d[3*Ncol*(j)+3*(k)] = z_d[3*Ncol*(j)+3*(k)]
            # if cl != 1:
            #     zo_d[3*Ncol*(j)+3*(k)+1] = z_d[3*Ncol*(j)+3*(k)+1]
            # if cl != 2:
            #     zo_d[3*Ncol*(j)+3*(k)+2] = z_d[3*Ncol*(j)+3*(k)+2]

            # v = zo_d[3*Ncol*(j)+3*(k)]+zo_d[3*Ncol*(j)+3*(k)+1]+zo_d[3*Ncol*(j)+3*(k)+2]
            # zo_d[3*Ncol*(j)+3*(k)] /= v
            # zo_d[3*Ncol*(j)+3*(k)+1] /= v
            # zo_d[3*Ncol*(j)+3*(k)+2] /= v
            

    # z[:] = z_out


##############################################################################
# Python class #
##############################################################################
class EMGMM:

	def __init__(self, int n_mixture, np.ndarray[DTYPE_t, ndim=2] data, Nlin, Ncol):
		
		self.n_mixture = n_mixture
		self.data = data
		self.dim = data.shape[1]
		self.means = np.ones((n_mixture, self.dim))
		self.covars = np.ones((n_mixture, self.dim, self.dim))
		self.covars *= 1 * np.identity(self.dim)
		self.means = kmeans(n_mixture, data)[0]
		self.z = np.zeros((data.shape[0], n_mixture))

		self.pk = np.ones((n_mixture,1)) / n_mixture

		self.coefs = np.zeros((n_mixture,))
		self.inv_covars = np.zeros(self.covars.shape)

		self.Ncol = Ncol
		self.Nlin = Nlin

	def iterate(self, iter):
#		try:
		fit(iter, self.n_mixture, self.data, self.means, self.covars, self.z, self.pk,
				self.coefs, self.inv_covars, self.Nlin, self.Ncol)

##############################################################################


##############################################################################
# K means #
##############################################################################
cpdef kmeans(unsigned int n_clusters, np.ndarray[DTYPE_t, ndim=2] data):

    cdef np.ndarray[DTYPE_t, ndim=2] k = np.random.randint(floor(data.min()), 
        ceil(data.max()+1), size=(n_clusters, data.shape[1]))/1

    cdef np.ndarray[DTYPE_t, ndim=2] kn = np.zeros_like(k)
    cdef np.ndarray[DTYPE_t, ndim=2] c = np.zeros((data.shape[0], k.shape[0]))

    cdef unsigned int i

    while(True):
        
        c = distance3(k, data)
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
<<<<<<< HEAD
cpdef distance(np.ndarray[DTYPE_t, ndim=2] clusters, 
	np.ndarray[DTYPE_t, ndim=2] data):
	
	cdef unsigned int i
	cdef np.ndarray[DTYPE_t, ndim=2] c = np.zeros((clusters.shape[0], data.shape[0]))

	for i in xrange(clusters.shape[0]):
		c[i] = np.sqrt(((data-clusters[i])**2).sum(axis=1))
	return c.T

cpdef distance3(np.ndarray[DTYPE_t, ndim=2] clusters, 
	np.ndarray[DTYPE_t, ndim=2] data):
		
	cdef unsigned int i, j
	cdef double x1, x2, x3
	cdef np.ndarray[DTYPE_t, ndim=2] c = np.zeros((data.shape[0],3))

	for j in xrange(data.shape[0]):
		for i in xrange(3):
			x1 = (data[j,0]-clusters[i,0])*(data[j,0]-clusters[i,0])
			x2 = (data[j,1]-clusters[i,1])*(data[j,1]-clusters[i,1])
			x3 = (data[j,2]-clusters[i,2])*(data[j,2]-clusters[i,2])
			c[j,i] = sqrt(x1+x2+x2)

	return c


## Local Variables:
## fill-column: 79
## End:
