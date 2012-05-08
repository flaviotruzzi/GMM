#!/usr/bin/env 
# cython: boundscheck=False
# cython. wraparound=False
# cython: cdivision=True
# cython: nonecheck(False)
# cython: infer_types(True)
# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cpdef KullbackLeiberDivergence(
    np.ndarray[DTYPE_t, ndim=1] u0, 
    np.ndarray[DTYPE_t, ndim=1] u1,
    np.ndarray[DTYPE_t, ndim=2] cov0,
    np.ndarray[DTYPE_t, ndim=2] cov1,    
    ):
  return .5 *( np.trace(np.dot(np.linalg.inv(cov1), cov0)) + np.dot(np.dot((u1-u0).T, np.linalg.inv(cov1)), (u1-u0)) - np.log(np.linalg.det(cov0)/np.linalg.det(cov1)))

cpdef mahalanobisDistance(
    np.ndarray[DTYPE_t, ndim=1] u0,
    np.ndarray[DTYPE_t, ndim=1] u1,
    np.ndarray[DTYPE_t, ndim=2] pooledCov
    ):
    return np.dot(np.dot(u0-u1, np.linalg.inv(pooledCov)), (u0-u1).T)

cpdef pooledCov(
    np.ndarray[DTYPE_t, ndim=2] cov1,
    np.ndarray[DTYPE_t, ndim=2] cov2
    ):
    return (cov1+cov2)/2

def correspondence(
    np.ndarray[DTYPE_t, ndim=2] means1,
    np.ndarray[DTYPE_t, ndim=2] means2,
    np.ndarray[DTYPE_t, ndim=3] covars1,
    np.ndarray[DTYPE_t, ndim=3] covars2
  ):
  cdef unsigned int n_mixture = means1.shape[0]
  cdef unsigned int i, j 
  cdef np.ndarray[DTYPE_t, ndim=2] presult = np.zeros((n_mixture,n_mixture))

  for i in xrange(n_mixture):
    for j in xrange(n_mixture):
      presult[i,j] = mahalanobisDistance(means1[i],means2[j],pooledCov(covars1[i],covars2[j]))

  return presult.argmin(axis=1)
