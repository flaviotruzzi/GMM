# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:58:05 2012

@author: ftruzzi
"""
import numpy as np
from pylab import *


def kmeans(n_clusters, training_set, init_clusters=None):

    if init_clusters == None:
        k = np.random.randint(np.floor(training_set.min()-1), np.ceil(training_set.max()), size=(n_clusters, training_set.shape[1])) / 1.0
    else:
        k = init_clusters

    kn = zeros_like(k)
    while(True):

        c = distance(k,training_set)
        print c.shape
        c[:,0] = c.argmin(axis=1)
        print c.shape
        for n,i in enumerate(k):
            if (len(training_set[c[:,0]==n])>0):
                i = training_set[c[:,0]==n].sum(axis=0)/(len(training_set[c[:,0]==n]))
                kn[n] = i
        
        if (len(where( (kn==k) == False)[0]) == 0):
            break
        else:
            k = kn
    c = distance(k,training_set)
    c[:,0] = c.argmin(axis=1)
    return [k,c[:,0]]


def distance(clusters,data):
    c = np.zeros((len(data),len(clusters)))
    for n, i in enumerate(clusters):                         
        c.T[n] = sqrt(((data-i)**2).sum(axis=1))    
    return c


def euclidian_distance(p0,p1):
    return np.sqrt((p0[1]-p1[1])**2 + (p0[0]-p1[0])**2)


