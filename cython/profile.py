#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile, Image, os
from pylab import *

# import pyximport
# pyximport.install()

import EMGMM

#im = array(Image.open("/home/nlw/ciencia/DADOS/abelhas/imagens/5_Euglossa_flammea_f.jpg"))
im = array(Image.open(os.getenv("HOME") + "/ciencia/DADOS/abelhas/imagens/001.png"))

#data = im.reshape(-1,3)[:,2:3]/256.0
data = im.reshape(-1,3)[:,0:3]/256.0
a =  EMGMM.EMGMM(3, data, im.shape[0], im.shape[1])
aini = copy(a.means)

# a.iterate(1)
# ast1 = copy(a.means)

cProfile.runctx("a.iterate(10)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

# raise Exception
# a.iterate(10)


ion()

if data.shape[1]==1:
    bb = mgrid[0:256:2]/256.
    hh = histogram(a.data[:,0], bb)[0]
    hh  = hh / bb[1] / hh.sum()
    bar(bb[:-1], hh, width=2/256., color='b')

    bbb = mgrid[0:1.0:0.001]
    for kk in range(3):
        xx = bbb-a.means[kk]
        yy = a.pk[kk] * a.coefs[kk] * exp(-0.5*xx**2*a.inv_covars[kk][0][0] )
        plot(bbb,yy,'r-', lw=2)

    title('Histograma do canal azul + GMM ajustado')

## Plota um risco sobre as médias iniciais e finais
# for m in aini:
#     plot([m,m], [0,hh.max()], 'k--')
# for m in ast1:
#     plot([m,m], [0,hh.max()], 'r--')
# for m in a.means:
#     plot([m,m], [0,hh.max()], 'r-')

else:

    figure(2)
    subplot(2,2,1)
    imshow(im)
    subplot(2,2,2)
    imshow(a.z.reshape(im.shape[0], im.shape[1],3)[:,:,0], cm.bone)
    subplot(2,2,3)
    imshow(a.z.reshape(im.shape[0], im.shape[1],3)[:,:,1], cm.bone)
    subplot(2,2,4)
    imshow(a.z.reshape(im.shape[0], im.shape[1],3)[:,:,2], cm.bone)

