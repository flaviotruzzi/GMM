#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile, Image
from pylab import *

import pyximport
pyximport.install()

import EMGMM

im = Image.open("/home/ftruzzi/abelhas/imagens/chaly/1_Euglossa_chaly_m.jpg")
im = array(im.resize((170,128),Image.ANTIALIAS))

a =  EMGMM.EMGMM(3, array([(im[:,:,2]/256.).flatten()]).T)


cProfile.runctx("a.iterate(1)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()