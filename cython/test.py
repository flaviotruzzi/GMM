import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import EMGMM
import sys, os
from pylab import *
import Image
from scipy import ndimage
from multiprocessing import Pool
import time



basedir = "imagens/"

files = open("imagens.txt").read().split("\n")
files.remove('')

def process(file):
  d = file.strip(".jpg")
  #if (os.path.exists(basedir + d + '/' + d[d.find('/')+1:] + ".npz") == False):
  if (os.path.exists(basedir + d + '/') == False):
    os.mkdir(basedir + d)
    os.mkdir(basedir + d + "/fillholes/")

  im = asarray(Image.open(basedir + file).resize((1360/2,1024/2),Image.ANTIALIAS))
  im2 = im.copy().reshape(-1,3)
  emm = EMGMM.EMGMM(4, im2/256.)
  emm.iterate(10)
  labels = emm.z.argmax(axis=1)

  np.savez(basedir + d + '/' + d[d.find('/')+1:] + ".npz",means=emm.means,covars=emm.covars)

  for i in xrange(4):
    figure(i)
    imsave(basedir + d + '/' + repr(i) + '.png',labels.reshape(im.shape[0],im.shape[1])==i,cmap=cm.bone)
    imsave(basedir + d + '/fillholes/' + repr(i) + '.png',ndimage.morphology.binary_fill_holes(labels.reshape(im.shape[0],im.shape[1])==i),cmap=cm.bone)
  print "Processed: ", d
  time.sleep(1)



pool = Pool(4)
result = pool.map(process,files)
