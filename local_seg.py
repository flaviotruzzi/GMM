from pylab import *
#coding: utf-8
import Image

#frame = array(Image.open('/home/nlw/ciencia/DADOS/abelhas/imagens/5_Euglossa_flammea_f.jpg'))
frame = array(Image.open('/home/nlw/ciencia/DADOS/abelhas/imagens/000.png'), dtype='float64')
frame = frame / 256.0

icovs = zeros((3,3,3))
pcoef = zeros(3)

mu = array([[ 0.77927603,  0.78362593,  0.79794092],
            [ 0.69904539,  0.66562307,  0.61600602],
            [ 0.47165596,  0.39739396,  0.36424941]])

covs = array([[[ 0.00092635,  0.00082778,  0.00047509],
               [ 0.00082778,  0.00075577,  0.00044785],
               [ 0.00047509,  0.00044785,  0.00035288]],
              
              [[ 0.00247591,  0.00278306,  0.00287095],
               [ 0.00278306,  0.00325617,  0.00358592],
               [ 0.00287095,  0.00358592,  0.00474468]],
              
              [[ 0.01265017,  0.01236573,  0.00587713],
               [ 0.01236573,  0.01249504,  0.00626174],
               [ 0.00587713,  0.00626174,  0.00391714]]])

pc = array([ 0.51482206,  0.39778619,  0.08739175])







for cl in range(3):
    icovs[cl] = inv(covs[cl])
    pcoef[cl] = 1.0/ sqrt( (2*pi) ** 3 * det(covs[cl]) )

zz = zeros((frame.shape[0],frame.shape[1],3))
err = zeros((3,frame.shape[0],frame.shape[1],3))

## Calcualte residues fmo each class
for cl in range(3):
    err[cl,:,:,:] = frame-mu[cl]

print 70*'='
print pc
print mu
print covs

for ii in range(3):
    ######################
    # E Step
    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
            for cl in range(3):
                zz[j,k,cl] = pc[cl] * pcoef[cl] * exp( -0.5 * dot(err[cl,j,k],dot(icovs[cl], err[cl,j,k])))
            ## Nomralize class probabilities for each point
            zz[j,k,:] /= zz[j,k,:].sum()

    if ii == 0:
        zz1 = copy(zz)


    ###########
    # M Step

    ## The class probabilities
    pc = zz.reshape(-1,3).sum(0)
    #pc /= pc.sum()
    if pc.min() == 0:
        break
    
    ## The means
    mu[:] = 0
    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
            for cl in range(3):
                mu[cl] += zz[j,k,cl] * frame[j,k]
    for cl in range(3):
        mu[cl] /= pc[cl]

    ## Calculate the residues from each class
    for cl in range(3):
        err[cl,:,:,:] = frame-mu[cl]

    ## The covariances
    covs[:] = 0
    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
            xx = outer(err[cl,j,k],err[cl,j,k])
            for cl in range(3):
                covs[cl] += zz[j,k,cl] * xx
    for cl in range(3):
        covs[cl] /= pc[cl]
        icovs[cl] = inv(covs[cl])
        pcoef[cl] = 1.0/ sqrt( (2*pi) ** 3 * det(covs[cl]) )

    #pc = zz.reshape(-1,3).sum(0)
    pc /= pc.sum()


    print 70*'-'
    print ii
    print pc
    print mu
    print covs




ion()

figure(1)
subplot(2,2,1)
imshow(frame/256.0, vmin=0, vmax=1, cmap=cm.bone)
for cl in range(3):
    subplot(2,2,2+cl)
    imshow(zz1[:,:,cl], vmin=0, vmax=1, cmap=cm.bone)

figure(2)
subplot(2,2,1)
imshow(frame/256.0, vmin=0, vmax=1, cmap=cm.bone)
for cl in range(3):
    subplot(2,2,2+cl)
    imshow(zz[:,:,cl], vmin=0, vmax=1, cmap=cm.bone)
