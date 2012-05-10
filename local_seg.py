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


immu = zeros((3,frame.shape[0],frame.shape[1],3))/3.0
for cl in range(3):
    immu[cl,:,:,:] = mu[cl]

zz = ones((frame.shape[0],frame.shape[1],3))/3.0
newzz = zeros((frame.shape[0],frame.shape[1],3))
err = zeros((3,frame.shape[0],frame.shape[1],3))
masked_frame = zeros((3,frame.shape[0],frame.shape[1],3))

## Calcualte residues fmo each class
for cl in range(3):
#    err[cl,:,:,:] = frame-mu[cl]
    err[cl] = frame-immu[cl]

print 70*'='
print pc
print mu
print covs

for ii in range(3):
    newzz[:] = 0
    ######################
    # E Step
    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
            for cl in range(3):
                zz[j,k,cl] *= pcoef[cl] * exp( -0.5 * dot(err[cl,j,k],dot(icovs[cl], err[cl,j,k])))
                #zz[j,k,cl] = pc[cl] * pcoef[cl] * exp( -0.5 * dot(err[cl,j,k],dot(icovs[cl], err[cl,j,k])))

    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
            ## Perform smoothing
            # for cl in range(2):
            #     for jj in range(-1,2):
            #         for kk in range(-1,2):
            #             newzz[j,k,cl] += zz[clip(j-jj,0,frame.shape[0]-1),
            #                                 clip(k-kk,0,frame.shape[1]-1),cl]
            #     newzz[j,k,cl] /= 9
            for cl in range(2):
                newzz[:,:,cl] = zz[:,:,cl]
                
            ## Perform grayscale dilation
            for cl in range(2,3):
                newzz[j,k,cl] = zz[clip(j-1,0,frame.shape[0]-1):clip(j+2,0,frame.shape[0]-1),
                                   clip(k-1,0,frame.shape[1]-1):clip(k+2,0,frame.shape[1]-1),cl].max()


    zz[:] = newzz

    zz+=1e-10


    ## Nomralize class probabilities for each point
    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
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

    for cl in range(3):
        masked_frame[cl] = (zz[:,:,cl].reshape(-1,1) * frame.reshape(-1,3)).reshape(frame.shape)
    ## The means
    immu[:] = 0
    immu[:] = masked_frame


    nn = 3
    for j in xrange(frame.shape[0]-nn+1):
        for k in xrange(frame.shape[1]-nn+1):            
            for jj in xrange(nn):
                for kk in xrange(nn):
                    for cl in range(3):
                        immu[cl,j+nn/2,k+nn/2] += masked_frame[cl,j+jj,k+kk]

    for j in xrange(frame.shape[0]-nn+1):
        for k in xrange(frame.shape[1]-nn+1):            
            for cl in range(3):
                soma = zz[j+nn/2,k+nn/2,cl]
                for jj in xrange(nn):
                    for kk in xrange(nn):
                        soma += zz[j+jj,k+kk,cl]
                immu[cl,j+nn/2,k+nn/2] /= soma


    # mu[:] = 0
    # for j in xrange(frame.shape[0]):
    #     for k in xrange(frame.shape[1]):
    #         for cl in range(3):
    #             mu[cl] += zz[j,k,cl] * frame[j,k]
    # for cl in range(3):
    #     mu[cl] /= pc[cl]

    ## Calculate the residues from each class
    for cl in range(3):
        err[cl,:,:,:] = frame-immu[cl,:,:]
        # err[cl,:,:,:] = frame-mu[cl]

    ## The covariances
    covs[:] = 0
    for j in xrange(frame.shape[0]):
        for k in xrange(frame.shape[1]):
            for cl in range(3):
                covs[cl] += zz[j,k,cl] * outer(err[cl,j,k],err[cl,j,k])
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
