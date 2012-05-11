
import numpy as np
import kmeans
from pylab import *
#import matplotlib as mpl

count = 0

class EMGMM:

    def __init__(self, n_mixture, data):
        self.n_mixture = n_mixture
        self.data = data

        self.dim = data.shape[1]

        self.means = np.ones((n_mixture, self.dim))
        self.covars = np.ones((n_mixture, self.dim, self.dim))
        self.covars *= np.identity(self.dim)

        #self.means = np.random.randint(data.min()-1, data.max(), size=(n_mixture, data.shape[1])) / 1.0#kmeans.kmeans(n_mixture, data)[0]
        self.means = kmeans.kmeans(n_mixture, data)[0]

        self.z = np.zeros((len(data), self.n_mixture))

#        self.weights = np.ones((n_mixture))/n_mixture

    def EStep(self):
        for i in range(self.n_mixture):  # Dim
            for j in range(len(self.data)):  # dado
                self.z[j, i] = self.veross(i, j)
        self.z = (self.z.T / self.z.sum(axis=1)).T

    def MStep(self):
        newmi = np.zeros_like(self.means)
        newcov = np.zeros_like(self.covars)
        for i in range(self.n_mixture):
            for j in range(len(self.data)):
                newmi[i] += self.z[j, i] * self.data[j]
                xm = self.data[j] - self.means[i]
                newcov[i] += self.z[j, i] * np.outer(xm, xm)
            self.means[i] = newmi[i] / self.z[:, i].sum()
            self.covars[i] = newcov[i] / self.z[:, i].sum()

    def veross(self, i, j):
        x = self.data[j]
        mean = self.means[i]
        cov = self.covars[i]
        xm = x - mean
        a = np.exp(-.5 * np.dot(np.dot(xm, np.linalg.inv(cov)), xm))
        return 1 / (2 * np.pi * np.linalg.det(cov) ** 0.5) * a

    def fit(self, iter):
        global count
        for it in range(iter):
            try:
                self.EStep()
                self.MStep()
            except:
                print "Singular Covariance Matrix... Restarting!"
                count += 1
                self.__init__(self.n_mixture, self.data)
                self.fit(iter)
                break

mu1 = [1,1,1]
mu2 = [5,5,5]
mu3 = [-1,1,5]
mu4 = [2,20,-24]

cov1 = [[  1, -.5,  2],
        [-.5,   1, .6],
        [  2,  .6,  1]]

cov2 = [[  1,   5,  .2],
        [  5,   1, .98],
        [ .2, .98,   1]]

cov3 = [[  1, -.66,  0],
        [-.66,   1, -.6],
        [  0,  -.6,  1]]

cov4 = [[  2, -.26,  1],
        [-.26,   2, -.1],
        [  1,  -.1,  2]]


data1 = np.random.multivariate_normal(mu1, cov1, size=100)
data2 = np.random.multivariate_normal(mu2, cov2, size=100)
data3 = np.random.multivariate_normal(mu3, cov3, size=100)
data4 = np.random.multivariate_normal(mu4, cov4, size=100)

data = np.vstack((data1,data2,data3,data4))

def plotElipsoid():
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    coefs = (1, 2, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    # Radii corresponding to the coefficients:
    rx, ry, rz = [1/np.sqrt(coef) for coef in coefs]

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()


im = imread("/home/ftruzzi/abelhas/imagens/chaly/1_Euglossa_chaly_m.jpg")

R = im[:,:,0]
G = im[:,:,1]
B = im[:,:,2]

y = np.random.randint(0,1360,5000)
x = np.random.randint(0,1024,5000)

points = im[x,y].transpose()/256.
color = im[x,y]/256.
