import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

######HALDANE MODEL######

#parameter list
j = complex(0,1)
a1 = (1/2) * np.array([-(3**(1/2)),3])
a2 = (1/2) * np.array([3**(1/2),3])     #a1,a2 are primitive vectors of the two-dimensional lattice
t = 1                                   #hopping integral for the nearest neighbours
l = 0                                   #hopping integral for the next neighbours
V = 0.1                                 #Staggerd potencial
delta = 0.05                            #lattice discretization
K = 4*np.pi*3**(1/2)/9
M = np.pi/(3**(1/2))                    #locations of the vertexes of points M, K in the first Brillouin zone
Elist1 = []
Elist2 = []
kxlist=[]
kylist=[]
dx=[]
dy=[]
dz=[]
Flist = []
a = np.array([0,1.5])
n = 0
c = 0
temp = []                                #temp list for heatmap plotting purposes
pathx = []
pathy = []
E1 = []
E2 = []
count = []
Fl = []


#Hamiltonian definition
def H(k):
    aa = l * j * (-np.exp(-j * np.vdot(k, (a2 - a1))) + np.exp(-j * np.vdot(k, (a1 - a2))) + np.exp(-j * np.vdot(k, a2)) -
         np.exp(-j * np.vdot(k, a1)) + np.exp(j * np.vdot(k, a1)) - np.exp(j * np.vdot(k, a2))) + V
    ab = t * (1 + np.exp(j * np.vdot(k, a1)) + np.exp(j * np.vdot(k, a2)))
    ba = t * (1 + np.exp(-j * np.vdot(k, a1)) + np.exp(-j *np.vdot(k, a2)))
    bb = l * j * (np.exp(j * np.vdot(k, (a2 - a1))) - np.exp(j * np.vdot(k, (a1 - a2))) - np.exp(j * np.vdot(k, a2)) +
         np.exp(-j * np.vdot(k, a1)) + np.exp(j * np.vdot(k, a1)) - np.exp(-j * np.vdot(k, a2))) - V
    return np.array([[aa, ab], [ba, bb]])

def eigval(k):
    return LA.eigvalsh(H(k))

def eigvec(k):
    return LA.eigh(H(k))[1]

#vector d(k) representing the bulk momentum-space Hamiltonian, normed to 1 definition
def vectorD(k):
    Dx = t*(1 + np.cos(np.dot(k,a1)) + np.cos(np.dot(k,a2)))
    Dy = -t*(np.sin(np.dot(k,a1))+np.sin(np.dot(k,a2)))
    Dz = -l*2*(np.sin(np.dot(k,(a2-a1)))+np.sin(np.dot(k,a1))-np.sin(np.dot(k,a2))) + V
    D = np.array([Dx,Dy,Dz])
    norm = (Dx**2+Dy**2+Dz**2)**(1/2)
    return D/norm

#path from point K to point M of the Brillouin zone definition
def ky_KM(kx):
    return -3**(1/2)*kx+4*np.pi/3

#path from point M to point Gamma of the Brillouin zone definition
def ky_MG(kx):
    return 3**(1/2)*kx/3


#Berry phase definition
def BerryPhase(k1,k2,k3,k4):
    braket12 = np.vdot(eigvec(k1)[0, n], eigvec(k2)[0, n])+np.vdot(np.exp(-j*np.dot(k1,a2))*eigvec(k1)[1, n], np.exp(-j*np.dot(k1,a2))*eigvec(k2)[1, n])
    braket23 = np.vdot(eigvec(k2)[0, n], eigvec(k3)[0, n])+np.vdot(np.exp(-j*np.dot(k1,a2))*eigvec(k2)[1, n], np.exp(-j*np.dot(k1,a2))*eigvec(k3)[1, n])
    braket34 = np.vdot(eigvec(k3)[0, n], eigvec(k4)[0, n])+np.vdot(np.exp(-j*np.dot(k1,a2))*eigvec(k3)[1, n], np.exp(-j*np.dot(k1,a2))*eigvec(k4)[1, n])
    braket41 = np.vdot(eigvec(k4)[0, n], eigvec(k1)[0, n])+np.vdot(np.exp(-j*np.dot(k1,a2))*eigvec(k4)[1, n], np.exp(-j*np.dot(k1,a2))*eigvec(k1)[1, n])
    ln = np.angle(braket12 * braket23 * braket34 * braket41 / np.abs(braket12 * braket23 * braket34 * braket41))
    return ln / (LA.norm(k1 - k2) * LA.norm(k3 - k4))


#collecting data in the bounds of the first Brillouin zone, Hamiltonian eigenvalues, Berry phase path, path of the endpoints of vector D
for kx in np.arange(-np.pi,np.pi,delta):
    x = []
    for ky in np.arange(-np.pi, np.pi,delta):
        k = np.array([kx,ky])
        kylist.append(ky)
        kxlist.append(kx)
        k1 = np.array([kx, ky])
        k2 = np.array([kx - delta, ky])
        k3 = np.array([kx - delta, ky - delta])
        k4 = np.array([kx, ky - delta])
        Flist.append(BerryPhase(k1, k2, k3, k4))
        Elist1.append(eigval(k)[0])
        Elist2.append(eigval(k)[1])
        dx.append(np.real(vectorD(k)[0]))
        dy.append(np.real(vectorD(k)[1]))
        dz.append(np.real(vectorD(k)[2]))
        x.append(BerryPhase(k1, k2, k3, k4))
    temp.append(x)
        
c=0
#collecting Chern number data
for ky in np.arange(-2*np.pi/3,0,delta):
    for kx in np.arange(-kx1(ky),kx1(ky),delta):
        k1 = np.array([kx, ky])
        k2 = np.array([kx - delta, ky])
        k3 = np.array([kx - delta, ky - delta])
        k4 = np.array([kx, ky - delta])
        ChernNum = BerryPhase(k1, k2, k3, k4) * (LA.norm(k1 - k2) * LA.norm(k3 - k4))
        c += ChernNum
        kxlist.append(kx)
        kylist.append(ky)
for ky in np.arange(0,2*np.pi/3,delta):
    for kx in np.arange(-kx2(ky),kx2(ky),delta):
        k1 = np.array([kx, ky])
        k2 = np.array([kx - delta, ky])
        k3 = np.array([kx - delta, ky - delta])
        k4 = np.array([kx, ky - delta])
        ChernNum = BerryPhase(k1, k2, k3, k4) * (LA.norm(k1 - k2) * LA.norm(k3 - k4))
        c += ChernNum
        kxlist.append(kx)
        kylist.append(ky)
        
        
i=0
#collecting data of the wavevectors going through the endpoints of the points of the first Brillouin zone in the honeycomb lattice
for kx in np.arange(0,K,d):
    k = np.array([kx,0])
    k1 = np.array([kx, 0])
    k2 = np.array([kx - delta, 0])
    k3 = np.array([kx - delta, 0 - delta])
    k4 = np.array([kx, 0 - delta])
    Fl.append(BerryPhase(k1, k2, k3, k4))
    E1.append(eigval(k)[0])
    E2.append(eigval(k)[1])
    pathx.append(kx)
    pathy.append(0)
    count.append(i)
    i=i+1
for kx in reversed(np.arange(M,K,d)):
    k = np.array([kx, ky_KM(kx)])
    k1 = np.array([kx, ky_KM(kx)])
    k2 = np.array([kx - delta, ky_KM(kx)])
    k3 = np.array([kx - delta, ky_KM(kx) - delta])
    k4 = np.array([kx, ky_KM(kx) - delta])
    Fl.append(BerryPhase(k1, k2, k3, k4))
    E1.append(eigval(k)[0])
    E2.append(eigval(k)[1])
    pathx.append(kx)
    pathy.append(ky_KM(kx))
    count.append(i)
    i = i + 1
for kx in reversed(np.arange(0,M,d)):
    k = np.array([kx,ky_MG(kx)])
    k1 = np.array([kx, ky_MG(kx)])
    k2 = np.array([kx - delta, ky_MG(kx)])
    k3 = np.array([kx - delta, ky_MG(kx) - delta])
    k4 = np.array([kx, ky_MG(kx) - delta])
    Fl.append(BerryPhase(k1, k2, k3, k4))
    E1.append(eigval(k)[0])
    E2.append(eigval(k)[1])
    pathx.append(kx)
    pathy.append(ky_MG(kx))
    count.append(i)
    i = i + 1

#plotting the band structure
plt.figure(1)
plt.subplot(211)
plt.plot(count,E2,'k-')
plt.plot(count,E1,'k-')
plt.subplot(212)
plt.plot(pathx,pathy,'m-',linewidth=1.2)


plt.figure(2)
ax = plt.axes(projection='3d')
ax.scatter3D(kxlist, kylist, Elist1,c=Elist1, cmap='coolwarm',linewidth=0.1)
ax.scatter3D(kxlist, kylist, Elist2,c=Elist1, cmap='coolwarm',linewidth=0.1)
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_zlabel('E')


#plotting the D vector
plt.figure(3)
ax = plt.axes(projection='3d')
ax.scatter3D(dx,dy,dz,c=dz, cmap='plasma',linewidth=0.1)
ax.set_xlabel('dx')
ax.set_ylabel('dy')
ax.set_zlabel('dz')
plt.title('Vector D')
plt.show()

#plotting the Berry phase
plt.figure(1)
plt.subplot(211)
plt.plot(count,Fl,'k-')
plt.plot(count,Fl,'k-')
plt.subplot(212)
plt.plot(pathx,pathy,'m-',linewidth=1.2)


plt.figure(2)
ax = plt.axes(projection='3d')
ax.scatter3D(kxlist,kylist,Flist,c=Flist, cmap='rainbow',linewidth=0.1)
plt.figure(3)
im = plt.imshow(np.array(np.transpose(temp)), cmap='rainbow', extent=(-3.14,3.14,3.14,-3.14),interpolation='bilinear')
plt.colorbar(im)
plt.show()

