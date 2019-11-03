import numpy as np
import scipy as sp 
from matplotlib import pyplot as plt

G=0.1 #gravitationnal constant
# epsilon=0.1 #prevent gravity to explose at short distance -> collision
# nparticles = 7
buls = np.zeros((nparticles,5))
#array with every bullets (5 component each: mass, x, y, vx, vy)
    
def calForceMesh(buls):
    grid = 100
    density, bx,by= np.histogram2d(buls[:,1],buls[:,2],bins=grid,range=[[0, 1], [0, 1]],weights=buls[:,0])
    frho = np.fft.fft2(density)
    ks = np.fft.fftfreq(grid)
    k2 = np.abs(ks)*np.abs(ks.reshape(-1,1))
    k2[:,0]=np.Inf
    k2[0,:]=np.Inf
    fphi = -G*frho/k2
    #fphi = np.nan_to_num(fphi) #try other values? 
    phi = np.real(np.fft.ifft2(fphi))
    #plt.figure(0)
    #plt.imshow(np.real(phi))
    #plt.figure(1)
    #plt.imshow(density)
    xgrad, ygrad = np.gradient(phi)
    forces = np.empty((nparticles,2))
    for i in range(len(buls)):
        x=buls[i,1]
        y=buls[i,2]
        digx = np.digitize(x,bx)
        digy = np.digitize(y,by)
        if digx>99:
            continue
        if digy>99:
            continue
        force = np.array([xgrad[digx,digy],ygrad[digy,digy]])
        forces[i]=force
    return forces
    
    
def takeStepRK4(dt, buls):
    '''Integration of d2y/dt2=f, here f is the acceleration'''
    M2=buls
    M2[:,1]+=(dt/2)*buls[:,3]
    M2[:,2]+=(dt/2)*buls[:,4]
    f2=calForceMesh(M2)/np.array([buls[:,0],buls[:,0]]).transpose()
    M3=buls
    M3[:,1]+=(dt/2)*buls[:,3]+(dt*dt/4)*f1[:,0]
    M3[:,2]+=(dt/2)*buls[:,4]+(dt*dt/4)*f1[:,1]
    f3=calForceMesh(M3)/np.array([buls[:,0],buls[:,0]]).transpose()
    M4=buls
    M4[:,1]+=(dt)*buls[:,3]+(dt*dt/2)*f2[:,0]
    M4[:,2]+=(dt)*buls[:,4]+(dt*dt/2)*f2[:,1]
    f4=calForceMesh(M4)/np.array([buls[:,0],buls[:,0]]).transpose()
    Mfinal=buls
    Mfinal[:,3]+=(dt/6)*(f1+2*f2+2*f3+f4)[:,0]
    Mfinal[:,4]+=(dt/6)*(f1+2*f2+2*f3+f4)[:,1]
    Mfinal[:,1]+=dt*buls[:,3]+((dt*dt/6)*(f1+f2+f3))[:,0]
    Mfinal[:,2]+=dt*buls[:,4]+((dt*dt/6)*(f1+f2+f3))[:,1]
    return Mfinal


T=20
t=0
dt=0.05

while (t<T): #Simulate from 0 to T
    buls = takeStepRK4(dt,buls)
    plt.clf()
    plt.axis([0,1,0,1])
    for bul in buls:
        plt.scatter(bul[1],bul[2])
    plt.pause(1e-3)
    t += dt
