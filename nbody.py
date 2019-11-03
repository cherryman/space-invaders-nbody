import numpy as np 
from matplotlib import pyplot as plt

G=0.001 #gravitationnal constant
nparticles=10
pos = np.random.rand(nparticles,2)
vel = (np.random.rand(nparticles,2)*2-1)*0
m = np.random.rand(nparticles)
    
def calForceMesh(x,m):
    grid = 100
    density,bx,by= np.histogram2d(x[:,0],x[:,1],bins=grid,range=[[0, 1], [0, 1]],weights=m)
    frho = np.fft.fft2(density)
    ks = np.fft.fftfreq(grid)
    k2 = np.abs(ks)*np.abs(ks.reshape(-1,1))
    k2[:,0]=np.Inf
    k2[0,:]=np.Inf
    fphi = -G*frho/k2
    #fphi = np.nan_to_num(fphi) #try other values? 
    phi = np.real(np.fft.ifft2(fphi))
    plt.figure(0)
    plt.imshow(np.real(phi))
    plt.figure(1)
    plt.imshow(density)
    xgrad, ygrad = np.gradient(phi)
    forces = np.empty((len(x),2))
    for i in range(len(x)):
        ex=x[i,0]
        y=x[i,1]
        digx = np.digitize(ex,bx)
        digy = np.digitize(y,by)
        if digx>99:
            continue
        if digy>99:
            continue
        force = np.array([xgrad[digx,digy],ygrad[digy,digy]])
        forces[i]=force
    return forces
    
    
def takeStepRK4(dt,pos,v,m):
    
    '''Integration of d2y/dt2=f, here f is the acceleration'''
    x=pos[:,0];y=pos[:,1];vx=v[:,0];vy=v[:,1];
    pos1=np.empty((len(x),2))
    pos1[:,0]=x
    pos1[:,1]=y
    f1=calForce(pos1, m)/np.array([m,m]).transpose()
    #M2=buls
    x2=x+(dt/2)*vx
    y2=y+(dt/2)*vy
    pos2=np.empty((len(x),2))
    pos2[:,0]=x2
    pos2[:,1]=y2
    f2=calForce(pos2,m)/np.array([m,m]).transpose()
    #M3=buls
    x3=x+(dt/2)*vx+(dt*dt/4)*f1[:,0]
    y3=y+(dt/2)*vy+(dt*dt/4)*f1[:,1]
    pos3=np.empty((len(x),2))
    pos3[:,0]=x3
    pos3[:,1]=y3
    f3=calForce(pos3, m)/np.array([m,m]).transpose()
    #M4=buls
    x4=x+(dt)*vx+(dt*dt/2)*f2[:,0]
    y4=y+(dt)*vy+(dt*dt/2)*f2[:,1]
    pos4=np.empty((len(x),2))
    pos4[:,0]=x4
    pos4[:,1]=y4
    f4=calForce(pos4,m)/np.array([m,m]).transpose()
    #Mfinal=buls
    vxfinal=vx+(dt/6)*(f1+2*f2+2*f3+f4)[:,0]
    vyfinal=vy+(dt/6)*(f1+2*f2+2*f3+f4)[:,1]
    xfinal=x+dt*vx+((dt*dt/6)*(f1+f2+f3))[:,0]
    yfinal=y+dt*vy+((dt*dt/6)*(f1+f2+f3))[:,1]
    posfinal=np.empty((len(x),2))
    vfinal=np.empty((len(x),2))
    posfinal[:,0]=xfinal
    posfinal[:,1]=yfinal 
    vfinal[:,0]=vxfinal
    vfinal[:,1]=vyfinal
    return posfinal,vfinal

def takeStepEuler(dt,pos,vel,m):
    f1=calForceMesh(pos, m)/np.array([m,m]).transpose()
    vf=vel+dt*f1
    xf=pos+dt*vf
    print(pos,vel)
    return xf,vf
    
def gravity(p1,p2,m1,m2):
    # gravitational force of bul2 on bul1 
    epsilon = 0.1
    x1=p1[0];x2=p2[0];y1=p1[1];y2=p2[1]
    num = G*m1*m2
    den = ((x1-x2)**2+(y1-y2)**2+epsilon**2)**(3/2)
    return np.array([-(num/den)*(x1-x2),-(num/den)*(y1-y2)])

def calForce(x,m): #calculate force between 2 particles
    nparticles = len(x)
    forces=np.zeros((nparticles,2))
    for i in range(nparticles):
        for j in range(nparticles):
            if (i!=j):
                forces[i]=gravity(x[i],x[j],m[i],m[j]) #here just the gravity
    return forces


T=20
t=0
dt=0.05

while (t<T): #Simulate from 0 to T
    pos,vel = takeStepRK4(dt,pos,vel,m)
    plt.clf()
    plt.axis([0,1,0,1])
    for p in pos:
        plt.scatter(p[0],p[1])
    plt.pause(1e-3)
    t += dt
    
    

    
