##
##
## Fig 5 from the mansucript
##
## Endogenous Modulators of NMDA Receptors Control Dendritic Field Expansion of Cortical Neurons
##
## by Pascal Jorratt, Jan Ricny, Christian Leibold, and Saak V. Ovsepian
##
##





import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

cmap=plt.get_cmap('tab10')

#Parameters
Csp=1  # muF/cm2
S1=800 #mum2 Surface soma
S2=10000 #mum2 Surface dendrite
Csom=Csp*S1/100#pF
Cden=Csp*S2/100#pF
Ctotal=Csom+2*Cden
#
#

g0=1/300000 # specific transmebrane conductance
gsom=S1*g0 #(MOhms)^-1
gden=S2*g0 #(MOhms)^-1
gax=1/3 #(MOhms)^-1
gtotal=gsom+2*gden
tau_single=Ctotal/gtotal/1.0e3 #single compartment time const in ms
#

#connectivity between compartments
#
#model 1
M1=np.array([[-(gsom+2*gax)/Csom, gax/Csom, gax/Csom], [gax/Cden, -(gden+gax)/Cden, 0], [0, gax/Cden,-(gden+gax)/Cden]])*1.0e3

#model 2
M2=np.array([[-(gsom+gax)/Csom, gax/Csom, 0], [gax/Cden, -(gden+2*gax)/Cden, gax/Cden], [0, gax/Cden,-(gden+gax)/Cden]])*1.0e3

#### eigenvalue decompostion
lam1,B1=np.linalg.eig(M1)
lam2,B2=np.linalg.eig(M2)

# copy an dpaste spikes
def spikeshape(tspan):
    a=.2
    tau1=.1
    
    tspan=(tspan-tspan[0])/(tspan[-1]-tspan[0])
    return (1-tspan)*((1+a)*np.exp(-tspan/tau1)-a)
    
# Greens function (inhomogeneous solution)
def green(t,lam):
    v0=-(1-np.exp(lam[0]*t))/lam[0]
    v1=-(1-np.exp(lam[1]*t))/lam[1]
    v2=-(1-np.exp(lam[2]*t))/lam[2]
    return np.array([v0,v1,v2])
#
# homogeneous solution
def homogeneous(t,lam,B,V0):
    alpha=np.linalg.inv(B)@V0
    v0=np.exp(lam[0]*t)
    v1=np.exp(lam[1]*t)
    v2=np.exp(lam[2]*t)

    tmp=np.diag(alpha)@np.array([v0,v1,v2])

    return B@tmp



#
# wrapper
def simulate(T0,I0,B,lam,thresh,V0=np.zeros(3)):
    beta=np.linalg.inv(B)*np.array([I0,0,0])
    ref_period=2.5#ms
    dt=0.01
    
    iref=int(ref_period/dt)
    
    tarr=np.arange(0,T0,dt) 
    tact=0
    i0=0
    tspike=[]
    Vret=np.zeros((3,len(tarr)))
    while tact<T0:
        Vtmp=B@(np.diag(beta[:,0])@green(tarr,lam))+homogeneous(tarr,lam,B,V0)    
        idexc=np.where(Vtmp[0,:]>thresh)[0]
        if len(idexc)>0:
            tspike.append(tarr[min(idexc)]+tact)
            tact=tspike[-1]+ref_period
            imax=min([i0+min(idexc),Vret.shape[1]])
            #print(i0,min(idexc),tact,imax,len(tarr),Vtmp.shape)
            Vret[:,i0:imax]=Vtmp[:,0:(imax-i0)*(imax>i0)]
            if i0+min(idexc)<Vret.shape[1]:
                Vret[:,i0+min(idexc)-1]=np.ones(3)*100
                imax2=min([Vret.shape[1], i0+min(idexc)+iref])
                if imax2>=i0+min(idexc):
                    Vret[:,i0+min(idexc):imax2]=np.zeros((3,(imax2-i0-min(idexc))))
                    Vret[0,i0+min(idexc):imax2]=100*spikeshape(np.array(range(i0+min(idexc),imax2)))
                
            i0=i0+min(idexc)+iref
        else:
            tact=T0
            Vret[:,i0:]=Vtmp

    return tspike,tarr,Vret
        
#

###
###
### generate data for plotting
###
###for panelA
T0=10  # total duration
I0=100 # current clamp amplitude

tmp,tarr,V1=simulate(T0,I0,B1,lam1,115)
tmp,tarre,V1e=simulate(T0,0,B1,lam1,115,V1[:,-1])
tmp,tarr,V2=simulate(T0,I0,B2,lam2,115)
tmp,tarre,V2e=simulate(T0,0,B2,lam2,115,V2[:,-1])
# compute input resistances
print('R1=', V1[0,-1]/I0*1000, 'MOhm,  R2=', V2[0,-1]/I0*1000, 'MOhm,  tausingle=', tau_single, 'ms')


# point neuron model
passive=(1-np.exp(-tarr/tau_single))*I0*tau_single/Ctotal
passivee=np.exp(-tarre/tau_single)*passive[-1]
V1=np.append(V1,V1e,axis=1)
V2=np.append(V2,V2e,axis=1)
tarr=np.append(tarr,tarre+tarr[-1],axis=0)
single=np.append(passive,passivee,axis=0)


panelA={'V1':V1,'V2':V2,'tarr':tarr,'single':single}

###
###
###for panelB
T0=30# ms
I0=150#pA
iarr=np.arange(-100,175,25)
model1=[]
model2=[]
for I0 in iarr:
    tmp,tarr,V1=simulate(T0,I0,B1,lam1,10)
    tmp,tarr,V2=simulate(T0,I0,B2,lam2,10)
    model1.append(V1)
    model2.append(V2)

panelB={'iarr':iarr,'tarr':tarr,'model1':model1, 'model2':model2}

###
###
###for panelC

iarr=np.arange(50,200,5)

fI1=[]
fI2=[]
T0=200
for I0 in iarr:
    cnt1=simulate(T0,I0,B1,lam1,10)
    cnt2=simulate(T0,I0,B2,lam2,10)
    fI1.append(len(cnt1[0])/T0)
    fI2.append(len(cnt2[0])/T0)

panelC={'iarr':iarr,'fI1':fI1, 'fI2':fI2};


### save numerical results
sio.savemat('panelA.mat', panelA)
sio.savemat('panelB.mat', panelB)
sio.savemat('panelC.mat', panelC)

##########
##########
##########
##########
##########
#### do the plotting
##########
##########
##########
##########
panelA=sio.loadmat('panelA.mat')
panelB=sio.loadmat('panelB.mat')
panelC=sio.loadmat('panelC.mat')

dy=0.07
dx=0.1
posa=np.array([0.1,0.7,0.15,0.15])
posb=posa-np.array([0,posa[3]+dy*1.5,0,0])
posc=posb+np.array([posa[2]+1.5*dx,-.08,0.1,0.08])
posd=posc+np.array([posa[2]+dx*1.7,0,0,0])
pose=posb-np.array([0,posa[3]+.201,0,0])
posf=posc-np.array([0,posc[3]+dy*.58,0,.08])
posg=posd-np.array([0,posc[3]+dy*.58,0,.08])

fig=plt.figure()
ax=fig.add_axes(posa)

ax.plot(panelA['tarr'][0],panelA['V1'][0,:])
ax.plot(panelA['tarr'][0],panelA['V2'][0,:])
ax.plot(panelA['tarr'][0],panelA['single'][0])
ax.set_ylabel('V$_\mathregular{Soma}}$ (mV)')
ax.set_xlabel('Time (ms)')

ax=fig.add_axes(posb)
ax.plot(panelA['tarr'][0],panelA['V1'][0,:]/np.max(panelA['V1'][0,:]))
ax.plot(panelA['tarr'][0],panelA['V2'][0,:]/np.max(panelA['V2'][0,:]))
ax.plot(panelA['tarr'][0],panelA['single'][0]/np.max(panelA['single'][0]))
ax.set_xlabel('Time (ms)')
ax.set_xlim([-.125,1])
ax.set_ylim([0,.6])
ax.set_ylabel('V$_\mathregular{Soma}$/max')

ax1=fig.add_axes(posc)
ax2=fig.add_axes(posd)
for ni,I0 in enumerate(panelB['iarr'][0]):
    ax1.plot(panelB['tarr'][0],panelB['model1'][ni][0,:], color=cmap.colors[0])
    ax2.plot(panelB['tarr'][0],panelB['model2'][ni][0,:], color=cmap.colors[1])


ax1.set_ylabel('V$_\mathregular{Soma}$ (mV)')
ax1.set_xlabel('Time (ms)')
ax2.set_xlabel('Time (ms)')

ax1=fig.add_axes(posf)
ax2=fig.add_axes(posg)
for ni,I0 in enumerate(panelB['iarr'][0]):
    ax1.plot(panelB['tarr'][0],panelB['model1'][ni][0,:], color=cmap.colors[0])
    ax2.plot(panelB['tarr'][0],panelB['model2'][ni][0,:], color=cmap.colors[1])


ax1.set_ylabel('V$_\mathregular{Soma}$ (mV)')
ax1.set_xlabel('Time (ms)')
ax2.set_xlabel('Time (ms)')
ax1.set_xlim([0,5])
ax2.set_xlim([0,5])
ax1.set_ylim([-20,30])
ax2.set_ylim([-20,30])

ax=fig.add_axes(pose)
ax.plot(panelC['iarr'][0],np.array(panelC['fI1'][0]), '.-')
ax.plot(panelC['iarr'][0],np.array(panelC['fI2'][0]), '.-')
ax.set_ylabel('Spikes/ms')
ax.set_xlabel('Current (pA)')

plt.savefig('traces.pdf')
plt.show()

