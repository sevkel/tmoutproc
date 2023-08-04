import numpy as np
import tmoutproc as top
import matplotlib.pyplot as plt
from os import path

plt.rcParams.update({'font.size':15})

def read_xyz_path_file(filename):
    """
    load xyz path file and returns data as coord_xyz.
    coord[i,:,:] ... Entries for structure i
    coord[:,:,i] ... Entries for atom i
    coord[:,i,:] ... Atoms types (str) [i=0], x (float) [i=1], y (float) [i=2], x (float) [z=3] for all atoms

    Args:
        param1 (String): filename

    Returns:
        coord_xyz, header (optional)
    """

    datContent = [i.strip().split() for i in open(filename).readlines()]
    Natoms=int(datContent[0][0])
    Ngeos=int(len(datContent)/(Natoms+2))

    energies=np.zeros(Ngeos)

    for igeo in range(Ngeos):
        datContGeo=np.array(datContent[(2+Natoms)*igeo+2:(2+Natoms)*(igeo+1)],dtype=object)
        for i, item in enumerate(datContGeo):
            datContGeo[i, 1] = float(item[1])
            datContGeo[i, 2] = float(item[2])
            datContGeo[i, 3] = float(item[3])
        coord_geo=np.transpose(datContGeo).reshape((1,4,Natoms))
        if (igeo==0):
            coord_path=coord_geo
        else:
            coord_path=np.append(coord_path,coord_geo,axis=0)

        #energies:
        energies[igeo]=float(datContent[(2+Natoms)*igeo+1][2])

    return coord_path,energies

def thermal_average(quantity,energy,temp=300):
    '''
    calculates thermal average in canonical ensemble
    over different structures with observables given by quantity
    and energies given by energy

    input
    quantity: array, float 
    energy: array, float (unit: Hartree)
    temp: float (unit: Kelvin)

    output:
    thavg: float (same units as quantity)

    '''

    # local energy array (to increase precision)
    emin=min(energy)
    eloc=energy-emin

    # canonical partition function
    Zk=np.sum(np.exp(-eloc/(top.KBOLTZ*temp)))
    # sum over qantity with energy weights
    Qs=np.sum(quantity*np.exp(-eloc/(top.KBOLTZ*temp)))
    # average
    thavg=Qs/Zk
    
    return thavg


def bonding_angle(geom,indices):

    if(len(indices)!=3):
        print("You need three atoms to calculate a bonding angle!")
        return 400
    else:
        if(len(np.shape(geom))==2):
            v1=geom[1:,indices[0]]-geom[1:,indices[1]]
            v2=geom[1:,indices[2]]-geom[1:,indices[1]]
            #print(np.shape(v2))
            #print(v2)
        elif(len(np.shape(geom))==3):
            # not yet working for paths
            return 400
            v1=geom[:,1:,indices[0]]-geom[:,1:,indices[1]]
            v2=geom[:,1:,indices[2]]-geom[:,1:,indices[1]]
            #print(np.shape(v2))
            #print(v2[0])
            #print(np.linalg.norm(v2,axis=0))
            #print(np.linalg.norm(v2,axis=1))
    
    n1=np.linalg.norm(v1)
    n2=np.linalg.norm(v2)
    cophi=np.sum(v1*v2,axis=-1)/(n1*n2)
    phi=np.arccos(cophi)*180./np.pi

    return phi

def energy_barrier(energies,regular=True,mloc=False):
    '''
    calculate energy barriers from path energies
    Input:
    param1 : energies
    param2: operation mode
    '''

    if (regular):
        ebar=top.Ha2eV*(np.amax(energies)-energies[0])
        if (mloc):
            barloc=np.argmax(energies)
            return ebar,barloc
        else:
            return ebar
    else:
        ebar=top.Ha2eV*(max(energies)-min(energies))
        return ebar
    

def find_index_by_element(cdata,elstr,ltrue=True):
    '''
    find all atoms for given element
    input:
    cdata as read from xyz
    elstr, string e.g. 'S','Au',...

    output:
    list of atom indices
    '''

    nat=len(cdata[0])
    ia=((cdata[0]==elstr)==ltrue)
    atlout=np.arange(nat)[ia]
    
    return atlout

runlist=[]
faillist=[]
geolist=[1,10,20,30,40,50,60,70,80,90]
bars=np.zeros(shape=(len(geolist),3))
sdist=np.zeros(shape=(len(geolist)))
for ig,g in enumerate(geolist):
    print(g)
    geopath='geom_'+str(g)
    # read relaxed structure to get S-S-distance
    relgeo=top.read_xyz_file(geopath+'/init_loop/geom_relaxed.xyz')
    iS=find_index_by_element(relgeo,'S')
    sdist[ig]=np.sum((relgeo[1:,iS[1]]-relgeo[1:,iS[0]])**2,axis=-1)**(1/2)
    for ia,axis in enumerate(['s','z','gold']):
        axispath=geopath+'/'+axis+'axis_wlfl/'
        if (axis=='gold'):
            axispath=geopath+'/'+axis+'_axis/'
       # if ((axis=='z') and (g==10)):
       #     axispath=geopath+'/'+axis+'axis/'
        if path.exists(axispath+"WOELFLING_CONVERGED"):
            print(axis+ " woelfling calculation converged")
            pathgeos,energies=read_xyz_path_file(axispath+'path.xyz')
            bars[ig,ia]=energy_barrier(energies,False)
        elif path.exists(axispath+"WOELFLING_RUNNING"):
            print(axis+" woelfling calculation running")
            runlist=np.append(runlist,str(g)+axis)
            bars[ig,ia]=np.nan
        elif path.exists(axispath+"WOELFLING_FAILED"):
            print(axis+" woelfling calculation failed")
            faillist=np.append(faillist,str(g)+axis)
            bars[ig,ia]=np.nan
        else: 
            print(axis+" No woelfling information found")
            bars[ig,ia]=np.nan

print(bars)
legendarry=[str(geolist[i]) + '\n' + str(round(sdist[i],2)) + r'$\AA$' for i in range(len(geolist))]
print(legendarry)
plt.title("rotation barriers (HH)")
plt.plot(geolist,bars[:,0],'bo',label='S-axis')
plt.plot(geolist,bars[:,1],'ro',label='z-axis')
plt.plot(geolist,bars[:,2],'go',label='Au-axis')
plt.xticks(ticks=geolist[::2],labels=legendarry[::2])
plt.xlabel("geometry")
plt.ylabel("barrier height (eV)")
plt.legend(loc="upper right")
plt.savefig('bardist.png',bbox_inches='tight',pad_inches=.3)

print('Currently running:')
print(runlist)
print('Failed:')
print(faillist)

'''

ic=find_index_by_element(pathgeos[0],'C')
iN=find_index_by_element(pathgeos[0],'N')
iO=find_index_by_element(pathgeos[0],'O')
# Bond length
dist=np.sum((pathgeos[:,1:,iN[0]]-pathgeos[:,1:,ic[0]])**2,axis=-1)**(1/2)
plt.plot(10*np.arange(len(dist)),dist,'o')
plt.ylabel("C-N distance ($\AA$)")
plt.xlabel('rotation angle')
plt.xticks(np.arange(0,190,20))
plt.savefig('ncdist.png',bbox_inches='tight',pad_inches=.3)
plt.clf()

# Bond length (thermal avg)
T=[0.01,0.1,1.,10,100.,300.,1000.,1.E4,1.E5,1.E6]
da=[]
for t in T:
    dist_avg=thermal_average(dist[:-1],energies[:-1],t)
    da=np.append(da,dist_avg)
# Bond length
plt.plot(T,da,'o')
plt.title("Average bondlength")
plt.ylabel("C-N distance ($\AA$)")
plt.xscale('log')
plt.xlabel('Temperature (K)')
plt.xticks(T)
plt.savefig('ncdist_avg.png',bbox_inches='tight',pad_inches=.3)
plt.clf()
plt.cla()


# Energy barrier
plt.title('Energy barrier')
plt.plot(30*np.arange(len(energies)),top.Ha2eV*(energies-energies[0]),'o')
ebar,iebar=energy_barrier(energies,True,True)
plt.vlines(30*iebar,0,ebar)
plt.text(30*iebar,ebar/2,str(round(ebar,3))+' eV',rotation='vertical',va='center',ha='center',backgroundcolor='white')
plt.ylabel("Energy (eV)")
plt.xlabel('rotation angle')
plt.xticks(np.arange(0,360,60))
plt.savefig('ebar.png',bbox_inches='tight',pad_inches=.3)
plt.clf()
plt.cla()

# convergence plot
gradient=top.read_plot_data('grad_conv.dat')[0]
# barrier convergence
ip=1
barriers=[]
while path.exists("path-"+str(ip)+".xyz"):
#    print(ip)
#    print("we have that file")
    pathgeos,energies=read_xyz_path_file("path-"+str(ip)+".xyz")
    ebar,iebar=energy_barrier(energies,True,True)
    barriers=np.append(barriers,ebar)
    ip+=1

fig,ax1=plt.subplots()
ax2=ax1.twinx()
plt.title("Convergence")
ax1.plot(np.arange(len(gradient)),gradient,color='blue')
ax2.plot([],[],color='blue',label='gradient')
ax2.plot(np.arange(len(barriers))+1,barriers,color='red',label='barriers')
ax1.set_xlabel('path number')
ax1.set_ylabel("gradient (RMS)")
ax2.set_ylabel("barrier height (eV)")
plt.legend(loc="upper right")
plt.savefig('convergence.png',bbox_inches='tight',pad_inches=.5)
plt.clf()
plt.cla()

# O-N-O bond angle
indices=[iO[0],iN[0],iO[1]]
phivec=[]
for i in range(path.shape[0]):
    phivec=np.append(phivec,bonding_angle(path[i],indices))

plt.plot(10*np.arange(len(phivec)),phivec,'o')
plt.ylabel("O-N-O angle")
plt.xlabel('rotation angle')
plt.xticks(np.arange(0,190,20))
plt.savefig('onoangl.png')
'''
