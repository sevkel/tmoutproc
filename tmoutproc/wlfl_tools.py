# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:07:28 2021

@author: leonm
"""
import tmoutproc as top
import numpy as np

def rotate_molparts(cdata,axis_atoms,rotated_atoms,phi):

    import copy
    # devide into pos and el:
    pos=cdata[1:,:]
    el=cdata[0,:]

    
    if type(axis_atoms)==type(str('z')):
        # rotation around given axis
        # so far only z-axis
        print('we do not have axis atoms')
        # set rotation axis to z-axis
        b1=np.array([0,0,1])
        p_shifted=pos[:,rotated_atoms]
        latoms=False
    else:
        print('we have axis atoms given')
        print(axis_atoms)
        # get rotation axis from axis_atoms, normalize
        v1=pos[:,axis_atoms[1]]-pos[:,axis_atoms[0]]
        b1=v1.reshape(3)/np.linalg.norm(v1)
        ## for safety reasons: remove axis atoms from rotated atoms
        ## (also simplifies work to define rotated_atoms
        # true if atom index is none of the axis atoms indices
        # false if one of the two axis atoms
        ia=((rotated_atoms!=axis_atoms[0])*(rotated_atoms!=axis_atoms[1]))
        # reduce to atoms that are not axis atoms
        ra=rotated_atoms[ia]
        rotated_atoms=ra
        latoms=True

        ## set first axis atom as new coordinate system origin
        p_shifted=pos[:,rotated_atoms]-pos[:,axis_atoms[0]].reshape((3,-1))
    

    ## component parallel to rotation axis
    C1=np.matmul(b1,p_shifted)

    ## remove parallel contribution
    Nm=len(rotated_atoms) #number of rotated atoms
#    print(Nm,C1.shape,p_shifted.shape,b1.shape)
    pos2=p_shifted-np.dot(b1.reshape((3,-1)),C1.reshape((-1,Nm)))

    ## first orthogonal vector
    v2=pos2[:,0]
    b2=v2.reshape(3)/np.linalg.norm(v2)
    ## first orthogonal component
    C2=np.matmul(b2,pos2)
    ## remove first orthogonal contribution
    pos3=pos2-np.dot(b2.reshape((3,-1)),C2.reshape((-1,Nm)))

    ## second orthogonal vector (cross product)
#    print(b1,b1.shape)
#    print(b2,b2.shape)
    b3=np.cross(b1.reshape((1,3)),b2.reshape((1,3))).reshape(3)
#    print(b3,b3.shape)
    ## second orthogonal component
    C3=np.matmul(b3,pos3)
    ## remove second orthogonal contribution (should become zero)
    pos4=pos3-np.dot(b3.reshape((3,-1)),C3.reshape((-1,Nm)))
    if (np.sum(pos4*pos4)>=Nm*1.E-9):
        print('This should be small, please check:')
        print(np.sum(pos4))


    ## TEST: rebuild original coordinates
    pos_rebuild=np.dot(b1.reshape((3,-1)),C1.reshape((-1,Nm)))
    pos_rebuild+=np.dot(b2.reshape((3,-1)),C2.reshape((-1,Nm)))
    pos_rebuild+=np.dot(b3.reshape((3,-1)),C3.reshape((-1,Nm)))
    if(latoms):
        pos_rebuild+=pos[:,axis_atoms[0]].reshape((3,-1))
    diff=pos[:,rotated_atoms]-pos_rebuild
    if (np.sum(diff*diff)>=Nm*1.E-9):
        print('This should be small, please check:')
        print(np.sum(pos4))

    ## Rotation
    phi_rad=phi*2.*np.pi/360.
    C2p=C2*np.cos(phi_rad)-C3*np.sin(phi_rad)
    C3p=C2*np.sin(phi_rad)+C3*np.cos(phi_rad)

    ## Build new coordinates
    # the deepcopy is needed to avoid that the input pos is changed as well
#    pos_new=copy.deepcopy(pos)
#    pos_new[:,rotated_atoms]=np.dot(b1.reshape((3,-1)),C1.reshape((-1,Nm)))
#    pos_new[:,rotated_atoms]+=np.dot(b2.reshape((3,-1)),C2p.reshape((-1,Nm)))
#    pos_new[:,rotated_atoms]+=np.dot(b3.reshape((3,-1)),C3p.reshape((-1,Nm)))
#    if(latoms):
#        pos_new[:,rotated_atoms]+=pos[:,axis_atoms[0]].reshape((3,-1))

    # define output cdata with original Elements and new positions
    cdata_out = copy.deepcopy(cdata)
    cdata_out[1:,rotated_atoms]=np.dot(b1.reshape((3,-1)),C1.reshape((-1,Nm)))
    cdata_out[1:,rotated_atoms]+=np.dot(b2.reshape((3,-1)),C2p.reshape((-1,Nm)))
    cdata_out[1:,rotated_atoms]+=np.dot(b3.reshape((3,-1)),C3p.reshape((-1,Nm)))
    if(latoms):
        cdata_out[1:,rotated_atoms]+=pos[:,axis_atoms[0]].reshape((3,-1))

    return cdata_out


def build_rotation_path(cdata,axis_atoms,rotated_atoms,phi_start,phi_end,steps,filename):
    # change in phy per step 
    dphi=(phi_end-phi_start)/steps
    # loop over steps, from n=0 (phi=phi_start) to steps+1 (phi_end)
    for n in range(steps+1):
        # current rotantion angle phi
        phi=phi_start+n*dphi
        # do rotation 
        cdata_rot=rotate_molparts(cdata,axis_atoms,rotated_atoms,phi)
        # write xyz to file
        top.write_xyz_file(filename+'.xyz',cdata_rot,'','a')
        # create coord file
        coordn=top.x2t(cdata_rot)
        # fix gold atoms in coord file
        coordn_fix=fix_default_gold(coordn,cdata_rot)
        # write coord to file
        top.write_coord_file('coords_'+filename,coordn_fix,'a')
    return 0

def fix_default_gold(coorddata,cdata):

    '''
    In our standard Au20-tips, both TT and HH have Au layers:
        10-6-3-(1)
    last two layers are always 16 Au atoms
    If coorddata is sorted along z-axis (or at least gold part is),
    the fixed Au atoms are the first 16 and last 16 gold atoms in 
    coorddata
    '''

    # find all Au atoms:
    i_au=find_index_by_element(cdata,'Au')
    # take first 16 and last 16
    i_auf=np.append(i_au[:16],i_au[-16:])
    # fix gold atoms
    coorddata_fixed=top.fix_atoms(coorddata,i_auf)
    return coorddata_fixed


def dihedral_angle(cdata,atoms):

    u1=np.array(cdata[1:,atoms[1]]-cdata[1:,atoms[0]])
    u2=np.array(cdata[1:,atoms[2]]-cdata[1:,atoms[1]])
    u3=np.array(cdata[1:,atoms[3]]-cdata[1:,atoms[2]])
    u1n=np.array([u1[0],u1[1],u1[2]])
    u2n=np.array([u2[0],u2[1],u1[2]])
    print(u1.shape,u1n.shape)
    print(u1,u1n)
    v12=np.cross(u1n,u2n)
#    v23=np.cross(u2,u3)
#    print(v12,v23)
#    cophi=np.sum(v12*v23)/(norm(v12)*norm(v23))
#    siphi=norm(u2)*np.sum(u1*v23)/(norm(v12)*norm(v23))

#    phi1=np.arccos(cophi)*180./np.pi
#    phi2=np.arcsin(siphi)*180./np.pi

    phi1=90.
    phi2=0.
    return phi1, phi2


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


#  read file
filename='geom_relaxed.xyz'
cdata=top.io.read_xyz_file(filename)

#rotation fixpoints: (atom indices)
rotfix=find_index_by_element(cdata,'S')
#rotated region: full molecule
rotated=find_index_by_element(cdata,'Au',False)

# initial path for s-s-rotation
build_rotation_path(cdata,rotfix,rotated,0.,360.,12,'mypath-S')
# initial path for z-axis rotation
build_rotation_path(cdata,'z',rotated,0.,360.,12,'mypath-z')
