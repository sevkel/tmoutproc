"""
Manipulating coord files in xyz and turbomole format
"""
__docformat__ = "google"
#__all__
import numpy as np
import fnmatch
import scipy.sparse
import re as r
from scipy.linalg import eig
from functools import partial
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from . import constants
from . import io as io

__ang2bohr__ = 1.88973


def shift_xyz_coord(coord_xyz, x_shift, y_shift, z_shift):
    """
    Shifts coordinates in coord file loaded with read_xyz_file.

    Args:
        coord_xyz: Coord file loaded with read_xyz_file
        x_shift: x shift in Angstrom
        y_shift: y shift in Angstrom
        z_shift: z shift in Angstrom

    Returns:
        coord_xyz: Shifted xyz coord file
    """

    for i in range(0, len(coord_xyz[1, :])):
        coord_xyz[1, i] = round(float(coord_xyz[1, i]) + x_shift, 5)
        coord_xyz[2, i] = round(float(coord_xyz[2, i]) + y_shift, 5)
        coord_xyz[3, i] = round(float(coord_xyz[3, i]) + z_shift, 5)
    return coord_xyz


def shift_coord_file(coord, x_shift, y_shift, z_shift):
    """
    Shifts coordinates in coord file loaded with io.read_coord_file.

    Args:
        coord_xyz: Coord file loaded with io.read_coord_file
        x_shift: x shift in Bohr
        y_shift: y shift in Bohr
        z_shift: z shift in Bohr

    Returns:
        coord: Shifted coord file
    """

    for i in range(0, coord.shape[1]):
        coord[0, i] = coord[0, i] + x_shift
        coord[1, i] = coord[1, i] + y_shift
        coord[2, i] = coord[2, i] + z_shift

    return coord


def sort_xyz_coord(coord_xyz, axis):
    """
    Sort coord_xyz according to axis (x=0, y=1, z=2)

    Args:
        coord_xyz: Coord file loaded with read_xyz_file
        axis: Axis (x=0, y=1, z=2)

    Returns:
        coord_xyz: Sorted coord_xyz
    """
    allowed = [0, 1, 2]
    if (axis not in allowed):
        raise ValueError(f"Axis {axis} not valid.")
    coord_xyz = np.transpose(coord_xyz)
    coord_xyz = coord_xyz[coord_xyz[:, axis + 1].argsort()]
    coord_xyz = np.transpose(coord_xyz)

    return coord_xyz


def sort_coord(coord, axis):
    """
    Sort coord_xyz according to axis (x=0, y=1, z=2)

    Args:
        coord_xyz: Coord file loaded with read_xyz_file
        axis: Axis (x=0, y=1, z=2)

    Returns:
        coord_xyz: Sorted coord_xyz
    """
    allowed = [0, 1, 2]
    if (axis not in allowed):
        raise ValueError(f"Axis {axis} not valid.")

    coord = np.transpose(coord)
    coord = sorted(coord, key=lambda item: item[axis])
    coord = np.transpose(coord)

    return coord

def align_molecule(coord_xyz, axis, molecule_axis):
    """
    Aligns molecule stored in coord_xyz along axis. Molecule_axis[0/1] give the indices of molecules which define the molecule
    axis. Please note: coord_xyz = top.read_xyz_file(filename)

    Args:
        coord_xyz (np.ndarray): coord_xyz file read with top from xyz file (coord_xyz = op.read_xyz_file(filename)))
        axis (np.array): axis is aligned along axis -> axis[x,y,z]
        molecule_axis (np.array): indices of atoms in coord_xyz which define the molecule axis (index1, index2)


    Returns:
        coord_xyz: Rotated coord_xyz file
    """


    # calculate vec(molecule_axis[0]->molecule_axis[1])
    x_left = coord_xyz[1,molecule_axis[0]]
    y_left = coord_xyz[2,molecule_axis[0]]
    z_left = coord_xyz[3,molecule_axis[0]]

    # shift to orgin
    for j in range(0, coord_xyz.shape[1]):
        coord_xyz[1,j] = coord_xyz[1,j] - x_left
        coord_xyz[2,j] = coord_xyz[2,j] - y_left
        coord_xyz[3,j] = coord_xyz[3,j] - z_left

    x_left = 0.0
    y_left = 0.0
    z_left = 0.0

    x_right = coord_xyz[1, molecule_axis[1]]
    y_right = coord_xyz[2, molecule_axis[1]]
    z_right = coord_xyz[3, molecule_axis[1]]

    molecule_axis = [x_right - x_left, y_right - y_left, z_right - z_left]

    # calculate rotation angel

    #angle = np.arccos(molecule_axis[0] / np.linalg.norm(molecule_axis))
    angle = np.arccos(np.dot(molecule_axis, axis) / (np.linalg.norm(molecule_axis)*np.linalg.norm(axis)))
    theta = angle

    if (angle != 0):
        # calculate rotation axis

        rotation_axis = np.cross(molecule_axis, [axis[0], axis[1], axis[2]])
        rotation_axis = 1.0 / np.linalg.norm(rotation_axis) * rotation_axis
        u = rotation_axis
        # print("rotation axis " + str(rotation_axis))

        # calculate rotation_matrix
        rotation_matrix = [
            [np.cos(theta) + u[0] ** 2 * (1 - np.cos(theta)), u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta),
             u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)],
            [u[0] * u[1] * (1 - np.cos(theta)) + u[2] * np.sin(theta), np.cos(theta) + u[1] ** 2 * (1 - np.cos(theta)),
             u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)],
            [u[0] * u[2] * (1 - np.cos(theta)) - u[1] * np.sin(theta),
             u[1] * u[2] * (1 - np.cos(theta)) + u[0] * np.sin(theta), np.cos(theta) + u[2] ** 2 * (1 - np.cos(theta))]]

        for j in range(0, coord_xyz.shape[1]):
            vector_to_rotate = [coord_xyz[1,j], coord_xyz[2,j],coord_xyz[3,j]]
            rotated_vector = np.asmatrix(rotation_matrix) * np.asmatrix(vector_to_rotate).T

            coord_xyz[1, j] = rotated_vector[0, 0]
            coord_xyz[2, j] = rotated_vector[1, 0]
            coord_xyz[3, j] = rotated_vector[2, 0]
        return coord_xyz
    else:
        return coord_xyz


def x2t(coord_xyz):
    """
    Transforms coord_xyz from xyz format to turbomole

    Args:
        coord_xyz: coord array in xyz

    Returns:
        coord (np.ndarray): coord in turbomole format
    """
    coord = np.empty((5,coord_xyz.shape[1]), dtype=object)
    for j in range(0, coord_xyz.shape[1]):
        coord[0, j] = coord_xyz[1, j] * constants.ANG2BOHR
        coord[1, j] = coord_xyz[2, j] * constants.ANG2BOHR
        coord[2, j] = coord_xyz[3, j] * constants.ANG2BOHR
        coord[3, j] = coord_xyz[0,j].lower()
        coord[4, j] = ""

    return coord

def t2x(coord):
    """
    Transforms turbomole coord to coord_xyz

    Args:
        coord (np.ndarray): coord

    Returns:
        coord_xyz (np.ndarray): coord_xyz
    """
    coord_xyz = np.empty((4,coord.shape[1]), dtype=object)
    for j in range(0, coord.shape[1]):
        coord_xyz[1, j] = coord[0, j] / constants.ANG2BOHR
        coord_xyz[2, j] = coord[1, j] / constants.ANG2BOHR
        coord_xyz[3, j] = coord[2, j] / constants.ANG2BOHR
        coord_xyz[0, j] = coord[3, j].capitalize()


    return coord_xyz

def fix_atoms(coord, indices, unfix_first = False):
    """
    fixes atoms in indices in turbomole coord file coord

    Args:
        coord: coord file read with top.io.read_coord_file
        indices: array-like of indices in coord file which should be fixed. "all" is also possible -> every atom is fixed. Other option is "invert" -> fixes are inverted
        unfix_first: if True, all atoms are unfixed before the atoms in indices are fixed

    Returns:
    coord with fixed atoms
    """
    if(unfix_first):
        coord[4,:] = ""

    if(indices == "all"):
        coord[4, :] = "f"
    elif(indices == "invert"):
        #if coord[4, :] == "f" set as "" and if coord[4, :] == "" set as "f"
        for i in range(coord.shape[1]):
            if(coord[4,i] == "f"):
                coord[4,i] = ""
            else:
                coord[4,i] = "f"
    else:
        if isinstance(indices, int) or (isinstance(indices, (np.ndarray, list)) and (
                all(isinstance(item, int) for item in indices) or all(
                isinstance(item, np.int32) for item in indices) or all(
                isinstance(item, np.int64) for item in indices))):
            coord[4,indices] = "f"
        else:
            raise ValueError(f"indices must be array-like of int or 'all' or 'invert'. Not {type(indices)} with {type(indices[0])}")

    return coord

def remove_fixed_atoms_from_coord(coord):
    """
    Removes fixed atoms from turbomole coord file.

    Args:
        coord: coord file data loaded with io.read_coord_file

    Returns:
        coord: filtered coord file data
    """
    coord_filtered = list()
    for i in range(coord.shape[1]):
        if(coord[4,i] == "f"):
            continue
        else:
            coord_filtered.append(coord[:,i])
    coord_filtered = np.asarray(coord_filtered,dtype=object)
    coord_filtered = np.transpose(coord_filtered)
    return coord_filtered


