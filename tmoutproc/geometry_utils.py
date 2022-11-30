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
	Shifts coordinates in coord file loaded with load_xyz_file.
	Args:
		coord_xyz: Coord file loaded with load_xyz_file
		x_shift: x shift in Angstrom
		y_shift: y shift in Angstrom
		z_shift: z shift in Angstrom

	Returns:
		coord_xyz: Shifted xyz coord file
	"""

    try:
        if len(coord_xyz[1, :]) == 0:
            raise ValueError("coord file is empty")
    except IndexError as e:
        raise ValueError(f"Is it a xyz file? {e}")
    except TypeError as e:
        raise ValueError(f"Is it a xyz file? {e}")

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

    if len(coord) == 0:
        raise ValueError("coord file is empty")

    for i in range(0, len(coord)):
        coord[i][0] = round(float(coord[i][0]) + x_shift, 5)
        coord[i][1] = round(float(coord[i][1]) + y_shift, 5)
        coord[i][2] = round(float(coord[i][2]) + z_shift, 5)
    return coord


def sort_xyz_coord(coord_xyz, axis):
    """
	Sort coord_xyz according to axis (x=0, y=1, z=2)
	Args:
		coord_xyz: Coord file loaded with load_xyz_file
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
		coord_xyz: Coord file loaded with load_xyz_file
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

def align_molecule(coord, axis, molecule_axis):
	"""
	Aligns molecule stored in coord along axis. Molecule_axis[0/1] give the indices of molecules which define the molecule
	axis. Please note: coord = top.load_xyz_file(filename)
	Args:
		coord: coord file read with top from xyz file (coord = op.load_xyz_file(filename)))
		axis: axis is aligned along axis -> axis[x,y,z]
		molecule_axis: indices of atoms in coord which define the molecule axis (index1, index2)


	Returns:
	Rotated coord file
	"""


	# calculate vec(molecule_axis[0]->molecule_axis[1])
	x_left = round(float(coord[molecule_axis[0]][1]), 5)
	y_left = round(float(coord[molecule_axis[0]][2]), 5)
	z_left = round(float(coord[molecule_axis[0]][3]), 5)

	# shift to orgin
	for j in range(0, len(coord)):
		coord[j][1] = round(float(coord[j][1]), 5) - x_left
		coord[j][2] = round(float(coord[j][2]), 5) - y_left
		coord[j][3] = round(float(coord[j][3]), 5) - z_left

	x_left = 0.0
	y_left = 0.0
	z_left = 0.0

	x_right = round(float(coord[molecule_axis[1]][1]), 5)
	y_right = round(float(coord[molecule_axis[1]][2]), 5)
	z_right = round(float(coord[molecule_axis[1]][3]), 5)

	molecule_axis = [x_right - x_left, y_right - y_left, z_right - z_left]

	# calculate rotation angel

	angle = np.arccos(molecule_axis[0] / np.linalg.norm(molecule_axis))
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

		for j in range(0, len(coord)):
			vector_to_rotate = [round(float(coord[j][1]), 5), round(float(coord[j][2]), 5),
								round(float(coord[j][3]), 5)]
			rotated_vector = np.asmatrix(rotation_matrix) * np.asmatrix(vector_to_rotate).T

			coord[j][1] = round(rotated_vector[0, 0], 5)
			coord[j][2] = round(rotated_vector[1, 0], 5)
			coord[j][3] = round(rotated_vector[2, 0], 5)
		return coord
	else:
		return coord

def x2t(coord_xyz):
	"""
	Transforms coord_xyz from xyz format to turbomole
	Args:
		coord_xyz: coord array in xyz

	Returns:
	coord in turbomole format
	"""
	coord = np.empty(coord_xyz.shape[0], dtype=object)
	for j in range(0, len(coord_xyz)):
		x = round(float(coord_xyz[j][1]), 5)
		y = round(float(coord_xyz[j][2]), 5)
		z = round(float(coord_xyz[j][3]), 5)
		element = coord_xyz[j][0].lower()
		coord[j] = list([x*__ang2bohr__, y*__ang2bohr__, z*__ang2bohr__, element])

	return coord

def fix_atoms(coord, indices):
	"""
	fixes atoms in indices in turbomole coord file coord
	Args:
		coord: coord file read with top.io.read_coord_file
		indices: array-like of indices in coord file which should be fixed. "all" is also possible -> every atom is fixed

	Returns:
	coord with fixed atoms
	"""
	if(indices == "all"):
		for j in range(0, len(coord)):
			if(len(coord[j])==4):
				new = list(coord[j])
				new.append("f")
				coord[j] = new
	else:
		for j in indices:
			if(len(coord[j])==4):
				new = list(coord[j])
				new.append("f")
				coord[j] = new

	return coord

def remove_fixed_atoms(coord):
	"""
	Removes fixed atoms from turbomole coord file.
	Args:
		coord: coord file data loaded with io.read_coord_file

	Returns:
		coord: filtered coord file data
	"""
	coord_filtered = list()
	for i, item in enumerate(coord):
		if(len(item)<5):
			coord_filtered.append(item)
	return np.array(coord_filtered)
