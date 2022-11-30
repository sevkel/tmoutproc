"""
Methods usefull for calculations using turbomole output
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


def diag_F(f_mat_path, s_mat_path, eigenvalue_list=list()):
    """
    diagonalizes f mat (generalized), other eigenvalues can be used (eigenvalue_list). Solves Fx=l*S*x
    Args:
        param1 (string): filename of fmat
        param2 (string): filename of smat
        param3 (list()): eigenvalue_list (if specified eigenvalues are taken from eigenvalue_list)

    Returns:
        fmat (np.ndarray), eigenvalues (np.array), eigenvectors (matrix)

    """

    print("read f")
    F_mat_file = io.read_packed_matrix(f_mat_path)
    print("read s")
    s_mat = io.read_packed_matrix(s_mat_path)
    s_mat_save = s_mat

    print("diag F")

    # Konfiguration 1 : Übereinstimmung
    # eigenvalues, eigenvectors = np.linalg.eigh(F_mat_file.todense())

    # Konfiguration 2 : keine Übereinstimmung (sollte eigentlich das gleiche machen wie np.linalg.eigh)
    # eigenvalues, eigenvectors = scipy.linalg.eigh(F_mat_file.todense())

    # Konfiguration 3: keine Übereinstimmung (generalierstes EW Problem)
    eigenvalues, eigenvectors = scipy.linalg.eigh(F_mat_file.todense(), s_mat.todense())

    eigenvectors = np.real(np.asmatrix(eigenvectors))
    # eigenvectors = np.transpose(eigenvectors)
    eigenvalues = np.real(eigenvalues)
    eigenvalue_output_list = eigenvalues
    # print(eigenvalues)
    # print("type random")
    # print(type(eigenvalues).__name__)
    # print(type(eigenvectors).__name__)
    print("diag F done")

    print("calc fmat ")
    # take eigenvalues from diagonalization or external eigenvalues (e.g from qpenergies)
    if (len(eigenvalue_list) == 0):
        eigenvalues = np.diag(eigenvalues)
    else:
        eigenvalues = np.diag(eigenvalue_list)
        eigenvalue_output_list = eigenvalue_list

    sc = s_mat * eigenvectors
    f_mat = eigenvalues * np.transpose(sc)
    f_mat = sc * f_mat
    print("calc fmat done")

    return f_mat, eigenvalue_output_list, eigenvectors


def find_c_range_atom(atom, number, coordfile, basis_set="dev-SV(P)"):
    """
    finds atoms range of expansion coef in mos file. number gives wich atoms should be found e.g. two sulfur : first number =1; second = 2

    Args:
        param1 (String): atom type
        param2 (int): number of atom type atom
        param3 (String): filename for coord file
        param4 (String): basis_set="dev-SV(P)"


    Returns:
        tuple (index_start, index_end)
    """

    def atom_type_to_number_coeff(atom):
        """
        returns ordinal number of atom

        Args:
            param1 (String): atom type

        Returns:
            ordinal number (int)
            """
        if (atom == "c"):
            return 14
        elif (atom == "h"):
            return 2
        elif (atom == "au"):
            return 25
        elif (atom == "s"):
            return 18
        elif (atom == "zn"):
            return 24
        elif (atom == "o"):
            return 14
        elif (atom == "fe"):
            return 24
        elif (atom == "f"):
            return 14
        elif (atom == "cl"):
            return 18
        elif (atom == "br"):
            return 32
        elif (atom == "i"):
            return 13

        else:
            raise ValueError('Sorry. This feature is not implemented for following atom: ' + atom)

    coord_content = io.read_coord_file(coordfile)
    if (basis_set == "dev-SV(P)"):
        number_found_atoms_of_type = 0
        number_expansion_coeff = 0
        for i in range(0, len(coord_content)):
            current_atom = coord_content[i][3]

            if (current_atom == atom):
                number_found_atoms_of_type += 1
                if (number_found_atoms_of_type == number):
                    return (number_expansion_coeff, number_expansion_coeff + atom_type_to_number_coeff(current_atom))
                number_expansion_coeff += atom_type_to_number_coeff(current_atom)
            else:
                number_expansion_coeff += atom_type_to_number_coeff(current_atom)
        raise ValueError('Sorry. Could not find the atom')


    else:
        raise ValueError('Sorry. This feature is not implemented for following basis_set: ' + basis_set)


def atom_weight(atom, u2kg=False):
    """
    Return mass of atom in kg or au (au is default)
    Args:
        param1 (String) : Atom type
        param2 (boolean) : convert to kg

    Returns:
        float
    """
    atom = atom.lower()

    u = constants.U2KG
    if (u2kg == False):
        u = 1
    if (atom == "test"):
        return 1.0
    try:
        dict_entry = constants.ATOM_DICT_SYM[atom]
        atom_weight = dict_entry[2]
    except KeyError as e:
        error_text = (f"Element {atom} cannot be found. Raising exception {e}")
        raise ValueError(error_text)

    return atom_weight * u


def atom_type_to_number(atom_type):
    """
    returns ordinal number of atom

    Args:
        param1 (String): atom type

    Returns:
        ordinal number (int)
    """
    if (atom_type.lower() == "h"):
        return 1
    elif (atom_type.lower() == "c"):
        return 6
    elif (atom_type.lower() == "n"):
        return 7
    elif (atom_type.lower() == "o"):
        return 8
    elif (atom_type.lower() == "f"):
        return 9
    elif (atom_type.lower() == "cl"):
        return 17
    elif (atom_type.lower() == "br"):
        return 35
    elif (atom_type.lower() == "i"):
        return 53
    elif (atom_type.lower() == "au"):
        return 79
    elif (atom_type.lower() == "s"):
        return 16
    elif (atom_type.lower() == "si"):
        return 14
    else:
        print("Sorry I do not know this atom yet")
        return -1


def determine_n_orbitals(coord_path, basis_set="dev-SV(P)"):
    """
    Determines number of orbitals for calculation using coord file (turbomole format, xyz support is planned) and basis set.
    Args:
        coord_path (String): path to coord file (turbomole coord file)
        basis_set (String): basis set

    Returns:
        n_orbitals (int): Number of orbitals

    """
    if (basis_set != "dev-SV(P)"):
        raise ValueError("Only support for dev-SV(P) Basis set right now.")

    coord = io.read_coord_file(coord_path)
    atom_dict = {}
    ranges = list()
    for i, item in enumerate(coord):
        try:
            atom_dict[item[3]] += 1
        except KeyError:
            atom_dict[item[3]] = 1

        start, end = find_c_range_atom(item[3], atom_dict[item[3]], coord_path)
        ranges.append((start, end, item[3]))
    n_orbitals = ranges[-1][1]

    return n_orbitals


def get_norb_from_config(config):
    """
    Get number of orbitals from turbomole config string (for example [6s3p2d])
    Args:
        config (String): configuration string

    Returns:
        Norb (int): Number of orbitals
    """
    orbital_types = ['s', 'p', 'd', 'f']
    orbital_degeneracy = [1, 3, 5, 7]
    number_of_orbital_types = np.zeros(len(orbital_types))

    for i in range(0, len(orbital_types)):
        ns, config = config.split(orbital_types[i])
        ns = int(ns)
        number_of_orbital_types[i] = ns
        if config == '':
            break
    if (config != ''):
        raise ValueError(f"There is something wrong with this configuration")

    Norb = np.sum(number_of_orbital_types * orbital_degeneracy)
    return int(Norb)



