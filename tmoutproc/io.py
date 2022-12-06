"""
Loading and saving coord files in xyz and turbomole format
"""
__docformat__ = "google"

import numpy as np
import fnmatch
import scipy.sparse
import re as r
from scipy.linalg import eig
from functools import partial
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from .constants import *
from .io import *
from .calc_utils import get_norb_from_config

__ang2bohr__ = 1.88973

def read_coord_file(filename):
    """
    Reads data in file (eg plot data). Coordinates are converted to float. Method is implemented in this way to allow
    fixed and unfixed atoms and their detection by the length of the lists.

    Args:
        param1 (String): Filename



    Returns:
        array of lists [line in coordfile][0=x, 1=y, 2=z, 3=atomtype, optional: 4=fixed]
    """

    datContent = [i.strip().split() for i in open(filename).readlines()]
    # filter everything but $coord information
    cut = 0
    for i in range(1, len(datContent)):
        if (len(datContent[i]) < 4):
            cut = i
            datContent = datContent[0:cut + 1]
            break

    datContent = np.array(datContent, dtype=object)
    datContent = np.transpose(datContent[1:len(datContent) - 1])
    for i, item in enumerate(datContent):
        datContent[i][0] = float(item[0])
        datContent[i][1] = float(item[1])
        datContent[i][2] = float(item[2])
    datContent = np.transpose(datContent)
    return datContent



def write_coord_file(filename, coord):
    """
    writes coord file.

    Args:
        param1 (String): filename
        param3 (np.ndarray): dat content

    Returns:

    """
    file = open(filename, "w")
    file.write("$coord")
    file.write("\n")
    for i in range(0, len(coord)):
        if (len(coord[i]) == 4):
            file.write(
                str(coord[i][0]) + " " + str(coord[i][1]) + " " + str(coord[i][2]) + " " + str(coord[i][3]) + "\n")
        elif (len(coord[i]) == 5):
            file.write(
                str(coord[i][0]) + " " + str(coord[i][1]) + " " + str(coord[i][2]) + " " + str(coord[i][3]) + " " + str(
                    coord[i][4]) + "\n")
        else:
            print("Coord file seems to be funny")
    file.write("$user-defined bonds" + "\n")
    file.write("$end")
    file.close()

def read_packed_matrix(filename, output="sparse"):
    """
    Reads packed symmetric matrix from file and creates matrix
    returns matrix in scipy.sparse.csc_matrix or dense matrix (np.ndarray) depending on output param

    Args:
        param1 (String): filename
        param2 (String): Output format: "sparse" (->scipy.sparse.csc_matrix) or "dense" (->np.ndarray)


    Example:
        761995      nmat
           1                 1.000000000000000000000000
           2                0.9362738788014223212385900
           3                 1.000000000000000222044605
           4                0.4813841609941338917089126
           5                0.6696882728399407014308053
           6                 1.000000000000000000000000
           .
           .
           .

    Returns:
        scipy.sparse.csc_matrix

    """
    # matrix data
    data = []
    # matrix rows, cols
    i = []
    j = []

    counter = -1
    col = 0
    col_old = 0
    row = 0;
    # open smat_ao.dat and create sparse matrix
    with open(filename) as f:
        for line in f:
            # skip first line
            if (counter == -1):
                counter += 1
                continue
            line_split = r.split('\\s+', line)
            matrix_entry = float(line_split[2])

            # calculate row and col
            if (col == col_old + 1):
                col_old = col
                col = 0;
                row += 1

            # skip zero elemts
            if (matrix_entry != 0.0):
                data.append(matrix_entry)
                i.append(col)
                j.append(row)
                # symmetrize matrix
                if (col != row):
                    data.append(matrix_entry)
                    i.append(row)
                    j.append(col)
            col += 1
            counter += 1

    coo = coo_matrix((data, (i, j)), shape=(row + 1, row + 1))
    csc = scipy.sparse.csc_matrix(coo, dtype=float)
    if(output == "sparse"):
        return (csc)
    elif(output == "dense"):
        return csc.todense()
    else:
        raise ValueError(f"Input {output=} not valid")


def write_packed_matrix(matrix, filename, header="default", footer=""):
    """
    write symmetric scipy.sparse.csc_matrix in  in packed storage form

    Args:
        param1 (scipy.sparse.csc_matrix or np.ndarray): input matrix
        param2 (String): filename
        param3 (String): Header. Unless otherwise specified "num_elements_to_write      nmat" is written to file
        param4 (String): Footer. Default setting is no footer ("")

    Returns:

    """

    num_rows = matrix.shape[0]
    num_elements_to_write = (num_rows ** 2 + num_rows) / 2

    element_counter = 1
    with open(filename, "w") as f:
        if(header == "default"):
            line_beginning_spaces = ' ' * (12 - len(str(num_elements_to_write)))
            f.write(line_beginning_spaces + str(int(num_elements_to_write)) + "      nmat\n")
        elif(header != ""):
            f.write(f"{header}\n")

        for r in range(0, num_rows):
            num_cols = r + 1
            for c in range(0, num_cols):
                matrix_element = matrix[r, c]
                f.write(f"{element_counter:>20}{matrix_element:>43.16e}\n")
                element_counter += 1
        if(footer != ""):
            f.write(footer)




def read_qpenergies(filename, col=1, skip_lines=1):
    """
    read qpenergies.dat and returns data of given col as list, skip_lines in beginning of file, convert to eV
    col = 0 qpenergies, col = 1 Kohn-Sham, 2 = GW

    Args:
        param1 (string): filename
        param2 (int): col=2 (column to read)
        param3 (int): skip_lines=1 (lines without data at beginning of qpenergiesfile)

    Returns:
        list
    """

    har_to_ev = 27.21138602
    qpenergies = list()
    datContent = [i.strip().split() for i in open(filename).readlines()[(skip_lines - 1):]]
    # print(datContent)
    # datContent = np.transpose(datContent)
    # print(datContent)

    for i in range(skip_lines, len(datContent)):
        energy = float(datContent[i][col]) / har_to_ev
        # print("qpenergy " + str(energy))
        qpenergies.append(energy)
    return qpenergies


def write_mos_file(eigenvalues, eigenvectors, filename="mos_new.dat"):
    """
    write mos file, requires python3

    Args:
        param1 (np.array): eigenvalues
        param2 (np.ndarray): eigenvectors
        param3 (string): filename="mos_new.dat"
    """
    f = open(filename, "w")
    # header
    f.write("$scfmo    scfconv=7   format(4d20.14)\n")
    f.write("# SCF total energy is    -6459.0496515472 a.u.\n")
    f.write("#\n")
    for i in range(0, len(eigenvalues)):
        # print("eigenvalue " + str(eigenvalues[i]) + "\n")
        first_string = ' ' * (6 - len(str(i))) + str(i + 1) + "  a      eigenvalue=" + eformat(eigenvalues[i], 14,
                                                                                               2) + "   nsaos=" + str(
            len(eigenvalues))
        f.write(first_string + "\n")
        j = 0
        while j < len(eigenvalues):
            for m in range(0, 4):
                num = eigenvectors[m + j, i]
                string_to_write = f"{num:+20.13E}".replace("E", "D")
                f.write(string_to_write)
            # f.write(eformat(eigenvectors[m+j,i], 14,2).replace("E", "D"))
            f.write("\n")
            j = j + 4
        # print("j " + str(j))
    f.write("$end")
    f.close()


def read_mos_file(filename, skip_lines=1):
    """
    read mos file

    Args:
        param1 (string): filename
        param2 (int): skip_lines=1 (lines to skip )

    Returns:
        eigenvalue_list (list),eigenvector_list (np.ndarray)


    """
    n = 20
    level = 0
    C_vec = list()
    counter = 0
    eigenvalue_old = -1
    eigenvalue = 0
    beginning = True
    eigenvalue_list = list()
    eigenvector_list = -1

    # open mos file and read linewise
    with open(filename) as f:
        for line in f:
            # skip prefix lines
            # find level and calculate A(i)
            if (counter > skip_lines):
                index1 = (line.find("eigenvalue="))
                index2 = (line.find("nsaos="))
                # eigenvalue and nsaos found -> new orbital
                if (index1 != -1 and index2 != -1):
                    level += 1
                    # find nsaos of new orbital
                    nsaos = (int(line[(index2 + len("nsaos=")):len(line)]))

                    # find eigenvalue of new orbital and store eigenvalue of current orbital in eigenvalue_old
                    if (beginning == True):
                        eigenvalue_old = float(line[(index1 + len("eigenvalue=")):index2].replace("D", "E"))
                        eigenvalue = float(line[(index1 + len("eigenvalue=")):index2].replace("D", "E"))
                        eigenvector_list = np.zeros((nsaos, nsaos), dtype=float)
                        beginning = False
                    else:
                        eigenvalue_old = eigenvalue

                        eigenvalue = float(line[(index1 + len("eigenvalue=")):index2].replace("D", "E"))

                    # calculate A matrix by adding A_i -> processing of previous orbital
                    if (len(C_vec) > 0):
                        eigenvalue_list.append(eigenvalue_old)
                        # print("level " + str(level))
                        eigenvector_list[:, (level - 2)] = C_vec
                        C_vec = list()
                    # everything finished (beginning of new orbital)
                    continue

                line_split = [line[i:i + n] for i in range(0, len(line), n)]
                try:
                    C_vec.extend([float(line_split[j].replace("D", "E")) for j in range(0, len(line_split) - 1)][0:4])
                except ValueError:
                    print("sorry faulty mos file " + filename)
                    raise ValueError('Sorry. faulty mos file')
            # print(len(C_vec))

            counter += 1

    # handle last mos
    eigenvalue_list.append(eigenvalue_old)
    eigenvector_list[:, (nsaos - 1)] = C_vec

    return eigenvalue_list, eigenvector_list


def write_plot_data(filename, data, header=""):
    """
    Writes data in file (eg plot data)

    Args:
        param1 (String): Filename
        param2 (tuple): tuple of lists (each entry is one column in data)
        param3 (String): Fileheader


    Returns:

    """
    file = open(filename, "w")
    # file = open(new_file-path,"w")
    # print(data[0])
    if (len(header) != 0):
        file.write(header)
        file.write("\n")

    for i in range(0, len(data[0])):

        for j in range(0, len(data)):
            file.write(str(data[j][i]))
            file.write("	")
        file.write("\n")
    # file.write(data)
    file.close()


def read_plot_data(filename):
    """
    Reads data in file (eg plot data)

    Args:
        param1 (String): Filename



    Returns:
        datContent, head (Array of lists and header of file)
    """
    datContent = [i.strip().split() for i in open(filename).readlines()]
    try:
        float(datContent[0][0])
    except ValueError:
        datContent = np.transpose(datContent[1:len(datContent)])
        # for 1D-01 to 1E-01 conversion
        datContent = np.core.defchararray.replace(datContent, 'D', 'E')
        return np.array(datContent, dtype=float), datContent[0]
    return np.array(np.transpose(datContent), dtype=float), ""


def read_xyz_file(filename, return_header = False):
    """
    load xyz file and returns data as coord_xyz.
    coord[:,i] ... Entries for atom i
    coord[i,:] ... Atoms types (str) [i=0], x (float) [i=1], y (float) [i=2], x (float) [z=3] for all atoms

    Args:
        param1 (String): filename
        param2 (Boolean): return_header: If set to true (coord_xyz, header) is returned


    Returns:
        coord_xyz, header (optional)
    """
    datContent = [i.strip().split() for i in open(filename).readlines()]
    comment_line = np.transpose(datContent[1])
    datContent = np.array(datContent[2:len(datContent)], dtype=object)
    for i, item in enumerate(datContent):
        datContent[i, 1] = float(item[1])
        datContent[i, 2] = float(item[2])
        datContent[i, 3] = float(item[3])
    coord_xyz = np.transpose(datContent)
    if(return_header):
        header = " ".join(str(x) for x in comment_line)
        return (coord_xyz, str(header))
    else:
        return coord_xyz


def write_xyz_file(filename, coord_xyz, comment_line=""):
    """
    writes xyz file.

    Args:
        param1 (String): filename
        param2 (np.ndarray): coord_xyz
        param3 (String): comment_line

    Returns:

    """
    with open(filename, "w") as file:
        file.write(str(len(coord_xyz)))
        file.write("\n")
        file.write(comment_line)
        file.write("\n")

        for i in range(0, coord_xyz.shape[1]):
            file.write(str(coord_xyz[0,i]) + "	" + str(coord_xyz[1,i]) + "	" + str(coord_xyz[2,i]) + "	" + str(
                coord_xyz[3,i]) + "\n")
    file.close()


def read_hessian(filename, n_atoms, dimensions=3):
    """
    Reads hessian from turbomole format and converts it to matrix of float

    Args:
        param1 (String) : Filename
        param2 (int) : Number of atoms

    Returns:
        np.ndarray
    """
    skip_lines = 1
    datContent = [i.strip().split() for i in open(filename).readlines()[(skip_lines):]]

    # for three dimensions 3N dimensions
    n_atoms = n_atoms * dimensions
    hessian = np.zeros((n_atoms, n_atoms))
    counter = 0
    # output from aoforce calculation
    if (len(datContent[0]) == 7):
        lower = 2
        aoforce_format = True
        # because of $end statement
        datContent = datContent[:len(datContent) - 1]
    elif (len(datContent[0]) == 5):
        lower = 0
        aoforce_format = False
    else:
        aoforce_format = False
        lower = 0

    for j in range(0, len(datContent)):
        # handle aoforce calculation and mismatch of first lines 1 1, .., 1|10  (-> read wrong)
        if (aoforce_format == True):
            try:
                int(datContent[j][1])
            except ValueError:
                lower = 1
        for i in range(lower, len(datContent[j])):
            counter += 1
            row = int((counter - 1) / (n_atoms))
            col = (counter - row * n_atoms) - 1
            try:
                hessian[row, col] = float(datContent[j][i])
            except IndexError:
                raise ValueError('n_atoms wrong. Check dimensions')

        # reset lower for next lines
        if (aoforce_format == True):
            lower = 2

    if (counter != n_atoms ** 2):
        raise ValueError('n_atoms wrong. Check dimensions')
    return hessian


def read_symmetric_from_triangular(input_path, skip_lines=0):
    """
    Read symmetric matrix stored in triangular form.
    Example Input : a, b, d, c, e, f
    Example Output:
    a b c
    b d e
    c e f

    Args:
        input_path (String): Path to file
        skip_lines (int): Lines of file which sould be skipped (e.g. header stuff)

    Returns:

    """
    datContent = np.asarray([i.strip().split() for i in open(input_path).readlines()[(skip_lines):]])
    datContent = datContent.flatten()
    #assume matrix has size n(n+1)/2=len(datContent)
    len_ = int((-1+np.sqrt(1+8*len(datContent)))/2)
    matrix = np.zeros((len_,len_), dtype = float)
    col = 0
    col_old = 0
    row = 0
    for i, item in enumerate(datContent):
        if(col == col_old + 1):
            col_old = col
            col = 0
            row += 1
        matrix[col, row] = float(item)
        matrix[row, col] = float(item)
        col +=1

    return matrix


def write_symmetric_to_triangular(matrix, output_path, threshold=0.0):
    """
    Write symmetric matrix to file. The triangular form ist written to file.
    Example Input:
    a b c
    b d e
    c e f
    Example Output:
    a, b, d, c, e, f

    Args:
        matrix (numpy.ndarray): Symmetric matrix
        output_path (String): Path of output file
        threshold (float): Threshold for accepted asymmetry


    Returns:

    """
    # make sure matrix is symmetric
    if (np.max(np.abs(matrix - np.transpose(matrix))) > threshold):
        raise ValueError("Matrix is asymmetric. Check matrix or increase threshold")
    shape = matrix.shape[0]
    if (shape != matrix.shape[1]):
        raise ValueError("Matrix is not quadratic")
    with open(output_path, "w") as file:
        counter = 0
        for i in range(0, shape):
            for j in range(0, i + 1):
                counter += 1
                if counter == 3:
                    counter = 0
                    file.write(f"{matrix[i, j]:< 24.16e}\n")
                else:
                    file.write(f"{matrix[i, j]:< 24.16e}")




def create_sysinfo(coord_path, basis_path, output_path):
    """
	Create sysinfo according to given template. Sysinfo is used for transport calculations used by Theo 1 Group of
	University of Augsburg. It contains information about the orbitals, the charge, number of ecps and the coordinates of the positions
	{number atom} {orbital_index_start} {orbital_index_end} {charge} {number ecps} {x in Bohr} {y in Bohr} {z in Bohr}

	Args:
		coord_path (String): Path to turbomole coord file
		basis_path (String): Path to turbomole basis file
		output_path (String): Path where sysinfo is written

	Returns:

	"""
    coord = read_coord_file(coord_path)
    # number of atoms:
    natoms = np.size(coord)
    # split cd into position array and element vector
    pos = np.zeros(shape=(natoms, 3))
    el = []
    for i in range(natoms):
        pos[i, :] = coord[i][0:3]
        el.append(coord[i][3])

    # make list from dictionary from list el
    # --> get rid of the douplicates
    # eldif is a list of all different elements in coord
    eldif = list(dict.fromkeys(el))
    # dictionary for Number of orbitals:
    Norb_dict = dict.fromkeys(eldif, 0)
    # dictionary for Number of ECP-electrons:
    Necp_dict = dict.fromkeys(eldif, 0)

    for elkind in eldif:
        nf = 0
        # the space in searchstr is to avoid additional findings (e.g. for s)
        searchstr = elkind + ' '
        # search for element in file 'basis'
        with open(basis_path) as fp:
            lines = fp.readlines()
            for line in lines:
                if line.find(searchstr) != -1:
                    nf += 1
                    if (nf == 1):
                        # first occurence: name of basis set
                        continue
                    if (nf == 2):
                        # second occurence electron configuration per atom
                        # this might look weird but leads to the part
                        # inside the []-brackets
                        config = line.strip().split('[')[1].split(']')[0]
                        Norb_dict[elkind] = get_norb_from_config(config)
                    if (nf == 3):
                        # third occurence: information on core-potential
                        # number of core electrons is 2 lines after that
                        nl_ecp = lines.index(line)
                        # get number of core electrons
                        N_ecp = lines[nl_ecp + 2].strip().split()[2]
                        # save this number in corresponding dictionary
                        Necp_dict[elkind] = N_ecp

    iorb = 0
    with open(output_path, 'w') as file:
        file.write(f"{natoms:>22}\n")
        for iat in range(natoms):
            atomic_number = ATOM_DICT_SYM[el[iat]][0]
            charge = atomic_number - int(Necp_dict[el[iat]])
            file.write(f"{iat+1:>8} {iorb + 1:>8} {iorb + Norb_dict[el[iat]]:>8} {charge:>24.12f} {Necp_dict[el[iat]]:>8} {pos[iat, 0]:>24} {pos[iat, 1]:>24} {pos[iat, 2]:>24}\n")
            iorb = iorb + Norb_dict[el[iat]]


def read_from_flag_to_flag(control_path, flag, output_path, header="", footer=""):
    """
    Reads file in control_path from flag to next flag or end of file. This can be used to get parts of the turbomole
    control file. Flags begin with "$". The extracted part is written to file output_path without the leading flag. If
    flag is found and corresponding part contains data, status 0 is returned, file is written. If flag is not found,
    status -1 is returned and no file is written. If flag is found, but corresponding part contains no data, status -2
    is returned and no file is written.

    Args:
        control_path (String): Path to considered file
        flag (String): Flag which
        output_path (String): Path to output file
        header (String): Header added to output file
        footer (String): Footer added to output file

    Returns:
        status (int)
    """
    found_part = False
    part = list()
    with open(control_path) as input_file:
        lines = input_file.readlines()
        for line in lines:
            if(line.find(flag) != -1):
                found_part = True
                continue
            if (found_part == False):
                continue
            if line.startswith("$"):
                break
            part.append(line)
    if(found_part == False):
        return -1
    if(found_part == True and len(part) == 0):
        return -2
    with open(output_path, "w") as output_file:
        if header != "":
            output_file.write(f"{header}\n")
        for line in part:
            output_file.write(f"{line}")
        if footer != "":
            output_file.write(f"{header}")
    return 0