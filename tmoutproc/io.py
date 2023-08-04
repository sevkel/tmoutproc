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
import itertools

__ang2bohr__ = 1.88973

def read_coord_file(filename, skip_lines=1):
    """
    Reads coord file in turbomole format. Coordinates are converted to float, atoms types are strings.
    coord[:,i]...All entries for atom i.
    coord[0,:]...x coordinates of all atoms (float)
    coord[1,:]...y coordinates of all atoms (float)
    coord[2,:]...z coordinates of all atoms (float)
    coord[3,:]...atom types of all atoms (str)
    coord[4,:]...fixed information
    Lines in the end of file starting with "$" are skippend (e.g. $user-defined bonds and $end)


    Args:
        param1 (String): Filename
        param2 (int): Lines to be skipped in the beginning (e.g. $coord line)

    Returns:
        coord (np.ndarray): coord with shape (5, N_atoms)
    """

    datContent = np.array([i.strip().split() for i in open(filename).readlines()],dtype=object)
    coord = list()
    for i, item in enumerate(datContent):
        if(datContent[i][0].startswith("$")):
            continue
        #store x,y,z coordinates
        list_entries = [float(datContent[i][j]) for j in range(3)]
        #append atom type
        list_entries.append(datContent[i][3])
        #handle information of fixed atoms
        if(len(datContent[i])>4):
            list_entries.append(datContent[i][4])
        else:
            list_entries.append("")
        coord.append(list_entries)

    coord = np.asarray(coord,dtype=object)
    coord = np.transpose(coord)

    return coord




def write_coord_file(filename, coord):
    """
    writes coord file.

    Args:
        param1 (String): filename
        param3 (np.ndarray): dat content

    Returns:

    """
    with open(filename, "w") as file:
        file = open(filename, "w")
        file.write("$coord")
        file.write("\n")
        for i in range(0, coord.shape[1]):
            if (len(coord[4,i]) == ""):
                file.write(f"{coord[0,i]} {coord[1,i]} {coord[2,i]} {coord[3,i]}\n")
            else:
                file.write(f"{coord[0,i]} {coord[1,i]} {coord[2,i]} {coord[3,i]} {coord[4,i]}\n")
        file.write("$user-defined bonds" + "\n")
        file.write("$end")

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


def write_mos_file(eigenvalues, eigenvectors, filename, scfconv="", total_energy=""):
    """
    Writes mos file in turbomole format.

    Args:
        param1 (np.array): eigenvalues
        param2 (np.ndarray): eigenvectors
        param3 (string): filename
        param4 (int): scfconv parameter (optional)
        param5 (float): total_energy in a.u. (optional)
    """

    assert len(eigenvalues) == eigenvectors.shape[0], "Mos file would be weird"
    assert len(eigenvalues) == eigenvectors.shape[1], "Mos file would be weird"

    def eformat(f, prec, exp_digits):

        s = "%.*e" % (prec, f)
        # s ="hallo"
        mantissa, exp = s.split('e')
        # add 1 to digits as 1 is taken by sign +/-
        return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))

    with open(filename, "w") as f:
        # header
        f.write(f"$scfmo    scfconv={scfconv}   format(4d20.14)\n")
        f.write(f"# SCF total energy is    {total_energy} a.u.\n")
        f.write("#\n")
        for i in range(0, len(eigenvalues)):
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


def read_mos_file(filename):
    """
    Reads mos file. Eigenvectors[:,i] are the components for MO i.

    Args:
        param1 (string): filename

    Returns:
        eigenvalue_list (list),eigenvectors (np.ndarray)


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
    in_cvec_range = False

    # open mos file and read linewise
    with open(filename) as f:
        for line in f:
            # find level and calculate A(i)

            # skip prefix lines
            index1 = (line.find("eigenvalue="))
            index2 = (line.find("nsaos="))
            if(index1 == -1 and in_cvec_range == False):
                continue

            # eigenvalue and nsaos found -> new orbital
            if (index1 != -1 and index2 != -1):
                in_cvec_range = True
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
    eigenvalue_list.append(eigenvalue)
    eigenvector_list[:, (nsaos - 1)] = C_vec

    return eigenvalue_list, eigenvector_list


def write_plot_data(filename, data, header="", delimiter="	"):
    """
    Writes data in file (eg plot data).
    data[j][i] : Column j, Line i

    Args:
        param1 (String): Filename
        param2 (List, Tuple, np.ndarray): Data to be stored (each entry is one column in data, data[j][i] : Column j, Line i)
        param3 (String): Fileheader
        param4 (String): Delimiter


    Returns:

    """
    if(delimiter == ""):
        raise ValueError(f"Delimiter is empty")
    try:
        float_delimiter = True
        float(delimiter)
    except ValueError:
        float_delimiter = False
    finally:
        if(float_delimiter == True):
            raise ValueError(f"Delimiter {delimiter} makes no sense")

    if(type(data) == list or type(data) == tuple):
        n_lines = len(data[0])
        n_cols = len(data)
    elif(type(data) == np.ndarray):
        n_lines = data.shape[1]
        n_cols = data.shape[0]
    else:
        raise ValueError(f"Data type {type(data)} not supported")

    with open(filename, "w") as file:

        if (len(header) != 0):
            file.write(header)
            file.write("\n")

        for i in range(0, n_lines):

            for j in range(0, n_cols):
                file.write(str(data[j][i]))
                file.write(delimiter)
            file.write("\n")



def read_plot_data(filename, return_header=False, delimiter="	", skip_lines_beginning = 0, skip_lines_end = 0):
    """
    Reads data in file (eg plot data). If return_header is set to True, the header is returned as String. Header can
    only returned if header is found.
    data[i,:] is the ith column in the data file. skip_lines are lines to be skipped.
    These lines are not read. Headers cannot be extracted from these lines.

    Args:
        param1 (String): Filename
        param2 (Boolean): Return Header
        param3 (String): Delimiter
        param4 (int): Lines to be skipped at the beginning
        param5 (int): Lines to be skipped at the end



    Returns:
        datContent (np.ndarray), Header
    """
    data = [i.replace(delimiter, "	").strip().split() for i in open(filename).readlines()[(skip_lines_beginning):]]
    del data[len(data) - skip_lines_end:]
    try:
        float(data[0][0])
    except ValueError:
        header = data[0]
        header = " ".join(str(x) for x in header)
        data = np.transpose(data[1:len(data)])
        # for 1D-01 to 1E-01 conversion
        data = np.core.defchararray.replace(data, 'D', 'E')
        if(return_header == True):
            return np.array(data, dtype=float), header
        else:
            return np.array(data, dtype=float)
    return np.array(np.transpose(data), dtype=float)


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

def read_xyz_path_file(filename, return_header = False, start_geo=0, end_geo=-1):
    """
    load xyz path file and returns data as coord_xyz with additional dimension.
    coord[i,:,:] ... Entries for structure i
    coord[:,:,i] ... Entries for atom i
    coord[:,i,:] ... Atoms types (str) [i=0], x (float) [i=1], y (float) [i=2], x (float) [z=3] for all atoms

    Args:
        param1 (String): filename
        param2 (Boolean): return_header: If set to true (coord_xyz_path, header) is returned. coord_xyz_path else
        param3 (int): start_geo: First geometry to be read
        param4 (int): end_geo: Last geometry to be read

    Returns:
        coord_xyz_path, header (optional)
    """



    def read_file_range(filename, start, end):
        """
        Read slice from file (for large files)
        Args:
            filename (String): Filename
            start (int): First line to be read
            end (int): Last line to be read

        Returns:
            lines (list): List of lines
        """
        lines = []
        with open(filename, "r") as text_file:
            for line in itertools.islice(text_file, start, end):
                lines.append(line.strip().split())
        return lines

    # determine number of lines and number of atoms (done in this way for large files)
    with open(filename, "r") as f:
        first_line = f.readline().strip().split()
        # plus one because first line is already read
        num_lines = sum(1 for _ in f) + 1


    Natoms=int(first_line[0])
    Ngeos=int(num_lines/(Natoms+2))

    if (end_geo==-1):
        end_geo=Ngeos

    if (start_geo<0):
        raise ValueError("start_geo must be >=0")
    if (end_geo<start_geo):
        raise ValueError("end_geo must be >= start_geo")
    if (end_geo > Ngeos):
        raise ValueError("end_geo must be <=Ngeos")


    energies=np.zeros(np.abs(end_geo-start_geo))
    energy_line_=""
    for igeo in range(start_geo, end_geo):
        datContGeo=read_file_range(filename,(2+Natoms)*igeo+1,(2+Natoms)*(igeo+1))
        energy_line_ = datContGeo[0]
        datContGeo = datContGeo[1:len(datContGeo)]
        datContGeo = np.array(datContGeo, dtype=object)
        for i, item in enumerate(datContGeo):
            datContGeo[i, 1] = float(item[1])
            datContGeo[i, 2] = float(item[2])
            datContGeo[i, 3] = float(item[3])
        coord_geo=np.transpose(datContGeo).reshape((1,4,Natoms))
        if (igeo==start_geo):
            coord_path=coord_geo
        else:
            coord_path=np.append(coord_path,coord_geo,axis=0)

        #sometimes energies are
        try:
            energies[igeo-start_geo]=float(energy_line_[2])
        except IndexError:
            energies[igeo-start_geo]=0.0

    if(return_header==True):
        return coord_path,energies
    else:
        return coord_path


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
        file.write(str(coord_xyz.shape[1]))
        file.write("\n")
        file.write(comment_line)
        file.write("\n")

        for i in range(0, coord_xyz.shape[1]):
            newline = "\n"
            if(i == coord_xyz.shape[1]-1):
                newline = ""
            file.write(f"{coord_xyz[0,i]}	{coord_xyz[1,i]}	{coord_xyz[2,i]}	{coord_xyz[3,i]}{newline}")
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
    natoms = coord.shape[1]
    # split cd into position array and element vector
    pos = np.zeros(shape=(natoms, 3))
    el = []
    for i in range(natoms):
        pos[i, :] = coord[0:3,i].T
        el.append(coord[3,i])

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
                    #valid search strings are at the beginning of the line are have a space in front of it -> avoid "zn" and "n" problems
                    if(line.index(searchstr) == 0 or line[line.index(searchstr)-1] == " " ):
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

def write_lammps_geo_data(filename, coord_xyz, xlo=0, xhi=0, ylo=0, yhi=0, zlo=0, zhi=0, charges=None, units="real", atom_style="full", header=""):
    """
    Writes xyz coordinates to lammps geometry data file. Right now only units real and atom_style full are supported.
    Args:
        filename (String): Filename of output file
        coord_xyz (np.ndarray): Coordinates of atoms in xyz format read by read_xyz_file
        xlo (float): Lower x boundary
        xhi (float): Upper x boundary
        ylo (float): Lower y boundary
        yhi (float): Upper y boundary
        zlo (float): Lower z boundary
        zhi (float): Upper z boundary
        charges (np.ndarray): Charges of atoms
        units (String): Lammps units (only "real" supported)
        atom_style (String): Lammps atom_style (only "full" supported)
        header (String): Header added to output file

    Returns:


    """

    if(atom_style!="full"):
        raise NotImplementedError("Only atom_style full supported right now")
    if(units!="real"):
        raise NotImplementedError("Only units real supported right now")

    #check if xlo, xhi, ylo, yhi, zlo, zhi are interpretable as floats
    try:
        xlo = float(xlo)
        xhi = float(xhi)
        ylo = float(ylo)
        yhi = float(yhi)
        zlo = float(zlo)
        zhi = float(zhi)
    except ValueError:
        raise ValueError("xlo, xhi, ylo, yhi, zlo, zhi must be interpretable as floats")

    n_atoms = coord_xyz.shape[1]

    #if charges is not none and number of charges does not match number of atoms, raise error
    if(charges is not None):
        if(len(charges) != n_atoms):
            raise ValueError("Number of charges does not match number of atoms")

    #determine different atom types
    unique_elements = list(set(coord_xyz[0,:]))
    n_atom_types = len(unique_elements)
    atom_type_keys = {}
    for idx, element in enumerate(unique_elements):
        atom_type_keys[element] = f'{idx + 1}'


    with open(filename, "w") as output_file:
        output_file.write(f"{header}\n")
        output_file.write(f"\n")

        output_file.write(f"\t{n_atoms} atoms\n")
        output_file.write(f"\t{n_atom_types} atom types\n")
        output_file.write("\n")

        output_file.write(f"{xlo} {xhi} xlo xhi\n")
        output_file.write(f"{ylo} {yhi} ylo yhi\n")
        output_file.write(f"{zlo} {zhi} zlo zhi\n")
        output_file.write("\n")

        output_file.write("Masses\n")
        output_file.write("\n")
        for i, element in enumerate(unique_elements):
            mass = ATOM_DICT_SYM[element.lower()][2]
            output_file.write(f"\t{i+1} {mass}\n")
            #atom_type_dict[i+1] = unique_elements[i]
        output_file.write("\n")

        output_file.write("Atoms\n")
        output_file.write("\n")

        charge = 0
        for i in range(n_atoms):
            if charges is None:
                output_file.write(f"{i+1} {i+1} {atom_type_keys[coord_xyz[0,i]]}  {charge} {coord_xyz[1,i]} {coord_xyz[2,i]} {coord_xyz[3,i]}\n")
            else:
                output_file.write(f"{i+1} {i+1} {atom_type_keys[coord_xyz[0,i]]}  {charges[i]} {coord_xyz[1,i]} {coord_xyz[2,i]} {coord_xyz[3,i]}\n")



if __name__ == '__main__':
    filename = "./tests/test_data/stretching.xyz"
    