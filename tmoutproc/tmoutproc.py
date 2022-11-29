"""
Package for processing turbomole Output such as mos files
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
from . import  constants

__ang2bohr__ = 1.88973



def read_packed_matrix(filename):
	"""
	read packed symmetric matrix from file and creates matrix
	returns matrix in scipy.sparse.csc_matrix
	Args:
		param1 (string): filename


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
	#matrix data
	data = []
	#matrix rows, cols
	i = []
	j = []

	counter =-1
	col = 0
	col_old = 0
	row = 0;
	#open smat_ao.dat and create sparse matrix
	with open(filename) as f:
		for line in f:
			#skip first line
			if(counter == -1):
				counter+=1
				continue
			line_split = r.split('\s+', line)
			matrix_entry = np.float(line_split[2])		

			#calculate row and col			
			if(col == col_old+1):
				col_old = col
				col = 0;
				row += 1

			#skip zero elemts
			if(matrix_entry != 0.0):
				data.append(matrix_entry)
				i.append(col)
				j.append(row)
				#symmetrize matrix
				if(col != row):
					data.append(matrix_entry)
					i.append(row)
					j.append(col)
			col += 1				
			counter+=1

	
	coo = coo_matrix((data, (i, j)), shape=(row+1, row+1))
	csc = scipy.sparse.csc_matrix(coo, dtype = float)	
	
	return(csc)




def write_matrix_packed(matrix, filename="test"):
	"""
	write symmetric scipy.sparse.csc_matrix in  in packed storage form

	Args:
		param1 (scipi.sparse.csc_matrix): input matrix
		param2 (string): filename

	Returns:

	"""
	print("writing packed matrix")
	num_rows = matrix.shape[0]
	num_elements_to_write = (num_rows**2+num_rows)/2	


	element_counter = 1
	f = open(filename, "w")

	line_beginning_spaces = ' ' * (12 - len(str(num_elements_to_write)))
	f.write(line_beginning_spaces + str(num_elements_to_write) + "      nmat\n")

	for r in range(0,num_rows):
		#print("writing row " +str(r))
		num_cols = r+1		
		for c in range(0, num_cols):
			matrix_element = matrix[r,c]
			line_beginning_spaces = ' ' * (20 -len(str(int(element_counter))))
			if(matrix_element>=0):
				line_middle_spaces = ' ' * 16
			else:
				line_middle_spaces = ' ' * 15
			
			f.write(line_beginning_spaces + str(int(element_counter)) + line_middle_spaces + eformat(matrix_element, 14,5) + "\n")			
			element_counter +=1
	f.close()
	print("writing done")


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
	datContent = [i.strip().split() for i in open(filename).readlines()[(skip_lines-1):]]
	#print(datContent)
	#datContent = np.transpose(datContent)
	#print(datContent)	

	for i in range(skip_lines, len(datContent)):
		energy = float(datContent[i][col])/har_to_ev
		#print("qpenergy " + str(energy))
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
	#header
	f.write("$scfmo    scfconv=7   format(4d20.14)\n")
	f.write("# SCF total energy is    -6459.0496515472 a.u.\n") 
	f.write("#\n")   
	for i in range(0, len(eigenvalues)):
		#print("eigenvalue " + str(eigenvalues[i]) + "\n")
		first_string = ' ' * (6-len(str(i))) + str(i+1) + "  a      eigenvalue=" + eformat(eigenvalues[i], 14,2) + "   nsaos=" + str(len(eigenvalues))
		f.write(first_string + "\n")
		j = 0
		while j<len(eigenvalues):
			for m in range(0,4):
				num = eigenvectors[m+j,i]
				string_to_write = f"{num:+20.13E}".replace("E", "D")
				f.write(string_to_write)
				#f.write(eformat(eigenvectors[m+j,i], 14,2).replace("E", "D"))
			f.write("\n")
			j = j +4
			#print("j " + str(j))
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
	eigenvalue_list= list()	
	eigenvector_list= -1	


	#open mos file and read linewise
	with open(filename) as f:
		for line in f:
			#skip prefix lines
			#find level and calculate A(i)
			if(counter>skip_lines):
				index1 = (line.find("eigenvalue="))
				index2 = (line.find("nsaos="))
				#eigenvalue and nsaos found -> new orbital
				if(index1 != -1 and index2 != -1):
					level += 1 
					#find nsaos of new orbital
					nsaos = (int(line[(index2+len("nsaos=")):len(line)]))	

					#find eigenvalue of new orbital and store eigenvalue of current orbital in eigenvalue_old
					if(beginning == True):
						eigenvalue_old = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
						eigenvector_list = np.zeros((nsaos,nsaos),dtype=float)						
						beginning = False
					else:
						eigenvalue_old = eigenvalue	
						
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
												 
													
						

					#calculate A matrix by adding A_i -> processing of previous orbital		
					if(len(C_vec)>0):							
						eigenvalue_list.append(eigenvalue_old)	
						#print("level " + str(level))
						eigenvector_list[:,(level-2)] = C_vec					
						C_vec = list()					
					#everything finished (beginning of new orbital)
					continue
				
				
				line_split = [line[i:i+n] for i in range(0, len(line), n)]
				try:							
					C_vec.extend([float(line_split[j].replace("D","E"))  for j in range(0,len(line_split)-1)][0:4])
				except ValueError:
					print("sorry faulty mos file " + filename)
					raise ValueError('Sorry. faulty mos file')
				#print(len(C_vec))
				
			counter += 1
			
	#handle last mos
	eigenvalue_list.append(eigenvalue_old)	
	eigenvector_list[:,(nsaos-1)] = C_vec
	
	return eigenvalue_list,eigenvector_list


def diag_F(f_mat_path, s_mat_path, eigenvalue_list = list()):
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
	F_mat_file = read_packed_matrix(f_mat_path)
	print("read s")
	s_mat = read_packed_matrix(s_mat_path)
	s_mat_save = s_mat	

	print("diag F")	
	

	
	#Konfiguration 1 : Übereinstimmung
	#eigenvalues, eigenvectors = np.linalg.eigh(F_mat_file.todense())

	#Konfiguration 2 : keine Übereinstimmung (sollte eigentlich das gleiche machen wie np.linalg.eigh)
	#eigenvalues, eigenvectors = scipy.linalg.eigh(F_mat_file.todense())

	#Konfiguration 3: keine Übereinstimmung (generalierstes EW Problem)
	eigenvalues, eigenvectors = scipy.linalg.eigh(F_mat_file.todense(), s_mat.todense())

	eigenvectors = np.real(np.asmatrix(eigenvectors))
	#eigenvectors = np.transpose(eigenvectors)
	eigenvalues = np.real(eigenvalues)
	eigenvalue_output_list = eigenvalues
	#print(eigenvalues)
	#print("type random")
	#print(type(eigenvalues).__name__)
	#print(type(eigenvectors).__name__)
	print("diag F done")

	print("calc fmat ")
	#take eigenvalues from diagonalization or external eigenvalues (e.g from qpenergies)
	if(len(eigenvalue_list) == 0):
		eigenvalues = np.diag(eigenvalues)
	else:		
		eigenvalues = np.diag(eigenvalue_list)
		eigenvalue_output_list = eigenvalue_list

	sc = s_mat * eigenvectors
	f_mat = eigenvalues * np.transpose(sc)
	f_mat = sc * f_mat
	print("calc fmat done")
	


	return f_mat,eigenvalue_output_list, eigenvectors
	

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
	#file = open(new_file-path,"w")
	#print(data[0])
	if(len(header) != 0):
		file.write(header)
		file.write("\n")

	for i in range(0, len(data[0])):

		for j in range(0,len(data)):			
			file.write(str(data[j][i]))
			file.write("	")
		file.write("\n")
	#file.write(data)
	file.close()

def read_plot_data(filename):
	"""
	Reads data in file (eg plot data)

	Args:
		param1 (String): Filename
		
		

	Returns:
		datContent, head (Array of lists and header of file)
	"""
	datContent= [i.strip().split() for i in open(filename).readlines()]
	try:
		float(datContent[0][0])
	except ValueError:
		datContent = np.transpose(datContent[1:len(datContent)])
		#for 1D-01 to 1E-01 conversion
		datContent = np.core.defchararray.replace(datContent,'D', 'E')
		return np.array(datContent,dtype=float), datContent[0]
	return np.array(np.transpose(datContent),dtype=float),""


def read_coord_file(filename):
	"""
	Reads data in file (eg plot data). Coordinates are converted to float. Method is implemented in this way to allow
	fixed and unfixed atoms and their detection by the length of the lists.
	
	Args:
		param1 (String): Filename
		
		

	Returns:
		array of lists [line in coordfile][0=x, 1=y, 2=z, 3=atomtype, optional: 4=fixed]
	"""

	datContent= [i.strip().split() for i in open(filename).readlines()]
	#filter everything but $coord information
	cut = 0
	for i in range(1, len(datContent)):
		if(len(datContent[i])<4):
			cut = i
			datContent=datContent[0:cut+1]
			break

	datContent=np.array(datContent, dtype=object)
	datContent= np.transpose(datContent[1:len(datContent)-1])
	for i, item in enumerate(datContent):
		datContent[i][0] = float(item[0])
		datContent[i][1] = float(item[1])
		datContent[i][2] = float(item[2])
	datContent = np.transpose(datContent)
	return datContent

def read_xyz_file(filename):
	"""
	Reads data in file (eg plot data) 
	
	Args:
		param1 (String): Filename
		
		

	Returns:
		array of lists [line in coordfile][0=x, 1=y, 2=z, 3=atomtype, optional: 4=fixed]
	"""

	datContent= [i.strip().split() for i in open(filename).readlines()]
	#filter everything but $coord information
	cut = 0
	for i in range(2, len(datContent)):
		if(len(datContent[i])<4):
			cut = i
			datContent=datContent[0:cut+1]
			break

	datContent=np.array(datContent, dtype=object)
	datContent= np.transpose(datContent[2:len(datContent)])
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
	for i in range(0,len(coord)):
		if(len(coord[i])==4):
			file.write(str(coord[i][0]) + " " + str(coord[i][1]) + " " + str(coord[i][2]) + " " + str(coord[i][3]) + "\n" )
		elif(len(coord[i])==5):
			file.write(str(coord[i][0]) + " " + str(coord[i][1]) + " " + str(coord[i][2]) + " " + str(coord[i][3]) + " "+ str(coord[i][4]) + "\n" )
		else:
			print("Coord file seems to be funny")
	file.write("$user-defined bonds" + "\n")
	file.write("$end")
	file.close()

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
		if(atom == "c"):
			return 14
		elif(atom == "h"):
			return 2
		elif(atom == "au"):
			return 25
		elif(atom == "s"):
			return 18
		elif(atom == "zn"):
			return 24
		elif(atom == "o"):
			return 14
		elif(atom == "fe"):
			return 24
		elif(atom == "f"):
			return 14
		elif(atom == "cl"):
			return 18
		elif(atom == "br"):
			return 32
		elif(atom == "i"):
			return 13

		else:
			raise ValueError('Sorry. This feature is not implemented for following atom: ' + atom)

	coord_content = read_coord_file(coordfile)
	if(basis_set=="dev-SV(P)"):
		number_found_atoms_of_type = 0
		number_expansion_coeff = 0
		for i in range(0, len(coord_content)):
			current_atom = coord_content[i][3]

			if(current_atom == atom):				
				number_found_atoms_of_type+=1
				if(number_found_atoms_of_type == number):				
					return (number_expansion_coeff, number_expansion_coeff+atom_type_to_number_coeff(current_atom))	
				number_expansion_coeff += atom_type_to_number_coeff(current_atom)				
			else:
				number_expansion_coeff += atom_type_to_number_coeff(current_atom)
		raise ValueError('Sorry. Could not find the atom')			
			

	else:
		raise ValueError('Sorry. This feature is not implemented for following basis_set: ' + basis_set)


def load_xyz_file(filename):
	"""
	load xyz file and returns data. Returns comment line and coord data. Dat content cols: atoms=0 (Str), x=1 (float), y=2 (float), z=3 (float).
	coord[:,i] ... Entries for atom i
	coord[i,:] ... Atoms types (str) [i=0], x (float) [i=1], y (float) [i=2], x (float) [z=3] for all atoms

	Args:
		param1 (String): filename


	Returns:
		(comment_line (String), coord (np.ndarray))
	"""
	datContent = [i.strip().split() for i in open(filename).readlines()]
	comment_line = np.transpose(datContent[1])
	datContent = np.array(datContent[2:len(datContent)], dtype=object)
	for i, item in enumerate(datContent):
		datContent[i, 1] = float(item[1])
		datContent[i, 2] = float(item[2])
		datContent[i, 3] = float(item[3])
	coord = np.transpose(datContent)
	return (comment_line, coord)

def write_xyz_file(filename, comment_line, coord_xyz):
	"""
	writes xyz file.

	Args:
		param1 (String): filename
		param2 (String): commment_line
		param3 (np.ndarray): dat content

	Returns:
		
	"""
	file = open(filename, "w")
	file.write(str(len(coord_xyz)))
	file.write("\n")
	file.write(comment_line)
	file.write("\n")
	for i in range(0,len(coord_xyz)):
		file.write(str(coord_xyz[i][0]) + "	" + str(coord_xyz[i][1]) + "	" + str(coord_xyz[i][2])+ "	" + str(coord_xyz[i][3]) + "\n")
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
	skip_lines=1
	datContent = [i.strip().split() for i in open(filename).readlines()[(skip_lines):]]
	
	#for three dimensions 3N dimensions
	n_atoms = n_atoms*dimensions
	hessian = np.zeros((n_atoms, n_atoms))
	counter=0
	#output from aoforce calculation
	if(len(datContent[0])==7):
		lower = 2
		aoforce_format = True
		# because of $end statement
		datContent = datContent[:len(datContent)-1]
	elif(len(datContent[0])==5):
		lower = 0
		aoforce_format = False
	else:
		aoforce_format = False
		lower = 0

	for j in range(0,len(datContent)):
		#handle aoforce calculation and mismatch of first lines 1 1, .., 1|10  (-> read wrong)
		if(aoforce_format == True):			
			try:
				int(datContent[j][1])
			except ValueError:
				lower = 1
		for i in range(lower,len(datContent[j])):
			counter +=1
			row = int((counter-1)/(n_atoms))
			col = (counter-row*n_atoms)-1
			hessian[row,col]=float(datContent[j][i])

		#reset lower for next lines
		if(aoforce_format == True):
			lower = 2	
			
	if(counter != n_atoms**2):
		raise ValueError('n_atoms wrong. Check dimensions')		
	return hessian

def create_dynamical_matrix(filename_hessian, filename_coord, t2SI=False, dimensions=3):
	"""
	Creates dynamical matrix by mass weighting hessian. Default output in turbomole format (hartree/bohr**2)
	Args:
		param1 (String) : Filename to hessian
		param2 (String) : Filename to coord file (xyz or turbomole format)
		param3 (boolean) : convert ouptput from turbomole (hartree/bohr**2) in SI units
		param3 (int) : number of dimensions

	Returns:
		np.ndarray 
	"""
	#read atoms from file
	atoms = list()
	#check if xyz or turbomoleformat is parsed
	if(fnmatch.fnmatch(filename_coord, '*.xyz')):
		datContent = load_xyz_file(filename_coord)[1][0]
		atoms.extend(datContent)
	else:
		datContent = read_coord_file(filename_coord)
		atoms = [str(datContent[i][3]).lower() for i in range(0,len(datContent))]
	#determine atom masses
	masses = list()
	for i in range(0,len(atoms)):
		if(t2SI==True):
			masses.append(atom_weight(atoms[i],u2kg=True))
		else:
			masses.append(atom_weight(atoms[i]))

	#create dynamical matrix by mass weighting hessian
	hessian = read_hessian(filename_hessian, len(atoms), dimensions)

	dynamical_matrix=np.zeros((len(atoms)*dimensions,len(atoms)*dimensions))
	for i in range(0,len(atoms)*dimensions):
		if(t2SI==True):
			#har/bohr**2 -> J/m**2
			hessian[i,i] = hessian[i,i]*(((1.89)**2)/(2.294))*1000
		dynamical_matrix[i,i]=(1./np.sqrt(masses[int(i/dimensions)]*masses[int(i/dimensions)]))*hessian[i,i]
		for j in range(0,i):
			#har/bohr**2 -> J/m**2
			if(t2SI==True):
				hessian[i,j] = hessian[i,j]*(((1.89)**2)/(2.294))*1000
			dynamical_matrix[i,j]=(1./np.sqrt(masses[int(i/dimensions)]*masses[int(j/dimensions)]))*hessian[i,j]
			dynamical_matrix[j,i]=dynamical_matrix[i,j]
	return dynamical_matrix


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
		coord: coord file read with top.read_coord_file
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

def atom_type_to_number(atom_type):
	"""
	returns ordinal number of atom

	Args:
		param1 (String): atom type		

	Returns:
		ordinal number (int)
	"""
	if(atom_type.lower() == "h"):
		return 1
	elif(atom_type.lower() == "c"):
		return 6
	elif(atom_type.lower() == "n"):
		return 7
	elif(atom_type.lower() == "o"):
		return 8
	elif(atom_type.lower() == "f"):
		return 9
	elif(atom_type.lower() == "cl"):
		return 17
	elif(atom_type.lower() == "br"):
		return 35
	elif(atom_type.lower() == "i"):
		return 53
	elif(atom_type.lower() == "au"):
		return 79
	elif(atom_type.lower() == "s"):
		return 16
	elif(atom_type.lower() == "si"):
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
	if(basis_set!="dev-SV(P)"):
		raise ValueError("Only support for dev-SV(P) Basis set right now.")

	coord = read_coord_file(coord_path)
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

def remove_fixed_atoms(coord):
	"""
	Removes fixed atoms from turbomole coord file. 
	Args:
		coord: coord file data loaded with read_coord_file

	Returns:
		coord: filtered coord file data
	"""
	coord_filtered = list()
	for i, item in enumerate(coord):
		if(len(item)<5):
			coord_filtered.append(item)
	return np.array(coord_filtered)

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
	coord = read_coord_file('coord')
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
		with open('basis') as fp:
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
			atomic_number = constants.ATOM_DICT_SYM[el[iat]][0]
			charge = atomic_number - int(Necp_dict[el[iat]])
			file.write(f"{iat+1:>8} {iorb + 1:>8} {iorb + Norb_dict[el[iat]]:>8} {charge:>24.12f} {Necp_dict[el[iat]]:>8} {pos[iat, 0]:>24} {pos[iat, 1]:>24} {pos[iat, 2]:>24}\n")
			iorb = iorb + Norb_dict[el[iat]]

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

	for i in range(0,len(orbital_types)):
		ns, config = config.split(orbital_types[i])
		ns = int(ns)
		number_of_orbital_types[i] = ns
		if config == '':
			break
	if (config != ''):
		raise ValueError(f"There is something wrong with this configuration")

	Norb = np.sum(number_of_orbital_types*orbital_degeneracy)
	return int(Norb)

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
		if len(coord_xyz[1,:]) == 0:
			raise ValueError("coord file is empty")
	except IndexError as e:
		raise ValueError(f"Is it a xyz file? {e}")
	except TypeError as e:
		raise ValueError(f"Is it a xyz file? {e}")


	for i in range(0,len(coord_xyz[1,:])):
		coord_xyz[1,i] = round(float(coord_xyz[1,i])+x_shift,5)
		coord_xyz[2,i] = round(float(coord_xyz[2,i])+y_shift,5)
		coord_xyz[3,i] = round(float(coord_xyz[3,i])+z_shift,5)
	return coord_xyz

def shift_coord_file(coord, x_shift, y_shift, z_shift):
	"""
	Shifts coordinates in coord file loaded with read_coord_file.
	Args:
		coord_xyz: Coord file loaded with read_coord_file
		x_shift: x shift in Bohr
		y_shift: y shift in Bohr
		z_shift: z shift in Bohr

	Returns:
		coord: Shifted coord file
	"""

	if len(coord) == 0:
		raise ValueError("coord file is empty")

	for i in range(0,len(coord)):
		coord[i][0] = round(float(coord[i][0])+x_shift,5)
		coord[i][1] = round(float(coord[i][1])+y_shift,5)
		coord[i][2] = round(float(coord[i][2])+z_shift,5)
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
	allowed = [0,1,2]
	if(axis not in allowed):
		raise ValueError(f"Axis {axis} not valid.")
	coord_xyz = np.transpose(coord_xyz)
	coord_xyz = coord_xyz[coord_xyz[:, axis+1].argsort()]
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
	allowed = [0,1,2]
	if(axis not in allowed):
		raise ValueError(f"Axis {axis} not valid.")

	coord = np.transpose(coord)
	coord = sorted(coord, key=lambda item: item[axis])
	coord = np.transpose(coord)

	return coord




	



