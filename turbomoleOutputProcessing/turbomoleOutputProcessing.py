"""
Package for processing turbomole Output such as mos files
"""


import numpy as np
import math
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
import scipy.sparse
import re as r
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.linalg import eig
from functools import partial
from multiprocessing import Pool




def outer_product(vector1, vector2):
	"""
	Calculates out product

	Args:
		param1 (np.array) : vector1
		param2 (np.array) : vector2

	Returns:
		np.ndarray

	"""
	return np.outer(vector1,vector2)


def eformat(f, prec, exp_digits):
	
    s = "%.*e"%(prec, f)
	#s ="hallo"
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%sE%+0*d"%(mantissa, exp_digits+1, int(exp))


def measure_asymmetry(matrix):	
	""" 
	gives a measure of asymmetry by calculatin max(|matrix-matrix^T|) (absvalue elementwise)

	Args:
		param1 (np.ndarray) : Input Matrix

	Returns:
		float

	"""	

	return np.max(np.abs(matrix-np.transpose(matrix)))

 
def measure_difference(matrix1, matrix2):
	"""
	gives a measure of difference of two symmetric matrices (only upper triangular)

	Args:
		param1 (np.ndarray) : matrix1
		param2 (np.ndarray) : matrix2

	Returns: 
		tuple of floats (min_error,max_error,avg_error,var_error)
	"""
	diff = (np.abs(matrix1-matrix2))
	min_error = np.min(diff)
	print("min "  + str(min_error))
	max_error = np.max(diff)
	print("max "  + str(max_error))
	avg_error = np.mean(diff[np.triu_indices(diff.shape[0])])
	print("avg "  + str(avg_error))
	var_error = np.var(diff[np.triu_indices(diff.shape[0])])
	print("var "  + str(var_error))
	return min_error,max_error,avg_error,var_error





def calculate_A(filename,prefix_length=2, eigenvalue_source = "mos", eigenvalue_path = "", eigenvalue_col = 2):
	"""
	calculates the A matrix using mos file, reads eigenvalues from qpenergies (->DFT eigenvalues), input: mosfile and prefixlength (number of lines in the beginning), eigenvalue source (qpenergiesKS or qpenergiesGW), 
	if eigenvalue_path is specified you can give custom path to qpenergies and col to read
	returns matrix in scipy.sparse.csc_matrix and eigenvalue list

	Args:
		param1 (string) : filename
		param2 (int) : prefix_length (number of skipped lines in mos file)
		param3 (string) : eigenvalue_source="mos" (choose eigenvalue source. If not changed: mos file. "qpenergiesKS" or "qpenergiesGW" : taken from ./data/qpenergies.dat)
		param4 (string) : eigenvalue_path="" (if specified: eigenvalues are taken from qpenergies file with given path)
		param5 (int) : eigenvalue_col (if eigenvalue_path is specified the eigenvalue_col in qpenergies is taken (KS=2, GW=3))

	Returns:
		tuple A_i (scipy.sparse.csc_matrix) ,eigenvalue_list (list)
	"""
	#length of each number
	n = 20
	level = 0
	C_vec = list()
	A_i = 0
	counter = 0
	eigenvalue_old = -1
	eigenvalue = 0
	beginning = True
	eigenvalue_list= list()
	#if eigenvalue source = qpEnergies
	if(eigenvalue_source == "qpenergiesKS" and eigenvalue_path == ""):
		eigenvalue_list = read_qpenergies("qpenergies.dat", col = 1)
	elif(eigenvalue_source == "qpenergiesGW" and eigenvalue_path == ""):
		eigenvalue_list = read_qpenergies("qpenergies.dat", col = 2)
	elif(eigenvalue_path != ""):
		print("custom path ")
		eigenvalue_list = read_qpenergies(eigenvalue_path, col = eigenvalue_col)
		#print(eigenvalue_list)


	#open mos file and read linewise
	with open(filename) as f:
		for line in f:
			#skip prefix lines
			#find level and calculate A(i)
			if(counter>prefix_length):
				index1 = (line.find("eigenvalue="))
				index2 = (line.find("nsaos="))
				#eigenvalue and nsaos found -> new orbital
				if(index1 != -1 and index2 != -1):
					level += 1 
					
					#find eigenvalue of new orbital and store eigenvalue of current orbital in eigenvalue_old
					if((beginning == True and eigenvalue_source == "mos") and eigenvalue_path == ""):
						eigenvalue_old = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))						
						beginning = False
					elif(eigenvalue_source == "mos" and eigenvalue_path == ""):
						#print("eigenvalue from mos")
						eigenvalue_old = eigenvalue
						eigenvalue_list.append(eigenvalue_old)
						#if(level != 1128):
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
					#if eigenvalues are taken from qpenergies
					elif(eigenvalue_source == "qpenergiesKS" or eigenvalue_source == "qpenergiesGW" or eigenvalue_path != ""):
						if(beginning == True):
							beginning = False
							pass
						else:
							#print("eigenvalue from eigenvaluelist")
							eigenvalue_old = eigenvalue_list[level-2]

												 
					#find nsaos of new orbital
					nsaos = (int(line[(index2+len("nsaos=")):len(line)]))

					#create empty A_i matrix
					if(isinstance(A_i,int)):
						A_i = np.zeros((nsaos,nsaos))						
						

					#calculate A matrix by adding A_i -> processing of previous orbital		
					if(len(C_vec)>0):			
						print("take " + str(eigenvalue_old))						
						A_i += np.multiply(outer_product(C_vec,C_vec),eigenvalue_old)						
						C_vec = list()					
					#everything finished (beginning of new orbital)
					continue
				
				
				line_split = [line[i:i+n] for i in range(0, len(line), n)]							
				C_vec.extend([float(line_split[j].replace("D","E"))  for j in range(0,len(line_split)-1)][0:4])
				#print(len(C_vec))
				
			counter += 1
			#for testing
			if(counter > 300):
				#break
				pass
	#handle last mos
	if(eigenvalue_source == "qpenergiesKS" or eigenvalue_source == "qpenergiesGW"):
		eigenvalue_old = eigenvalue_list[level-1]
		print("lasteigen " + str(eigenvalue_old))
	
	A_i += np.multiply(outer_product(C_vec,C_vec),eigenvalue_old)

	A_i = scipy.sparse.csc_matrix(A_i, dtype = float)
	print("A_mat symmetric: " + str(np.allclose(A_i.toarray(), A_i.toarray(), 1e-04, 1e-04)))
	print("A_mat asymmetry measure " + str(measure_asymmetry(A_i.toarray())))
	print("------------------------")	
	return A_i,eigenvalue_list


def read_packed_matrix(filename):
	"""
	read packed symmetric matrix from file and creates matrix
	returns matrix in scipy.sparse.csc_matrix
	Args:
		param1 (string) : filename

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
			#line_split = [i.split() for i in line]
			line_split = r.split('\s+', line)
			#print(counter)
			#print(line_split[2])

			matrix_entry = np.float(line_split[2])
			#print(eformat(matrix_entry,100,5))
			#print(line_split[2])
			#print(eformat(np.longdouble(line_split[2]),100,5))
			#print("-----")
			#matrix_entry = round(matrix_entry,25)
			matrix_enty = round(float(line_split[2]),24)

			

			#calculate row and col
			
			if(col == col_old+1):
				col_old = col
				col = 0;
				row += 1
			#print("setting up row " + str(counter))
			#print(row,col)
			#skip zero elemts
			if(matrix_entry != 0.0):
				data.append(matrix_entry)
				#print(matrix_entry)
				#print("------")
				i.append(col)
				j.append(row)
				#symmetrize matrix
				if(col != row):
					data.append(matrix_entry)
					i.append(row)
					j.append(col)
					pass
			col += 1	

			
			counter+=1
			#for testing
			if(counter>25):
				#break
				pass
	
	coo = coo_matrix((data, (i, j)), shape=(row+1, row+1))
	csc = scipy.sparse.csc_matrix(coo, dtype = float)	
	
	return(csc)




def write_matrix_packed(matrix, filename="test"):
	"""
	write symmetric scipy.sparse.csc_matrix in  in packed storage form

	Args:
		param1 (scipi.sparse.csc_matrix) : input matrix
		param2 (string) : filename

	Returns:

	"""
	print("writing packed matrix")
	num_rows = matrix.shape[0]
	num_elements_to_write = (num_rows**2+num_rows)/2
	

	col = 0
	row = 0
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
		param1 (string) : filename
		param2 (int) : col=2 (column to read)
		param3 (int) : skip_lines=1 (lines without data at beginning of qpenergiesfile)

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
		pass	
	return qpenergies






def write_mos_file(eigenvalues, eigenvectors, filename="mos_new.dat"):
	"""
	write mos file, requires python3

	Args:
		param1 (np.array) : eigenvalues
		param2 (np.ndarray) : eigenvectors
		param3 (string) : filename="mos_new.dat"
	"""
	f = open(filename, "w")
	#header
	f.write("$scfmo    scfconv=7   format(4d20.14)\n")
	f.write("# SCF total energy is    -6459.0496515472 a.u.\n") 
	f.write("#\n")   
	for i in range(0, len(eigenvalues)):
		print("eigenvalue " + str(eigenvalues[i]) + "\n")
		first_string = ' ' * (6-len(str(i))) + str(i+1) + "  a      eigenvalue=" + eformat(eigenvalues[i], 14,2) + "   nsaos=" + str(len(eigenvalues))
		f.write(first_string + "\n")
		j = 0
		while j<len(eigenvalues):
			for m in range(0,4):
				num = eigenvectors[m+j,i]
				#string_to_write = f"{num:+20.13E}".replace("E", "D")
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
		param1 (string) : filename
		param2 (int) : skip_lines=1 (lines to skip )

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
				C_vec.extend([float(line_split[j].replace("D","E"))  for j in range(0,len(line_split)-1)][0:4])
				#print(len(C_vec))
				
			counter += 1
			
	#handle last mos
	eigenvalue_list.append(eigenvalue_old)	
	eigenvector_list[:,(nsaos-1)] = C_vec
	
	return eigenvalue_list,eigenvector_list



			
def trace_mo(ref_mos, input_mos, s_mat_path, tolerance=-1, num_cpu = 8):	
	"""
	traces mos from input_mos with reference to ref_mos (eg when order has changed) 
	calculates the scalarproduct of input mos with ref_mos and takes the highest match (close to 1)

	Args:
		param1 (np.ndarray) : ref_mos
		param2 (np.ndarray) : input_mos
		param3 (string) : smat path
		param4 (int) : tolerance = -1 (if specified a maximum shift of mo index (+-tolerance) is assumed -> saves computation time)
		param4 (int) : num_cpu (number of cores for calculation)

	Returns:
		list (index with highest match)
	"""
	input_mos_list = list()
	#prepare
	for i in range(0, ref_mos.shape[0]):
		input_mos_list.append(input_mos[:,i])
	
	print("filling pool")
	p = Pool(num_cpu)	
	s_mat = read_packed_matrix(s_mat_path).todense()
	para = (ref_mos, input_mos, tolerance, s_mat)

	#all eigenvectors

	#index_to_check = range(0, input_mos.shape[0])
	index_to_check = np.arange(0,input_mos.shape[0],1)
	result = p.map(partial(scalarproduct, para=para), index_to_check)		
	print("done")
	return result

def trace_mo(ref_mos, input_mos, s_mat_path):	
	"""
	traces mos from input_mos with reference to ref_mos (eg when order has changed) 
	calculates the scalarproduct of input mos with ref_mos and takes the highest match (close to 1)

	Args:
		param1 (np.ndarray) : ref_mos
		param2 (np.ndarray) : input_mos
		param3 (string) : smat path		

	Returns:
		list (index with highest match)
	"""
		
	s_mat = read_packed_matrix(s_mat_path).todense()
	
	scalar_product_matrix = np.dot(s_mat,np.asarray(input_mos))
	scalar_product_matrix = np.dot(np.transpose(ref_mos),scalar_product_matrix)
	scalar_product_matrix = np.abs(scalar_product_matrix)
	most_promising = list()
	for i in range(0, scalar_product_matrix.shape[0]):
		most_promising.append(np.where(scalar_product_matrix[:,i] == scalar_product_matrix[:,i].max())[0][0])
		#print(scalar_product_matrix[most_promising[i],i])
		
	#all eigenvectors


	return most_promising




def diag_F(f_mat_path, s_mat_path, eigenvalue_list = list()):
	"""
	diagonalizes f mat (generalized), other eigenvalues can be used (eigenvalue_list). Solves Fx=l*S*x
	Args:
		param1 (string) : filename of fmat
		param2 (string) : filename of smat
		param3 (list()) : eigenvalue_list (if specified eigenvalues are taken from eigenvalue_list)

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

	sc = s_mat * eigenvectors
	f_mat = eigenvalues * np.transpose(sc)
	f_mat = sc * f_mat
	print("calc fmat done")
	


	return f_mat,eigenvalues, eigenvectors
	

def calculate_F_i(i, eigenvalues, eigenvectors, sc_mat):
	"""
	Calculates contribution of mo i to fmat
	Args:
		param2 (int) : i (mo which is considered)
		param2 (list) : eigenvalues
		param3 (matrix) : eigenvectors
		param4 (matrix) : s_mat*c_mat

	Returns:
		F_i (F_i matrix (contribution of mo i to F_mat))

	"""
	#sc_mat = np.dot(sc_mat,np.asarray(eigenvectors))
	#sc_mat = 
	F_i = np.multiply(float(eigenvalues[i,i]) , outer_product(sc_mat[:,i], sc_mat[:,i]))
	#F_i = np.dot(sc_mat,F_i)
	#F_i = np.dot(F_i, sc_mat)
	return F_i

def calculate_F_splitted(eigenvalues, eigenvectors, s_mat_path):
	"""
	Calculates contribution of every mo to fmat 
	Args:
		param1 (list) : eigenvalues
		param2 (matrix) : eigenvectors
		param3 (string) : path to smat

	Returns:
		F_i (list of matrices)

	"""
	s_mat = read_packed_matrix(s_mat_path).todense()
	sc_mat = s_mat * np.asmatrix(eigenvectors)
	F_i = list()
	for i in range(0, len(eigenvectors)):
		F_i.append(calculate_F_i(i, eigenvalues, eigenvectors, sc_mat))

	return F_i

def calculate_localization_mo_i(c_l, c_r, s_ll, s_lr, s_rr ):
	"""
	Calculates localization of mo 1 = c_l*s_lr*c_r + c_r*s_lr*c_l+c_l*s_ll*c_l + c_r*s_rr*c_r.
	The last two summands describe the localization in left and richt part (defined as Q_l and Q_r) or any other
	given part of mos.
	Args:
		param1 (array) : mo_left
		param2 (array) : mo_right
		param3 (matrix) : s_ll
		param4 (matrix) : s_lr
		param5 (matrix) :  s_rr 

	Returns:
		(Q_l, Q_r) , tuple (float, float)
	"""

	Q_l = s_ll * c_l
	Q_l = float(np.transpose(c_l) * Q_l)

	Q_r = s_rr * c_r
	Q_r = float(np.transpose(c_r) * Q_r)

	Q_lr = s_lr * c_r
	Q_lr = float(np.transpose(c_l) * Q_lr)

	Q_rl = np.transpose(s_lr) * c_l
	Q_rl = float(np.transpose(c_r) * Q_rl)

	return (Q_l, Q_r, Q_lr, Q_rl)

	
def calculate_localization_mo(c, s_mat, left, right):
	"""
	Calculates localization of mos 1 = c_l*s_lr*c_r + c_r*s_lr*c_l+c_l*s_ll*c_l + c_r*s_rr*c_r.
	The last two summands describe the localization in left and richt part (defined as Q_l and Q_r).
	The algorithm is not restricted to the left and right parts, but can be used to analyse the given
	mos. Thus the L-C, C-R coupling and localization can be analysed, too. 
	Args:
		param1 (matrix) : eigencectors
		param2 (matrix) : s matrix
		param3 ((int, int)) : left (matrix indices of left part start, matrix indices of left part end)
		param4 ((int, int)) : right (matrix indices of right part start, matrix indices of right part end)
		

	Returns:
		List of tuples (Q_l, Q_r) (Localization of each mo)
	"""

	localization = list()
	for i in range(0, c.shape[0]):
		s_ll= s_mat[left[0]:left[1], left[0]:left[1]]
		s_lr = s_mat[left[0]:left[1], right[0]:right[1]]
		s_rr = s_mat[right[0]:right[1], right[0]:right[1]]
		Q_l, Q_r, Q_lr, Q_rl = calculate_localization_mo_i(c[left[0]:left[1],i], c[right[0]:right[1], i], s_ll, s_lr, s_rr)
		
		localization.append((Q_l,Q_r))

	return localization





