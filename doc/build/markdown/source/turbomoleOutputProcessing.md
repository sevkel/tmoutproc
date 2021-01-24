# turbomoleOutputProcessing package

## Submodules

## turbomoleOutputProcessing.setup module

## turbomoleOutputProcessing.turbomoleOutputProcessing module

Package for processing turbomole Output such as mos files


### turbomoleOutputProcessing.turbomoleOutputProcessing.build_permutation_matrix(input_list)
Builds permutation matrix to permute cols in eigenvector matrix (-> reordering). Usage: (matrix)\*permut or permut\*eigenvector


* **Parameters**

    **param1** (*list**(**int**)*) -- List of current order



* **Returns**

    permutation matrix (np.ndarray)



### turbomoleOutputProcessing.turbomoleOutputProcessing.calculate_A(filename, prefix_length=2, eigenvalue_source='mos', eigenvalue_path='', eigenvalue_col=2)
calculates the A matrix using mos file, reads eigenvalues from qpenergies (->DFT eigenvalues), input: mosfile and prefixlength (number of lines in the beginning), eigenvalue source (qpenergiesKS or qpenergiesGW),
if eigenvalue_path is specified you can give custom path to qpenergies and col to read
returns matrix in scipy.sparse.csc_matrix and eigenvalue list


* **Parameters**

    
    * **param1** (*string*) -- filename


    * **param2** (*int*) -- prefix_length (number of skipped lines in mos file)


    * **param3** (*string*) -- eigenvalue_source="mos" (choose eigenvalue source. If not changed: mos file. "qpenergiesKS" or "qpenergiesGW" : taken from ./data/qpenergies.dat)


    * **param4** (*string*) -- eigenvalue_path="" (if specified: eigenvalues are taken from qpenergies file with given path)


    * **param5** (*int*) -- eigenvalue_col (if eigenvalue_path is specified the eigenvalue_col in qpenergies is taken (KS=2, GW=3))



* **Returns**

    tuple A_i (scipy.sparse.csc_matrix) ,eigenvalue_list (list)



### turbomoleOutputProcessing.turbomoleOutputProcessing.calculate_F_i(i, eigenvalues, eigenvectors, sc_mat)
Calculates contribution of mo i to fmat
:param param2: i (mo which is considered)
:type param2: int
:param param2: eigenvalues
:type param2: list
:param param3: eigenvectors
:type param3: matrix
:param param4: s_mat\*c_mat
:type param4: matrix


* **Returns**

    F_i (F_i matrix (contribution of mo i to F_mat))



### turbomoleOutputProcessing.turbomoleOutputProcessing.calculate_F_splitted(eigenvalues, eigenvectors, s_mat_path)
Calculates contribution of every mo to fmat
:param param1: eigenvalues
:type param1: list
:param param2: eigenvectors
:type param2: matrix
:param param3: path to smat
:type param3: string


* **Returns**

    F_i (list of matrices)



### turbomoleOutputProcessing.turbomoleOutputProcessing.calculate_localization_mo(c, s_mat, left, right)
Calculates localization of mos 1 = c_l\*s_lr\*c_r + c_r\*s_lr\*c_l+c_l\*s_ll\*c_l + c_r\*s_rr\*c_r.
The last two summands describe the localization in left and richt part (defined as Q_l and Q_r).
The algorithm is not restricted to the left and right parts, but can be used to analyse the given
mos. Thus the L-C, C-R coupling and localization can be analysed, too.
:param param1: eigencectors
:type param1: matrix
:param param2: s matrix
:type param2: matrix
:param param3: left (matrix indices of left part start, matrix indices of left part end)
:type param3: (int, int)
:param param4: right (matrix indices of right part start, matrix indices of right part end)
:type param4: (int, int)


* **Returns**

    List of tuples (Q_l, Q_r) (Localization of each mo)



### turbomoleOutputProcessing.turbomoleOutputProcessing.calculate_localization_mo_i(i, eigenvectors, s_mat, left, center, right)
Calculates localization of mo 1 = c_l\*s_lr\*c_r + c_r\*s_lr\*c_l+c_l\*s_ll\*c_l + c_r\*s_rr\*c_r.
The last two summands describe the localization in left and richt part (defined as Q_l and Q_r) or any other
given part of mos.
:param param1: molecular oribital
:type param1: int
:param param2: eigenvectors
:type param2: ndarray
:param param3: smat
:type param3: ndarray
:param param4: left (matrix indices of left part start, matrix indices of left part end)
:type param4: tuple (int, int)
:param param5: center (matrix indices of center part start, matrix indices of center part end)
:type param5: tuple (int, int)
:param param6: right (matrix indices of right part start, matrix indices of right part end)
:type param6: tuple (int, int)


* **Returns**

    (ll,lc,lr,cl,cc,cr,rl,rc,rr) , tuple (float, float)



### turbomoleOutputProcessing.turbomoleOutputProcessing.diag_F(f_mat_path, s_mat_path, eigenvalue_list=[])
diagonalizes f mat (generalized), other eigenvalues can be used (eigenvalue_list). Solves Fx=l\*S\*x
:param param1: filename of fmat
:type param1: string
:param param2: filename of smat
:type param2: string
:param param3: eigenvalue_list (if specified eigenvalues are taken from eigenvalue_list)
:type param3: list()


* **Returns**

    fmat (np.ndarray), eigenvalues (np.array), eigenvectors (matrix)



### turbomoleOutputProcessing.turbomoleOutputProcessing.eformat(f, prec, exp_digits)

### turbomoleOutputProcessing.turbomoleOutputProcessing.find_c_range_atom(atom, number, coordfile, basis_set='dev-SV(P)')
finds atoms range of expansion coef in mos file. number gives wich atoms should be found e.g. two sulfur : first number =1; second = 2


* **Parameters**

    
    * **param1** (*String*) -- atom type


    * **param2** (*int*) -- number of atom type atom


    * **param3** (*String*) -- filename for coord file


    * **param4** (*String*) -- basis_set="dev-SV(P)"



* **Returns**

    tuple (index_start, index_end)



### turbomoleOutputProcessing.turbomoleOutputProcessing.load_xyz_file(filename)
load xyz file and return data. Returns comment line and coord data. Dat content cols: atoms=0, x=1, y=2, z=3

> param1 (String): filename


* **Returns**

    (comment_line (String), datContent (np.ndarray))



### turbomoleOutputProcessing.turbomoleOutputProcessing.measure_asymmetry(matrix)
gives a measure of asymmetry by calculatin max(

```
|matrix-matrix^T|
```

) (absvalue elementwise)


* **Parameters**

    **param1** (*np.ndarray*) -- Input Matrix



* **Returns**

    float



### turbomoleOutputProcessing.turbomoleOutputProcessing.measure_difference(matrix1, matrix2)
gives a measure of difference of two symmetric matrices (only upper triangular)


* **Parameters**

    
    * **param1** (*np.ndarray*) -- matrix1


    * **param2** (*np.ndarray*) -- matrix2



* **Returns**

    tuple of floats (min_error,max_error,avg_error,var_error)



### turbomoleOutputProcessing.turbomoleOutputProcessing.outer_product(vector1, vector2)
Calculates out product


* **Parameters**

    
    * **param1** (*np.array*) -- vector1


    * **param2** (*np.array*) -- vector2



* **Returns**

    np.ndarray



### turbomoleOutputProcessing.turbomoleOutputProcessing.read_coord_file(filename)
Reads data in file (eg plot data)


* **Parameters**

    **param1** (*String*) -- Filename



* **Returns**

    4=fixed]



* **Return type**

    array of lists [line in coordfile][0=x, 1=y, 2=z, 3=atomtype, optional



### turbomoleOutputProcessing.turbomoleOutputProcessing.read_mos_file(filename, skip_lines=1)
read mos file


* **Parameters**

    
    * **param1** (*string*) -- filename


    * **param2** (*int*) -- skip_lines=1 (lines to skip )



* **Returns**

    eigenvalue_list (list),eigenvector_list (np.ndarray)



### turbomoleOutputProcessing.turbomoleOutputProcessing.read_packed_matrix(filename)
read packed symmetric matrix from file and creates matrix
returns matrix in scipy.sparse.csc_matrix
:param param1: filename
:type param1: string


* **Returns**

    scipy.sparse.csc_matrix



### turbomoleOutputProcessing.turbomoleOutputProcessing.read_plot_data(filename)
Reads data in file (eg plot data)
:param param1: Filename
:type param1: String


* **Returns**

    datContent, head (Array of lists and header of file)



### turbomoleOutputProcessing.turbomoleOutputProcessing.read_qpenergies(filename, col=1, skip_lines=1)
read qpenergies.dat and returns data of given col as list, skip_lines in beginning of file, convert to eV
col = 0 qpenergies, col = 1 Kohn-Sham, 2 = GW


* **Parameters**

    
    * **param1** (*string*) -- filename


    * **param2** (*int*) -- col=2 (column to read)


    * **param3** (*int*) -- skip_lines=1 (lines without data at beginning of qpenergiesfile)



* **Returns**

    list



### turbomoleOutputProcessing.turbomoleOutputProcessing.trace_mo(ref_mos, input_mos, s_mat_path)
traces mos from input_mos with reference to ref_mos (eg when order has changed)
calculates the scalarproduct of input mos with ref_mos and takes the highest match (close to 1)


* **Parameters**

    
    * **param1** (*np.ndarray*) -- ref_mos


    * **param2** (*np.ndarray*) -- input_mos


    * **param3** (*string*) -- smat path



* **Returns**

    list (index with highest match)



### turbomoleOutputProcessing.turbomoleOutputProcessing.write_matrix_packed(matrix, filename='test')
write symmetric scipy.sparse.csc_matrix in  in packed storage form


* **Parameters**

    
    * **param1** (*scipi.sparse.csc_matrix*) -- input matrix


    * **param2** (*string*) -- filename


Returns:


### turbomoleOutputProcessing.turbomoleOutputProcessing.write_mos_file(eigenvalues, eigenvectors, filename='mos_new.dat')
write mos file, requires python3


* **Parameters**

    
    * **param1** (*np.array*) -- eigenvalues


    * **param2** (*np.ndarray*) -- eigenvectors


    * **param3** (*string*) -- filename="mos_new.dat"



### turbomoleOutputProcessing.turbomoleOutputProcessing.write_plot_data(filename, data, header='')
Writes data in file (eg plot data)
:param param1: Filename
:type param1: String
:param param2: tuple of lists (each entry is one column in data)
:type param2: tuple
:param param3: Fileheader
:type param3: String

Returns:


### turbomoleOutputProcessing.turbomoleOutputProcessing.write_xyz_file(filename, comment_line, datContent)
writes xyz file.


* **Parameters**

    
    * **param1** (*String*) -- filename


    * **param2** (*String*) -- commment_line


    * **param3** (*np.ndarray*) -- dat content


Returns:

## Module contents
