import tmoutproc
import tmoutproc as top
import tmoutproc.constants as constants
import pytest
import numpy as np

def test_create_dynamical_matrix():
    #TODO: improve test qualitiy
    test_data_path= "./tests/test_data/"
    dyn_matrix = top.create_dynamical_matrix(f"{test_data_path}/hessian_xtb", f"{test_data_path}/coord_h2", t2SI=False, dimensions=3)
    h_weight = tmoutproc.atom_weight("h")
    assert np.isclose(dyn_matrix[0,0], 0.2680162066/h_weight)
    assert np.isclose(dyn_matrix[0, 1], -0.0000014025 / h_weight)
    assert np.max(np.abs(dyn_matrix-np.transpose(dyn_matrix))) == 0

    dyn_matrix = top.create_dynamical_matrix(f"{test_data_path}/hessian_xtb", f"{test_data_path}/coord_h2.xyz", t2SI=False, dimensions=3)
    h_weight = tmoutproc.atom_weight("h")
    assert np.isclose(dyn_matrix[0, 0], 0.2680162066 / h_weight)
    assert np.isclose(dyn_matrix[0, 1], -0.0000014025 / h_weight)
    assert np.max(np.abs(dyn_matrix - np.transpose(dyn_matrix))) == 0



def test_determine_n_orbitals():
    #test if basis set is checked
    with pytest.raises(ValueError):
        top.determine_n_orbitals("./tests/test_data/coord_PCP_filtered", basis_set="Not Valid")
    n_orbitals = top.determine_n_orbitals("./tests/test_data/coord_PCP_filtered")
    assert n_orbitals == 622

def test_atom_weight():
    assert top.atom_weight("test") == 1, "Test case not recognized"
    with pytest.raises(ValueError):
        top.atom_weight("NOT VALID")
    assert np.isclose(top.atom_weight("h"), 1.008)
    assert np.isclose(top.atom_weight("h", u2kg=True), 1.008*constants.U2KG)


def test_find_c_range_atom():
    atom = 'au'
    number = 1
    coordfile = "./tests/test_data/coord_PCP_filtered"
    coord = top.read_coord_file(coordfile)
    with pytest.raises(ValueError):
        top.find_c_range_atom(atom='au', number=1, coord=coord, basis_set="NOT VALID")
    assert top.find_c_range_atom(atom='au', number=1, coord=coord, basis_set="dev-SV(P)") == (0,25)
    assert top.find_c_range_atom(atom='au', number=2, coord=coord, basis_set="dev-SV(P)") == (25,50)
    assert top.find_c_range_atom(atom='s', number=2, coord=coord, basis_set="dev-SV(P)") == (529, 547)

def test_get_norb_from_config():
    orbital_degeneracy = [1, 3, 5, 7]
    assert top.get_norb_from_config("3s1p5d2f") == 3*orbital_degeneracy[0] + 1*orbital_degeneracy[1] + 5*orbital_degeneracy[2] + 2*orbital_degeneracy[3]
    assert top.get_norb_from_config("5s2p8d1f") == 5 * orbital_degeneracy[0] + 2 * orbital_degeneracy[1] + 8 * \
           orbital_degeneracy[2] + 1 * orbital_degeneracy[3]
    with pytest.raises(ValueError):
        top.get_norb_from_config("5s2p8d1fNOTVALID")

def atom_type_to_number():
    Z = top.atom_type_to_atomic_number("Au")
    assert Z == 79
    Z = top.atom_type_to_atomic_number("au")
    assert Z == 79

    with pytest.raises(ValueError):
        top.atom_type_to_atomic_number("NOTVALID")

def test_atom_type_to_atomic_number():
    assert top.atom_type_to_atomic_number("h") == 1
    assert top.atom_type_to_atomic_number("hf") == 72
    with pytest.raises(ValueError):
        top.atom_type_to_atomic_number("NotValid")