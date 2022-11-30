import tmoutproc as top
from tmoutproc import constants
import pytest
import numpy as np

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
    with pytest.raises(ValueError):
        top.find_c_range_atom(atom='au', number=1, coordfile=coordfile, basis_set="NOT VALID")
    assert top.find_c_range_atom(atom='au', number=1, coordfile=coordfile, basis_set="dev-SV(P)") == (0,25)
    assert top.find_c_range_atom(atom='au', number=2, coordfile=coordfile, basis_set="dev-SV(P)") == (25,50)
    assert top.find_c_range_atom(atom='s', number=2, coordfile=coordfile, basis_set="dev-SV(P)") == (529, 547)

def test_get_norb_from_config():
    orbital_degeneracy = [1, 3, 5, 7]
    assert top.get_norb_from_config("3s1p5d2f") == 3*orbital_degeneracy[0] + 1*orbital_degeneracy[1] + 5*orbital_degeneracy[2] + 2*orbital_degeneracy[3]
    assert top.get_norb_from_config("5s2p8d1f") == 5 * orbital_degeneracy[0] + 2 * orbital_degeneracy[1] + 8 * \
           orbital_degeneracy[2] + 1 * orbital_degeneracy[3]
    with pytest.raises(ValueError):
        top.get_norb_from_config("5s2p8d1fNOTVALID")

if __name__ == '__main__':
    test_find_c_range_atom()