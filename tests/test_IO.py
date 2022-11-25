from turbomoleOutputProcessing import turbomoleOutputProcessing as top
import pytest
import numpy as np

def test_load_xyz_file():
    #problem not acutal code is used but installed code
    header, coord_xyz = top.load_xyz_file("./tests/test_data/benz.xyz")

    assert header[0] == "Header"
    assert coord_xyz.shape == (4,12)
    assert coord_xyz[0, 0] == "C"
    assert float(coord_xyz[1, 0]) == -2.97431
    assert coord_xyz[0, 6] == "H"

def test_determine_n_orbitals():
    #test if basis set is checked
    with pytest.raises(ValueError):
        top.determine_n_orbitals("./tests/test_data/coord_PCP_filtered", basis_set="Not Valid")
    n_orbitals = top.determine_n_orbitals("./tests/test_data/coord_PCP_filtered")
    assert n_orbitals == 622

def test_remove_fixed_atoms():
    coord_PCP = top.read_coord_file("./tests/test_data/coord_PCP")
    coord_PCP_filtered = top.remove_fixed_atoms(coord_PCP)
    assert len(coord_PCP) == 90, "Error in read_coord_file"
    assert len(coord_PCP_filtered) == 58
    assert coord_PCP_filtered[0][3] == 'au', "This element must be au"
    assert float(coord_PCP_filtered [0][0]) == -0.03330929466830, "Wrong value in coord_PCP_filtered"

def test_shift_xyz_coord():
    coord_xyz = top.load_xyz_file("./tests/test_data/benz.xyz")[1]
    coord = top.read_coord_file("./tests/test_data/coord_PCP")

    with pytest.raises(ValueError):
        top.shift_xyz_coord(coord, 0,0,0)

    coord_xyz_shifted = top.shift_xyz_coord(coord_xyz, 1.0 ,2.0 ,3.0)
    assert np.isclose(float(coord_xyz_shifted[1, 0]),-2.97431+1.0)
    assert np.isclose(float(coord_xyz_shifted[2, 0]),0.42856 + 2.0)
    assert np.isclose(float(coord_xyz_shifted[3, 0]),-0.00000 + 3.0)

    assert np.isclose(float(coord_xyz_shifted[1, 11]), -4.08087  + 1.0)
    assert np.isclose(float(coord_xyz_shifted[2, 11]), -2.79039 + 2.0)
    assert np.isclose(float(coord_xyz_shifted[3, 11]), -0.00000 + 3.0)

def test_shift_coord_file():

    coord = top.read_coord_file("./tests/test_data/coord_PCP")

    coord_shifted = top.shift_coord_file(coord, 1.0 ,2.0 ,3.0)
    assert np.isclose(float(coord_shifted[0][0]),-8.17778980431802+1.0)
    assert np.isclose(float(coord_shifted[0][1]),-4.72144292785499 + 2.0)
    assert np.isclose(float(coord_shifted[0][2]),-12.84405276595290 + 3.0)

    assert np.isclose(float(coord_shifted[89][0]), -4.84905613305097  + 1.0)
    assert np.isclose(float(coord_shifted[89][1]), -0.90619926580304 + 2.0)
    assert np.isclose(float(coord_shifted[89][2]), 39.80494546134004 + 3.0)

if __name__ == '__main__':
    test_load_xyz_file()
    test_determine_n_orbitals()