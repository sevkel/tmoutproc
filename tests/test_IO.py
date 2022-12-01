import tmoutproc as top
import pytest
import numpy as np

def test_load_xyz_file():
    #problem not acutal code is used but installed code
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")

    assert coord_xyz.shape == (4,12)
    assert coord_xyz[0, 0] == "C"
    assert float(coord_xyz[1, 0]) == -2.97431
    assert coord_xyz[0, 6] == "H"
    coord_xyz, header = top.read_xyz_file("./tests/test_data/benz.xyz", return_header=True)
    assert header == "Header eins"

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
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")
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

def test_sort_xyz_coord():
    coord_xyz = top.read_xyz_file("./tests/test_data/benz.xyz")
    with pytest.raises(ValueError):
        top.sort_xyz_coord(coord_xyz, axis=3)
    axes = [2,1,0]
    # test sorting
    for axis in axes:
        coord_sorted = top.sort_xyz_coord(coord_xyz, axis=axis)
        assert coord_sorted.shape == coord_xyz.shape
        assert all(coord_sorted[axis+1, i] <= coord_sorted[axis+1, i+1] for i in range(len(coord_sorted[:, axis+1]) - 1))
    # test if elements are correct
    assert coord_sorted[0, 0] == "H"
    assert coord_sorted[1, 0] == -6.35080
    assert coord_sorted[2, 0] == 0.85915
    assert coord_sorted[3, 0] == 0.00000

def test_sort_coord():
    coord = top.read_coord_file("./tests/test_data/coord_PCP")
    with pytest.raises(ValueError):
        top.sort_xyz_coord(coord, axis=3)
    axes = [2,0,1]
    for axis in axes:
        coord_sorted = top.sort_coord(coord, axis)
        assert len(coord_sorted) == len(coord)
        assert all(coord_sorted[i][axis] <= coord_sorted[i+1][axis] for i in range(len(coord_sorted) - 1))

    assert coord_sorted[0][0] == 0.60280373649437
    assert coord_sorted[0][1] == -10.34910401877428
    assert coord_sorted[0][2] == 39.80494546134004
    assert coord_sorted[0][3] == 'au'
    assert coord_sorted[0][4] == 'f'

def test_read_hessian():
    #"""
    hessians = ["hessian_tm", "hessian_xtb"]
    for hessian in hessians:
        with pytest.raises(ValueError):
            top.read_hessian(f"./tests/test_data/{hessian}", n_atoms=1, dimensions=3)
        with pytest.raises(ValueError):
            top.read_hessian(f"./tests/test_data/{hessian}", n_atoms=3, dimensions=3)

        hessian = top.read_hessian(f"./tests/test_data/{hessian}", n_atoms=2, dimensions=3)
        assert np.max(np.abs(hessian-np.transpose(hessian))) == 0
        assert hessian[0,0]

def test_read_symmetric_from_triangular():
    hessian = top.read_symmetric_from_triangular("./tests/test_data/hessian_direct")
    #symmetry
    assert np.max(hessian-np.transpose(hessian)) == 0
    assert hessian.shape == (3*3, 3*3)
    assert hessian[0,0] == 3.806638367204165E-002
    assert hessian[8,8] == 0.235824323786876
    assert hessian[7,8] == -0.222050038511708
    assert hessian[6,8] == -0.112940116315714
    top.write_symmetric_to_triangular(hessian, "/tmp/hessian")
    hessian_reread = top.read_symmetric_from_triangular("/tmp/hessian")
    assert np.max(np.abs(hessian_reread-hessian)) == 0

if __name__ == '__main__':
    test_load_xyz_file()
    test_determine_n_orbitals()